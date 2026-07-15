from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_runtime import ContinuousBatchScheduler, SchedulingMetadata  # noqa: E402
from inference_runtime.adapters import LlamaCppSteppableBackend  # noqa: E402
from model_worker.manifest import load_manifest  # noqa: E402
from model_worker.preflight import preflight  # noqa: E402
from scripts.benchmark_inference_runtime import (  # noqa: E402
    git_identity,
    gpu_identity,
    percentile,
    sha256,
)
from scripts.monitor_resources import (  # noqa: E402
    PROCESS_TREE_SCOPE,
    TOTAL_SYSTEM_FALLBACK_SCOPE,
    collect_sample,
)


LONG_SOAK_SECONDS = 900.0
LONG_SOAK_REQUESTS = 500
SOAK_EXPECTED_VALUES = (
    "soak-alpha",
    "soak-bravo",
    "soak-charlie",
    "soak-delta",
    "soak-echo",
    "soak-foxtrot",
    "soak-golf",
    "soak-hotel",
)


class NullSink:
    def publish(self, event: object) -> None:
        del event


def body(model_id: str, expected: str) -> dict[str, Any]:
    return {
        "protocol_version": "model-worker.v1",
        "model_id": model_id,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return a JSON object whose result field is exactly "
                    f"{expected}."
                ),
            }
        ],
        "output_contract": {
            "version": "structured-output.v1",
            "schema": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            },
            "instructions": f"The result field must be exactly {expected}.",
        },
        "limits": {
            "reasoning_tokens": 256,
            "final_tokens": 64,
            "total_tokens": 320,
            "queue_timeout_ms": 30000,
            "execution_timeout_ms": 120000,
        },
        "stream": {"enabled": False, "include_reasoning": False},
    }


def metadata(request_id: str) -> SchedulingMetadata:
    return SchedulingMetadata(
        request_id,
        "runtime-soak",
        f"agent-{request_id}",
        "throughput",
        1,
        None,
    )


def soak_expected(slot_index: int) -> str:
    if not 0 <= slot_index < len(SOAK_EXPECTED_VALUES):
        raise ValueError("soak slot index is outside the supported 2..8 concurrency")
    return SOAK_EXPECTED_VALUES[slot_index]


def linear_slope_per_hour(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    origin = points[0][0]
    x = [point[0] - origin for point in points]
    y = [point[1] for point in points]
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator == 0:
        return None
    slope_per_second = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x, y, strict=True)
    ) / denominator
    return slope_per_second * 3600.0


def stability_report(
    samples: list[dict[str, Any]],
    *,
    field: str,
    required_scope_field: str | None = None,
    required_scope: str | None = None,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    points = []
    for sample in samples:
        if (
            required_scope_field is not None
            and sample.get(required_scope_field) != required_scope
        ):
            continue
        value = sample.get(field)
        timestamp = sample.get("timestamp_unix")
        if (
            type(value) in {int, float}
            and type(timestamp) in {int, float}
            and math.isfinite(value)
            and math.isfinite(timestamp)
        ):
            points.append((float(timestamp), float(value)))
    if len(points) < 5:
        return {
            "available": False,
            "samples": len(points),
            "passed": False,
            "reason": "insufficient_process_scoped_samples",
        }
    stable = points[max(1, len(points) // 5) :]
    window = max(1, len(stable) // 5)
    baseline = statistics.median(value for _, value in stable[:window])
    ending = statistics.median(value for _, value in stable[-window:])
    drift = ending - baseline
    tolerance = max(absolute_tolerance, baseline * relative_tolerance)
    return {
        "available": True,
        "samples": len(points),
        "baseline": baseline,
        "ending": ending,
        "minimum": min(value for _, value in points),
        "maximum": max(value for _, value in points),
        "drift": drift,
        "allowed_positive_drift": tolerance,
        "slope_per_hour": linear_slope_per_hour(stable),
        "passed": drift <= tolerance,
        "reason": None if drift <= tolerance else "positive_drift_exceeded",
    }


def resource_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rss = stability_report(
        samples,
        field="process_tree_rss_bytes",
        absolute_tolerance=128 * 1024 * 1024,
        relative_tolerance=0.05,
    )
    process_vram = stability_report(
        samples,
        field="gpu_vram_mib",
        required_scope_field="gpu_vram_scope",
        required_scope=PROCESS_TREE_SCOPE,
        absolute_tolerance=128.0,
        relative_tolerance=0.02,
    )
    total_vram_fallback = stability_report(
        samples,
        field="gpu_vram_mib",
        required_scope_field="gpu_vram_scope",
        required_scope=TOTAL_SYSTEM_FALLBACK_SCOPE,
        absolute_tolerance=256.0,
        relative_tolerance=0.03,
    )
    selected_vram = (
        process_vram if process_vram["available"] else total_vram_fallback
    )
    selected_scope = (
        PROCESS_TREE_SCOPE
        if process_vram["available"]
        else TOTAL_SYSTEM_FALLBACK_SCOPE
    )
    return {
        "sample_count": len(samples),
        "rss_process_tree": rss,
        "vram_process_tree": process_vram,
        "vram_total_system_fallback": total_vram_fallback,
        "selected_vram_scope": selected_scope,
        "passed": rss["passed"] and selected_vram["passed"],
        "samples": samples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-executable", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=LONG_SOAK_SECONDS)
    parser.add_argument("--minimum-requests", type=int, default=LONG_SOAK_REQUESTS)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.sample_interval <= 0:
        parser.error("durations must be positive")
    if args.minimum_requests <= 0:
        parser.error("--minimum-requests must be positive")
    if args.concurrency < 2 or args.concurrency > 8:
        parser.error("--concurrency must be in 2..8")
    return args


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.model_manifest)
    backend = LlamaCppSteppableBackend(
        args.runtime_executable,
        args.runtime_manifest,
        startup_timeout=180,
        command_timeout=180,
    )
    scheduler = ContinuousBatchScheduler(backend, tick_token_budget=512)
    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    durations: list[float] = []
    completed = 0
    sampled_tokens = 0
    monitor_stop = threading.Event()
    monitor_thread: threading.Thread | None = None
    started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc)
    initial_generation = 0
    try:
        backend.start()
        scheduler.start()
        native_pid = backend.process_id
        if native_pid is None:
            raise RuntimeError("native runtime PID unavailable after readiness")
        initial_generation = backend.process_generation

        def monitor() -> None:
            while not monitor_stop.is_set():
                samples.append(collect_sample(native_pid))
                monitor_stop.wait(args.sample_interval)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        wave = 0
        while (
            time.monotonic() - started_monotonic < args.duration_seconds
            or completed < args.minimum_requests
        ):
            results: dict[int, Any] = {}
            errors: dict[int, BaseException] = {}

            def invoke(index: int) -> None:
                request_index = wave * args.concurrency + index
                expected = soak_expected(index)
                request_started = time.perf_counter()
                try:
                    result = scheduler.infer(
                        preflight(body(manifest.id, expected), manifest),
                        scheduling=metadata(f"soak-{request_index}"),
                        events=NullSink(),
                    )
                    if result.output != {"result": expected}:
                        raise RuntimeError("semantic output mismatch")
                    results[index] = result
                    durations.append(time.perf_counter() - request_started)
                except BaseException as exc:
                    errors[index] = exc

            threads = [
                threading.Thread(target=invoke, args=(index,))
                for index in range(args.concurrency)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(180)
            alive = [index for index, thread in enumerate(threads) if thread.is_alive()]
            if alive:
                failures.append({"wave": wave, "error": "thread_timeout", "indices": alive})
                break
            if errors:
                failures.extend(
                    {
                        "wave": wave,
                        "index": index,
                        "exception": type(exc).__name__,
                        "detail": str(exc),
                    }
                    for index, exc in sorted(errors.items())
                )
                break
            completed += len(results)
            sampled_tokens += sum(
                result.usage["sampled_tokens"] for result in results.values()
            )
            wave += 1
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(max(2.0, args.sample_interval + 1.0))
        shutdown_clean = scheduler.shutdown(timeout=20)

    elapsed = time.monotonic() - started_monotonic
    resources = resource_report(samples)
    generation_stable = (
        initial_generation > 0
        and backend.process_generation == initial_generation
    )
    workload_passed = not failures and completed >= args.minimum_requests
    requested_duration_met = elapsed >= args.duration_seconds
    long_scale_met = elapsed >= LONG_SOAK_SECONDS and completed >= LONG_SOAK_REQUESTS
    gate_passed = (
        workload_passed
        and requested_duration_met
        and long_scale_met
        and resources["passed"]
        and generation_stable
        and shutdown_clean
    )
    artifact = {
        "artifact_version": "inference-runtime-soak.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at.isoformat(),
        "identity": {
            "git": git_identity(ROOT),
            "gpu": gpu_identity(),
            "model_manifest": str(args.model_manifest.resolve()),
            "model_manifest_file_sha256": sha256(args.model_manifest),
            "model_manifest_digest": manifest.digest,
            "runtime_manifest": str(args.runtime_manifest.resolve()),
            "runtime_manifest_file_sha256": sha256(args.runtime_manifest),
            "runtime_executable": str(args.runtime_executable.resolve()),
            "runtime_executable_sha256": sha256(args.runtime_executable),
            "soak_source_sha256": sha256(Path(__file__).resolve()),
            "resource_monitor_source_sha256": sha256(
                ROOT / "scripts" / "monitor_resources.py"
            ),
            "scheduler_source_sha256": sha256(
                ROOT / "inference_runtime" / "continuous_scheduler.py"
            ),
            "governance_source_sha256": sha256(
                ROOT / "inference_runtime" / "governance.py"
            ),
            "adapter_source_sha256": sha256(
                ROOT / "inference_runtime" / "adapters" / "llama_cpp.py"
            ),
            "native_source_sha256": sha256(
                ROOT / "native" / "inference_runtime_main.cpp"
            ),
            "runtime_identity": backend.runtime_identity,
        },
        "workload": {
            "requested_duration_seconds": args.duration_seconds,
            "minimum_requests": args.minimum_requests,
            "concurrency": args.concurrency,
            "prompt_profile": "fixed-slot-shapes-warmed-in-first-wave",
            "prompt_shape_count": args.concurrency,
            "sample_interval_seconds": args.sample_interval,
            "completed_requests": completed,
            "sampled_tokens": sampled_tokens,
            "failures": failures,
            "elapsed_seconds": elapsed,
            "requests_per_second": completed / elapsed if elapsed else 0.0,
            "latency_ms": {
                "p50": percentile([value * 1000 for value in durations], 0.50),
                "p95": percentile([value * 1000 for value in durations], 0.95),
                "max": max((value * 1000 for value in durations), default=None),
            },
        },
        "resources": resources,
        "gate": {
            "requires_duration_seconds": LONG_SOAK_SECONDS,
            "requires_completed_requests": LONG_SOAK_REQUESTS,
            "workload_passed": workload_passed,
            "requested_duration_met": requested_duration_met,
            "long_scale_met": long_scale_met,
            "resource_stability_passed": resources["passed"],
            "process_generation_stable": generation_stable,
            "shutdown_clean": shutdown_clean,
            "m3_soak_exit_gate_passed": gate_passed,
            "decision": (
                "passed: multi-sequence runtime remained correct and resource-stable"
                if gate_passed
                else "blocked: correctness, scale, resource, generation, or shutdown gate failed"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["gate"], ensure_ascii=False, indent=2))
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
