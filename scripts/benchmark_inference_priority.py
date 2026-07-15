from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_runtime import (  # noqa: E402
    ContinuousBatchScheduler,
    DecodeStatus,
    PrefillStatus,
    SchedulingMetadata,
)
from inference_runtime.adapters import LlamaCppSteppableBackend  # noqa: E402
from model_worker.manifest import load_manifest  # noqa: E402
from model_worker.preflight import preflight  # noqa: E402


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def publish(self, event) -> None:
        with self._lock:
            self.events.append(
                {
                    "kind": event.kind.value,
                    "at_monotonic": round(event.at_monotonic, 6),
                    "tokens": event.tokens,
                }
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _git_identity() -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).splitlines()
    return {"revision": revision, "dirty": bool(changed), "changed_paths": len(changed)}


def _gpu_identity() -> dict[str, Any] | None:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        name, driver, memory = (part.strip() for part in raw.split(","))
        return {"name": name, "driver": driver, "memory_mib": int(memory)}
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def _body(
    model_id: str,
    expected: str,
    *,
    prompt_padding: str = "",
) -> dict[str, Any]:
    padding = f"\nAdditional context: {prompt_padding}" if prompt_padding else ""
    return {
        "protocol_version": "model-worker.v1",
        "model_id": model_id,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Return a JSON object whose result field is exactly {expected}."
                    f"{padding}"
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
            "queue_timeout_ms": 60000,
            "execution_timeout_ms": 120000,
        },
        "stream": {"enabled": False, "include_reasoning": False},
    }


def _metadata(request_id: str, service_class: str, index: int) -> SchedulingMetadata:
    return SchedulingMetadata(
        request_id,
        f"priority-workflow-{index}",
        f"priority-agent-{index}",
        service_class,
        1,
        None,
    )


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * quantile)))
    return round(ordered[index], 3)


def _warm_backend(backend, manifest) -> None:
    request = preflight(_body(manifest.id, "priority-warmup"), manifest)
    sink = RecordingSink()
    handle = backend.open_sequence(
        request,
        scheduling=_metadata("priority-warmup", "throughput", 999),
        events=sink,
    )
    try:
        while True:
            outcome = backend.prefill(
                handle,
                token_budget=backend.capabilities.max_prefill_tokens_per_step,
                events=sink,
            )
            if outcome.status is PrefillStatus.READY:
                break
        while True:
            outcome = backend.decode(
                handle,
                token_budget=backend.capabilities.max_decode_tokens_per_step,
                events=sink,
            )
            if outcome.status is DecodeStatus.FINISHED:
                if outcome.completion is None:
                    raise RuntimeError("warmup completion missing")
                break
            if outcome.status is DecodeStatus.FAILED:
                raise RuntimeError(f"warmup failed: {outcome.error_code}")
    finally:
        backend.release(handle, events=sink)


def _scenario(
    args: argparse.Namespace,
    manifest,
    *,
    name: str,
    service_classes: list[str],
    high_indices: set[int],
) -> dict[str, Any]:
    backend = LlamaCppSteppableBackend(
        args.runtime_executable,
        args.runtime_manifest,
        startup_timeout=180,
        command_timeout=180,
    )
    _warm_backend(backend, manifest)
    scheduler = ContinuousBatchScheduler(
        backend,
        tick_token_budget=backend.runtime_manifest.scheduler.tick_token_budget,
        autostart=False,
    )
    records: dict[int, dict[str, Any]] = {}
    failures: dict[int, str] = {}
    completion_order: list[int] = []
    record_lock = threading.Lock()
    resource_samples: list[dict[str, int]] = []
    monitor_stop = threading.Event()

    def invoke(index: int) -> None:
        expected = "priority-result"
        prompt_padding = (
            "Context item. " * (16 * (index % 4))
            if args.heterogeneous_prompts
            else ""
        )
        sink = RecordingSink()
        try:
            result = scheduler.infer(
                preflight(
                    _body(
                        manifest.id,
                        expected,
                        prompt_padding=prompt_padding,
                    ),
                    manifest,
                ),
                scheduling=_metadata(
                    f"{name}-{index}",
                    service_classes[index],
                    index,
                ),
                events=sink,
            )
            if result.output != {"result": expected}:
                raise RuntimeError("semantic output mismatch")
            timing = dict(result.timing)
            record = {
                "index": index,
                "service_class": service_classes[index],
                "high_cohort": index in high_indices,
                "output": result.output,
                "sampled_tokens": result.usage["sampled_tokens"],
                "queue_ms": timing["queue_ms"],
                "first_sample_ms": timing["first_sample_ms"],
                "ttft_ms": round(timing["queue_ms"] + timing["first_sample_ms"], 3),
                "total_ms": timing["total_ms"],
                "events": sink.events,
            }
            with record_lock:
                records[index] = record
                completion_order.append(index)
        except BaseException as exc:
            with record_lock:
                failures[index] = f"{type(exc).__name__}: {exc}"

    def monitor() -> None:
        while not monitor_stop.wait(0.005):
            snapshot = scheduler.admission_snapshot
            resource_samples.append(
                {
                    "pending": snapshot.pending_requests,
                    "active": snapshot.active_sequences,
                    "reserved_kv_tokens": snapshot.reserved_kv_tokens,
                }
            )

    threads = [threading.Thread(target=invoke, args=(index,)) for index in range(args.count)]
    try:
        for expected_count, thread in enumerate(threads, start=1):
            thread.start()
            deadline = time.monotonic() + 10
            while scheduler.active_requests < expected_count and time.monotonic() < deadline:
                time.sleep(0.002)
            if scheduler.active_requests != expected_count:
                raise RuntimeError("failed to stage deterministic admission order")

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        started = time.perf_counter()
        scheduler.start()
        deadline = time.monotonic() + args.timeout_seconds
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        wall_seconds = time.perf_counter() - started
        monitor_stop.set()
        monitor_thread.join(2)
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("priority benchmark did not terminate")
        if failures:
            raise RuntimeError(f"priority benchmark failures: {failures}")

        ordered = [records[index] for index in range(args.count)]
        high_ttft = [record["ttft_ms"] for record in ordered if record["high_cohort"]]
        low_total = [record["total_ms"] for record in ordered if not record["high_cohort"]]
        sampled_tokens = sum(record["sampled_tokens"] for record in ordered)
        final_snapshot = scheduler.admission_snapshot
        return {
            "name": name,
            "wall_seconds": round(wall_seconds, 6),
            "completed_requests": len(ordered),
            "sampled_tokens": sampled_tokens,
            "requests_per_second": round(len(ordered) / wall_seconds, 6),
            "sampled_tokens_per_second": round(sampled_tokens / wall_seconds, 6),
            "high_ttft_ms": {
                "mean": round(statistics.mean(high_ttft), 3),
                "p50": _percentile(high_ttft, 0.5),
                "p95": _percentile(high_ttft, 0.95),
            },
            "low_total_ms": {
                "maximum": round(max(low_total), 3),
                "p95": _percentile(low_total, 0.95),
            },
            "completion_order": completion_order,
            "capacity": {
                "max_active_sequences": max(
                    (sample["active"] for sample in resource_samples), default=0
                ),
                "max_reserved_kv_tokens": max(
                    (sample["reserved_kv_tokens"] for sample in resource_samples),
                    default=0,
                ),
                "final_active_sequences": final_snapshot.active_sequences,
                "final_reserved_kv_tokens": final_snapshot.reserved_kv_tokens,
            },
            "records": ordered,
        }
    finally:
        monitor_stop.set()
        scheduler.shutdown(timeout=20)


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU mixed-priority M5 gate")
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-executable", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--high-count", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--mixed-first", action="store_true")
    parser.add_argument("--heterogeneous-prompts", action="store_true")
    args = parser.parse_args()
    if args.count < 4 or not 1 <= args.high_count < args.count:
        parser.error("count must be >=4 and high-count must be within the workload")

    args.model_manifest = args.model_manifest.resolve()
    args.runtime_manifest = args.runtime_manifest.resolve()
    args.runtime_executable = args.runtime_executable.resolve()
    manifest = load_manifest(args.model_manifest)
    high_indices = set(range(args.count - args.high_count, args.count))
    mixed_classes = ["background"] * args.count
    for index in high_indices:
        mixed_classes[index] = "interactive-critical"
    specifications = {
        "neutral": ["throughput"] * args.count,
        "mixed-priority": mixed_classes,
    }
    order = (
        ("mixed-priority", "neutral")
        if args.mixed_first
        else ("neutral", "mixed-priority")
    )
    scenarios = {
        name: _scenario(
            args,
            manifest,
            name=name,
            service_classes=specifications[name],
            high_indices=high_indices,
        )
        for name in order
    }
    neutral = scenarios["neutral"]
    mixed = scenarios["mixed-priority"]

    ttft_ratio = mixed["high_ttft_ms"]["p95"] / neutral["high_ttft_ms"]["p95"]
    request_ratio = mixed["requests_per_second"] / neutral["requests_per_second"]
    token_ratio = (
        mixed["sampled_tokens_per_second"] / neutral["sampled_tokens_per_second"]
    )
    runtime_config = json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
    max_sequences = int(runtime_config["scheduler"]["max_sequences"])
    comparison = {
        "high_p95_ttft_ratio_mixed_over_neutral": round(ttft_ratio, 6),
        "high_p95_ttft_improvement_percent": round((1 - ttft_ratio) * 100, 3),
        "request_throughput_ratio_mixed_over_neutral": round(request_ratio, 6),
        "token_throughput_ratio_mixed_over_neutral": round(token_ratio, 6),
    }
    gates = {
        "high_priority_ttft_improves_by_at_least_20_percent": ttft_ratio <= 0.8,
        "request_throughput_loss_at_most_10_percent": request_ratio >= 0.9,
        "token_throughput_loss_at_most_10_percent": token_ratio >= 0.9,
        "all_requests_complete_without_starvation": (
            neutral["completed_requests"] == args.count
            and mixed["completed_requests"] == args.count
            and mixed["low_total_ms"]["maximum"] < 60000
        ),
        "capacity_never_exceeded_and_reconciles_to_zero": (
            neutral["capacity"]["max_active_sequences"] <= max_sequences
            and mixed["capacity"]["max_active_sequences"] <= max_sequences
            and neutral["capacity"]["final_active_sequences"] == 0
            and mixed["capacity"]["final_active_sequences"] == 0
            and neutral["capacity"]["final_reserved_kv_tokens"] == 0
            and mixed["capacity"]["final_reserved_kv_tokens"] == 0
        ),
    }
    passed = all(gates.values())
    artifact = {
        "artifact_version": "inference-runtime-m5-priority.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "git": _git_identity(),
            "gpu": _gpu_identity(),
            "model_manifest": str(args.model_manifest),
            "model_manifest_sha256": _sha256(args.model_manifest),
            "runtime_manifest": str(args.runtime_manifest),
            "runtime_manifest_sha256": _sha256(args.runtime_manifest),
            "runtime_executable": str(args.runtime_executable),
            "runtime_executable_sha256": _sha256(args.runtime_executable),
            "benchmark_source_sha256": _sha256(Path(__file__).resolve()),
            "scheduler_source_sha256": _sha256(
                ROOT / "inference_runtime" / "continuous_scheduler.py"
            ),
            "governance_source_sha256": _sha256(
                ROOT / "inference_runtime" / "governance.py"
            ),
            "adapter_source_sha256": _sha256(
                ROOT / "inference_runtime" / "adapters" / "llama_cpp.py"
            ),
            "native_source_sha256": _sha256(
                ROOT / "native" / "inference_runtime_main.cpp"
            ),
        },
        "workload": {
            "count": args.count,
            "high_count": args.high_count,
            "prompt_profile": (
                "heterogeneous-four-length-cycle"
                if args.heterogeneous_prompts
                else "homogeneous"
            ),
            "high_indices": sorted(high_indices),
            "admission_order": list(range(args.count)),
            "scenario_order": list(order),
        },
        "neutral": neutral,
        "mixed_priority": mixed,
        "comparison": comparison,
        "gate": {
            **gates,
            "m5_priority_exit_gate_passed": passed,
            "decision": "passed" if passed else "failed",
        },
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact": str(args.artifact), "gate": artifact["gate"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
