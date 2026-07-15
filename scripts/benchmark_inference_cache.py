from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_runtime import (  # noqa: E402
    CacheScope,
    CacheVisibility,
    DecodeStatus,
    PrefillStatus,
    SchedulingMetadata,
)
from inference_runtime.adapters import LlamaCppSteppableBackend  # noqa: E402
from model_worker.manifest import load_manifest  # noqa: E402
from model_worker.preflight import preflight  # noqa: E402
from model_worker.strict_json import loads  # noqa: E402
from scripts.monitor_resources import collect_sample  # noqa: E402


class NullSink:
    def publish(self, _event: object) -> None:
        return


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


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * quantile)))
    return round(ordered[index], 3)


def _latency_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("latency summary requires at least one record")
    result: dict[str, Any] = {"samples": len(records)}
    for field in ("prompt_decode_ms", "ttft_ms", "total_ms"):
        values = [float(record[field]) for record in records]
        result[field] = {
            "mean": round(statistics.mean(values), 3),
            "p50": _percentile(values, 0.5),
            "p95": _percentile(values, 0.95),
            "minimum": round(min(values), 3),
            "maximum": round(max(values), 3),
        }
    return result


def _resource_view(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "root_pid": sample["root_pid"],
        "rss_scope": sample["rss_scope"],
        "process_tree_pids": sample["process_tree_pids"],
        "process_tree_rss_bytes": sample["process_tree_rss_bytes"],
        "gpu_vram_scope": sample["gpu_vram_scope"],
        "gpu_vram_mib": sample["gpu_vram_mib"],
        "gpu_vram_fallback_reason": sample["gpu_vram_fallback_reason"],
        "gpu_vram_backend": sample["gpu_vram_backend"],
    }


def _resource_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    rss_before = before["process_tree_rss_bytes"]
    rss_after = after["process_tree_rss_bytes"]
    vram_before = before["gpu_vram_mib"]
    vram_after = after["gpu_vram_mib"]
    same_vram_scope = (
        before["gpu_vram_scope"] == after["gpu_vram_scope"]
        and before["gpu_vram_scope"] != "unavailable"
    )
    return {
        "rss_bytes": (
            rss_after - rss_before
            if rss_before is not None and rss_after is not None
            else None
        ),
        "vram_scope": before["gpu_vram_scope"] if same_vram_scope else None,
        "vram_mib": (
            vram_after - vram_before
            if same_vram_scope and vram_before is not None and vram_after is not None
            else None
        ),
    }


def _body(model_id: str, expected: str, *, long_context: bool) -> dict[str, Any]:
    shared = ""
    if long_context:
        shared = (
            "Stable cache benchmark context. Preserve it exactly while following "
            "the final instruction. "
            * 40
        )
    return {
        "protocol_version": "model-worker.v1",
        "model_id": model_id,
        "messages": [
            {
                "role": "user",
                "content": (
                    shared
                    + f"Return a JSON object whose result field is exactly {expected}."
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
            "queue_timeout_ms": 180000,
            "execution_timeout_ms": 180000,
        },
        "stream": {"enabled": False, "include_reasoning": False},
    }


def _metadata(request_id: str, scope: CacheScope | None) -> SchedulingMetadata:
    return SchedulingMetadata(
        request_id,
        scope.workflow_id if scope is not None else "cache-benchmark-warmup",
        scope.agent_id if scope is not None else "cache-benchmark-warmup",
        "throughput",
        1,
        None,
        scope,
    )


def _run_request(
    backend: LlamaCppSteppableBackend,
    manifest: Any,
    *,
    request_id: str,
    expected: str,
    scope: CacheScope | None,
    long_context: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    sink = NullSink()
    handle = backend.open_sequence(
        preflight(
            _body(manifest.id, expected, long_context=long_context),
            manifest,
        ),
        scheduling=_metadata(request_id, scope),
        events=sink,
    )
    completion = None
    try:
        while True:
            prefill = backend.prefill(
                handle,
                token_budget=backend.capabilities.max_prefill_tokens_per_step,
                events=sink,
            )
            if prefill.status is PrefillStatus.READY:
                break
        while True:
            decode = backend.decode(
                handle,
                token_budget=backend.capabilities.max_decode_tokens_per_step,
                events=sink,
            )
            if decode.status is DecodeStatus.FAILED:
                raise RuntimeError(f"native decode failed: {decode.error_code}")
            if decode.status is DecodeStatus.FINISHED:
                completion = decode.completion
                break
    finally:
        backend.release(handle, events=sink)
    if completion is None:
        raise RuntimeError("native completion missing")
    wall_ms = (time.perf_counter() - started) * 1000
    output = loads(completion.final_text)
    if output != {"result": expected}:
        raise RuntimeError(f"semantic output mismatch: {output!r}")
    return {
        "request_id": request_id,
        "output": output,
        "cache_hit": completion.cache_hit,
        "cache_match": completion.cache_match,
        "cached_prompt_tokens": completion.cached_prompt_tokens,
        "prompt_tokens": completion.prompt_tokens,
        "sampled_tokens": completion.sampled_tokens,
        "queue_ms": 0.0,
        "prompt_decode_ms": round(completion.prompt_decode_ms, 3),
        "first_sample_ms": round(completion.first_sample_ms, 3),
        "ttft_ms": round(completion.first_sample_ms, 3),
        "total_ms": round(
            completion.prompt_decode_ms + completion.generation_ms,
            3,
        ),
        "wall_ms": round(wall_ms, 3),
    }


def _scenario(
    *,
    name: str,
    manifest: Any,
    runtime_manifest: Path,
    runtime_executable: Path,
    repetitions: int,
    cache_enabled: bool,
) -> dict[str, Any]:
    backend = LlamaCppSteppableBackend(
        runtime_executable,
        runtime_manifest,
        startup_timeout=180,
        command_timeout=180,
    )
    backend.start()
    scope = CacheScope(
        "cache-benchmark-tenant",
        "cache-benchmark-workflow",
        "cache-benchmark-agent",
        CacheVisibility.PRIVATE,
    )
    expected = "cache-benchmark-result"
    try:
        _run_request(
            backend,
            manifest,
            request_id=f"{name}-warmup",
            expected="cache-benchmark-warmup",
            scope=None,
            long_context=False,
        )
        backend.clear_cache()
        pid = backend.process_id
        if pid is None:
            raise RuntimeError("native runtime PID unavailable after warmup")
        baseline_resource = _resource_view(collect_sample(pid))
        stats_before = backend.cache_stats()

        seed = None
        if cache_enabled:
            seed = _run_request(
                backend,
                manifest,
                request_id=f"{name}-seed",
                expected=expected,
                scope=scope,
                long_context=True,
            )
            if seed["cache_hit"]:
                raise RuntimeError("cache seed unexpectedly hit")

        records = [
            _run_request(
                backend,
                manifest,
                request_id=f"{name}-{index}",
                expected=expected,
                scope=scope,
                long_context=True,
            )
            for index in range(repetitions)
        ]
        populated_resource = _resource_view(collect_sample(pid))
        stats_after = backend.cache_stats()
        removed_entries = backend.clear_cache()
        cleared_resource = _resource_view(collect_sample(pid))
        stats_cleared = backend.cache_stats()
        return {
            "name": name,
            "cache_enabled": cache_enabled,
            "seed": seed,
            "records": records,
            "latency": _latency_summary(records),
            "cache_stats": {
                "before": stats_before,
                "after": stats_after,
                "after_clear": stats_cleared,
                "removed_entries": removed_entries,
                "delta_hits": stats_after["hits"] - stats_before["hits"],
                "delta_misses": stats_after["misses"] - stats_before["misses"],
                "delta_saved_prefill_tokens": (
                    stats_after["saved_prefill_tokens"]
                    - stats_before["saved_prefill_tokens"]
                ),
            },
            "resources": {
                "baseline": baseline_resource,
                "populated": populated_resource,
                "after_clear": cleared_resource,
                "population_delta": _resource_delta(
                    baseline_resource, populated_resource
                ),
                "clear_delta": _resource_delta(
                    populated_resource, cleared_resource
                ),
            },
        }
    finally:
        backend.shutdown()


def _disabled_runtime_manifest(source: Path, target: Path) -> str:
    raw = json.loads(source.read_text(encoding="utf-8"))
    model_manifest = Path(raw["model_manifest"])
    if not model_manifest.is_absolute():
        raw["model_manifest"] = str((source.parent / model_manifest).resolve())
    raw["backend_id"] = raw["backend_id"] + "-cache-disabled-benchmark"
    raw["cache"]["enabled"] = False
    encoded = json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
    target.write_text(encoded, encoding="utf-8")
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _gate(disabled: dict[str, Any], enabled: dict[str, Any]) -> dict[str, Any]:
    disabled_records = disabled["records"]
    enabled_records = enabled["records"]
    enabled_stats = enabled["cache_stats"]
    disabled_stats = disabled["cache_stats"]
    attempts = enabled_stats["delta_hits"] + enabled_stats["delta_misses"]
    hit_rate = enabled_stats["delta_hits"] / attempts if attempts else 0.0
    vram_delta = enabled["resources"]["population_delta"]["vram_mib"]
    measurements_available = all(
        scenario["resources"][stage]["gpu_vram_mib"] is not None
        for scenario in (disabled, enabled)
        for stage in ("baseline", "populated", "after_clear")
    )
    checks = {
        "disabled_requests_are_all_misses": all(
            not record["cache_hit"] for record in disabled_records
        ),
        "enabled_requests_are_all_hits": all(
            record["cache_hit"] for record in enabled_records
        ),
        "enabled_hit_rate_at_least_80_percent": hit_rate >= 0.8,
        "saved_prefill_tokens_positive": (
            enabled_stats["delta_saved_prefill_tokens"] > 0
        ),
        "disabled_runtime_saved_no_prefill_tokens": (
            disabled_stats["delta_saved_prefill_tokens"] == 0
        ),
        "prompt_decode_p50_improved": (
            enabled["latency"]["prompt_decode_ms"]["p50"]
            < disabled["latency"]["prompt_decode_ms"]["p50"]
        ),
        "ttft_p50_not_regressed_more_than_10_percent": (
            enabled["latency"]["ttft_ms"]["p50"]
            <= disabled["latency"]["ttft_ms"]["p50"] * 1.10
        ),
        "semantic_outputs_identical": (
            {json.dumps(record["output"], sort_keys=True) for record in disabled_records}
            == {json.dumps(record["output"], sort_keys=True) for record in enabled_records}
            == {json.dumps({"result": "cache-benchmark-result"}, sort_keys=True)}
        ),
        "vram_measurement_available": measurements_available,
        "cache_population_vram_delta_within_256_mib": (
            vram_delta is not None and vram_delta <= 256
        ),
        "clear_reconciles_native_cache_bytes_to_zero": (
            enabled_stats["after_clear"]["bytes_used"] == 0
        ),
    }
    return {
        **checks,
        "enabled_hit_rate": round(hit_rate, 6),
        "ttft_p50_ratio_enabled_over_disabled": round(
            enabled["latency"]["ttft_ms"]["p50"]
            / disabled["latency"]["ttft_ms"]["p50"],
            6,
        ),
        "prompt_decode_p50_ratio_enabled_over_disabled": round(
            enabled["latency"]["prompt_decode_ms"]["p50"]
            / disabled["latency"]["prompt_decode_ms"]["p50"],
            6,
        ),
        "m6_performance_measurement_passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU M6 cache on/off measurement gate")
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-executable", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=4)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("repetitions must be at least 3")

    args.model_manifest = args.model_manifest.resolve()
    args.runtime_manifest = args.runtime_manifest.resolve()
    args.runtime_executable = args.runtime_executable.resolve()
    manifest = load_manifest(args.model_manifest)

    with tempfile.TemporaryDirectory(prefix="agent-harness-cache-benchmark-") as temp:
        disabled_manifest = Path(temp) / "runtime-cache-disabled.json"
        disabled_digest = _disabled_runtime_manifest(
            args.runtime_manifest, disabled_manifest
        )
        disabled = _scenario(
            name="cache-disabled",
            manifest=manifest,
            runtime_manifest=disabled_manifest,
            runtime_executable=args.runtime_executable,
            repetitions=args.repetitions,
            cache_enabled=False,
        )
        enabled = _scenario(
            name="cache-enabled",
            manifest=manifest,
            runtime_manifest=args.runtime_manifest,
            runtime_executable=args.runtime_executable,
            repetitions=args.repetitions,
            cache_enabled=True,
        )

    gate = _gate(disabled, enabled)
    artifact = {
        "artifact_version": "inference-runtime-m6-performance.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "git": _git_identity(),
            "gpu": _gpu_identity(),
            "model_manifest": str(args.model_manifest),
            "model_manifest_sha256": _sha256(args.model_manifest),
            "runtime_manifest": str(args.runtime_manifest),
            "runtime_manifest_sha256": _sha256(args.runtime_manifest),
            "disabled_runtime_manifest_sha256": disabled_digest,
            "runtime_executable": str(args.runtime_executable),
            "runtime_executable_sha256": _sha256(args.runtime_executable),
            "benchmark_source_sha256": _sha256(Path(__file__).resolve()),
            "resource_monitor_source_sha256": _sha256(
                ROOT / "scripts" / "monitor_resources.py"
            ),
        },
        "method": {
            "repetitions": args.repetitions,
            "comparison": "same model, executable, prompt and sampling; one runtime has cache.enabled=false and one uses the pinned enabled cache configuration",
            "ttft_definition": "native first_sample_ms on the direct steppable lifecycle; scheduler queue is intentionally excluded from the cache-effect measurement",
            "resource_definition": "native process-tree RSS and process-scoped VRAM when available, otherwise explicitly labelled total-system fallback",
            "vram_tolerance_mib": 256,
        },
        "cache_disabled": disabled,
        "cache_enabled": enabled,
        "gate": {
            **gate,
            "decision": (
                "passed"
                if gate["m6_performance_measurement_passed"]
                else "failed"
            ),
        },
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact": str(args.artifact), "gate": artifact["gate"]}, indent=2))
    return 0 if gate["m6_performance_measurement_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
