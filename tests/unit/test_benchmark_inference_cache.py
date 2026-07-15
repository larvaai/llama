from __future__ import annotations

import pytest

from scripts import benchmark_inference_cache as benchmark


def record(prompt_decode: float, ttft: float, total: float) -> dict:
    return {
        "prompt_decode_ms": prompt_decode,
        "ttft_ms": ttft,
        "total_ms": total,
    }


def resource(vram: int | None, *, scope: str = "total_system_fallback") -> dict:
    return {
        "root_pid": 1,
        "rss_scope": "service_process_tree",
        "process_tree_pids": [1],
        "process_tree_rss_bytes": 100,
        "gpu_vram_scope": scope,
        "gpu_vram_mib": vram,
        "gpu_vram_fallback_reason": "fixture",
    }


def scenario(*, enabled: bool) -> dict:
    records = [
        {
            **record(0.0 if enabled else 100.0, 90.0 if enabled else 100.0, 200.0),
            "cache_hit": enabled,
            "output": {"result": "cache-benchmark-result"},
        }
        for _ in range(4)
    ]
    baseline = resource(6000)
    populated = resource(6000 if enabled else 6010)
    cleared = resource(6000)
    return {
        "records": records,
        "latency": benchmark._latency_summary(records),
        "cache_stats": {
            "delta_hits": 4 if enabled else 0,
            "delta_misses": 0 if enabled else 4,
            "delta_saved_prefill_tokens": 400 if enabled else 0,
            "after_clear": {"bytes_used": 0},
        },
        "resources": {
            "baseline": baseline,
            "populated": populated,
            "after_clear": cleared,
            "population_delta": benchmark._resource_delta(baseline, populated),
        },
    }


def test_latency_summary_and_percentile_are_deterministic():
    summary = benchmark._latency_summary(
        [record(1.0, 4.0, 7.0), record(2.0, 5.0, 8.0), record(3.0, 6.0, 9.0)]
    )
    assert summary["samples"] == 3
    assert summary["prompt_decode_ms"] == {
        "mean": 2.0,
        "p50": 2.0,
        "p95": 3.0,
        "minimum": 1.0,
        "maximum": 3.0,
    }
    with pytest.raises(ValueError):
        benchmark._latency_summary([])


def test_resource_delta_requires_matching_available_vram_scope():
    assert benchmark._resource_delta(resource(6000), resource(6010)) == {
        "rss_bytes": 0,
        "vram_scope": "total_system_fallback",
        "vram_mib": 10,
    }
    assert benchmark._resource_delta(
        resource(None, scope="unavailable"), resource(None, scope="unavailable")
    )["vram_mib"] is None
    assert benchmark._resource_delta(
        resource(6000), resource(6000, scope="service_process_tree")
    )["vram_mib"] is None


def test_gate_requires_real_hits_latency_measurements_and_reconciled_bytes():
    gate = benchmark._gate(scenario(enabled=False), scenario(enabled=True))
    assert gate["m6_performance_measurement_passed"] is True
    assert gate["enabled_hit_rate"] == 1.0
    assert gate["prompt_decode_p50_ratio_enabled_over_disabled"] == 0.0

    missing_vram = scenario(enabled=True)
    missing_vram["resources"]["populated"] = resource(
        None, scope="unavailable"
    )
    missing_vram["resources"]["population_delta"] = benchmark._resource_delta(
        missing_vram["resources"]["baseline"],
        missing_vram["resources"]["populated"],
    )
    failed = benchmark._gate(scenario(enabled=False), missing_vram)
    assert failed["vram_measurement_available"] is False
    assert failed["m6_performance_measurement_passed"] is False
