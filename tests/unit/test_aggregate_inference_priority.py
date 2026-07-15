from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_inference_priority import aggregate


def write_run(
    path: Path,
    *,
    order: tuple[str, str],
    ratio: float = 0.5,
    throughput: float = 1.0,
    identity_suffix: str = "same",
    passed: bool = True,
) -> Path:
    scenario = {
        "capacity": {
            "max_active_sequences": 4,
            "max_reserved_kv_tokens": 100,
            "final_active_sequences": 0,
            "final_reserved_kv_tokens": 0,
        },
        "low_total_ms": {"maximum": 1000},
    }
    value = {
        "artifact_version": "inference-runtime-m5-priority.v1",
        "identity": {
            "git": {"revision": "r", "dirty": True},
            "gpu": {"name": "gpu"},
            "model_manifest_sha256": f"sha256:model-{identity_suffix}",
            "runtime_manifest_sha256": "sha256:runtime",
            "runtime_executable_sha256": "sha256:binary",
            "benchmark_source_sha256": "sha256:benchmark",
            "scheduler_source_sha256": "sha256:scheduler",
            "governance_source_sha256": "sha256:governance",
            "adapter_source_sha256": "sha256:adapter",
            "native_source_sha256": "sha256:native",
        },
        "workload": {"scenario_order": list(order)},
        "neutral": scenario,
        "mixed_priority": scenario,
        "comparison": {
            "high_p95_ttft_ratio_mixed_over_neutral": ratio,
            "request_throughput_ratio_mixed_over_neutral": throughput,
            "token_throughput_ratio_mixed_over_neutral": throughput,
        },
        "gate": {"m5_priority_exit_gate_passed": passed},
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_aggregate_requires_identity_order_controls_and_computes_gate(tmp_path):
    runs = [
        write_run(tmp_path / "one.json", order=("neutral", "mixed-priority")),
        write_run(tmp_path / "two.json", order=("neutral", "mixed-priority")),
        write_run(tmp_path / "three.json", order=("mixed-priority", "neutral")),
    ]
    report = aggregate(runs, [])
    assert report["aggregate"]["run_count"] == 3
    assert report["aggregate"]["median_high_p95_ttft_improvement_percent"] == 50
    assert report["gate"]["m5_priority_exit_gate_passed"] is True
    assert report["stress_evidence"]["observed_token_throughput_ratio_range"] is None

    mismatched = write_run(
        tmp_path / "mismatch.json",
        order=("mixed-priority", "neutral"),
        identity_suffix="other",
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        aggregate([runs[0], runs[1], mismatched], [])

    with pytest.raises(ValueError, match="both scenario orders"):
        aggregate(runs[:2] + [runs[0]], [])


def test_aggregate_rejects_too_few_or_failed_runs(tmp_path):
    first = write_run(tmp_path / "one.json", order=("neutral", "mixed-priority"))
    with pytest.raises(ValueError, match="at least three"):
        aggregate([first], [])

    failed = write_run(tmp_path / "failed.json", order=("mixed-priority", "neutral"))
    value = json.loads(failed.read_text(encoding="utf-8"))
    value["gate"]["m5_priority_exit_gate_passed"] = False
    failed.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        aggregate([first, first, failed], [])


def test_aggregate_retains_failed_stress_as_backlog_but_checks_identity(tmp_path):
    runs = [
        write_run(tmp_path / "one.json", order=("neutral", "mixed-priority")),
        write_run(tmp_path / "two.json", order=("neutral", "mixed-priority")),
        write_run(tmp_path / "three.json", order=("mixed-priority", "neutral")),
    ]
    stress = write_run(
        tmp_path / "stress.json",
        order=("neutral", "mixed-priority"),
        throughput=0.87,
        passed=False,
    )
    report = aggregate(runs, [stress])
    assert report["gate"]["m5_priority_exit_gate_passed"] is True
    assert report["stress_evidence"]["input_gate_status"] == [False]
    assert report["stress_evidence"]["status"] == "optimization_backlog"
    assert report["stress_evidence"]["observed_token_throughput_ratio_range"] == [
        0.87,
        0.87,
    ]

    mismatch = write_run(
        tmp_path / "stress-mismatch.json",
        order=("neutral", "mixed-priority"),
        identity_suffix="other",
        passed=False,
    )
    with pytest.raises(ValueError, match="stress identity mismatch"):
        aggregate(runs, [mismatch])
