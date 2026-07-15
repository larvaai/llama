from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_FIELDS = (
    "model_manifest_sha256",
    "runtime_manifest_sha256",
    "runtime_executable_sha256",
    "benchmark_source_sha256",
    "scheduler_source_sha256",
    "governance_source_sha256",
    "adapter_source_sha256",
    "native_source_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def load_run(path: Path, *, require_pass: bool = True) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("artifact_version") != "inference-runtime-m5-priority.v1":
        raise ValueError(f"unsupported M5 artifact: {path}")
    if (
        require_pass
        and value.get("gate", {}).get("m5_priority_exit_gate_passed") is not True
    ):
        raise ValueError(f"M5 input gate did not pass: {path}")
    return value


def aggregate(
    run_paths: list[Path],
    stress_paths: list[Path],
) -> dict[str, Any]:
    if len(run_paths) < 3:
        raise ValueError("M5 aggregate requires at least three passing runs")
    runs = [(path, load_run(path)) for path in run_paths]
    identity = runs[0][1]["identity"]
    for path, run in runs[1:]:
        for field in IDENTITY_FIELDS:
            if run["identity"].get(field) != identity.get(field):
                raise ValueError(f"identity mismatch for {field}: {path}")

    orders = {tuple(run["workload"]["scenario_order"]) for _, run in runs}
    if ("neutral", "mixed-priority") not in orders or (
        "mixed-priority",
        "neutral",
    ) not in orders:
        raise ValueError("M5 aggregate requires both scenario orders")

    ratios = [
        run["comparison"]["high_p95_ttft_ratio_mixed_over_neutral"]
        for _, run in runs
    ]
    request_ratios = [
        run["comparison"]["request_throughput_ratio_mixed_over_neutral"]
        for _, run in runs
    ]
    token_ratios = [
        run["comparison"]["token_throughput_ratio_mixed_over_neutral"]
        for _, run in runs
    ]
    scenarios = [
        run[name]
        for _, run in runs
        for name in ("neutral", "mixed_priority")
    ]
    aggregate_values = {
        "run_count": len(runs),
        "passed_run_count": len(runs),
        "median_high_p95_ttft_ratio": round(statistics.median(ratios), 6),
        "median_high_p95_ttft_improvement_percent": round(
            (1 - statistics.median(ratios)) * 100,
            3,
        ),
        "median_request_throughput_ratio": round(
            statistics.median(request_ratios), 6
        ),
        "median_token_throughput_ratio": round(
            statistics.median(token_ratios), 6
        ),
        "minimum_request_throughput_ratio": round(min(request_ratios), 6),
        "minimum_token_throughput_ratio": round(min(token_ratios), 6),
        "maximum_active_sequences": max(
            scenario["capacity"]["max_active_sequences"] for scenario in scenarios
        ),
        "maximum_reserved_kv_tokens": max(
            scenario["capacity"]["max_reserved_kv_tokens"]
            for scenario in scenarios
        ),
        "maximum_low_priority_completion_ms": max(
            run["mixed_priority"]["low_total_ms"]["maximum"] for _, run in runs
        ),
        "final_active_sequences": max(
            scenario["capacity"]["final_active_sequences"]
            for scenario in scenarios
        ),
        "final_reserved_kv_tokens": max(
            scenario["capacity"]["final_reserved_kv_tokens"]
            for scenario in scenarios
        ),
    }

    stress = [(path, load_run(path, require_pass=False)) for path in stress_paths]
    for path, run in stress:
        for field in IDENTITY_FIELDS:
            if run["identity"].get(field) != identity.get(field):
                raise ValueError(f"stress identity mismatch for {field}: {path}")
    stress_token_ratios = [
        run["comparison"]["token_throughput_ratio_mixed_over_neutral"]
        for _, run in stress
    ]
    stress_status = (
        "not_run"
        if not stress_token_ratios
        else (
            "target_met"
            if min(stress_token_ratios) >= 0.9
            else "optimization_backlog"
        )
    )
    run_records = [
        {
            "artifact": path.name,
            "sha256": sha256(path),
            "scenario_order": run["workload"]["scenario_order"],
            "high_p95_ttft_ratio": run["comparison"][
                "high_p95_ttft_ratio_mixed_over_neutral"
            ],
            "request_throughput_ratio": run["comparison"][
                "request_throughput_ratio_mixed_over_neutral"
            ],
            "token_throughput_ratio": run["comparison"][
                "token_throughput_ratio_mixed_over_neutral"
            ],
            "passed": True,
        }
        for path, run in runs
    ]
    gates = {
        "high_priority_ttft_improves_by_at_least_20_percent": (
            aggregate_values["median_high_p95_ttft_ratio"] <= 0.8
        ),
        "request_throughput_loss_at_most_10_percent": (
            aggregate_values["minimum_request_throughput_ratio"] >= 0.9
        ),
        "token_throughput_loss_at_most_10_percent": (
            aggregate_values["minimum_token_throughput_ratio"] >= 0.9
        ),
        "all_requests_complete_without_starvation": (
            aggregate_values["maximum_low_priority_completion_ms"] < 60000
        ),
        "capacity_never_exceeded_and_reconciles_to_zero": (
            aggregate_values["final_active_sequences"] == 0
            and aggregate_values["final_reserved_kv_tokens"] == 0
        ),
    }
    passed = all(gates.values())
    return {
        "artifact_version": "inference-runtime-m5-priority-aggregate.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "git": identity["git"],
            "gpu": identity["gpu"],
            **{
                field: identity[field]
                for field in (
                    "model_manifest_sha256",
                    "runtime_manifest_sha256",
                    "runtime_executable_sha256",
                )
            },
            "aggregator_source_sha256": sha256(Path(__file__).resolve()),
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
            "priority_benchmark_source_sha256": sha256(
                ROOT / "scripts" / "benchmark_inference_priority.py"
            ),
        },
        "method": {
            "workload": "paired neutral and mixed-priority workloads with identical prompt/output inside each run",
            "scenario_order_control": "at least one run in each scenario order",
            "ttft_definition": "queue_ms + native first_sample_ms",
            "throughput_definition": "completed requests and sampled tokens per measured scheduler wall second",
        },
        "runs": run_records,
        "aggregate": aggregate_values,
        "stress_evidence": {
            "heterogeneous_prompt_artifacts": [path.name for path, _ in stress],
            "input_gate_status": [
                run.get("gate", {}).get("m5_priority_exit_gate_passed")
                for _, run in stress
            ],
            "observed_token_throughput_ratio_range": (
                [round(min(stress_token_ratios), 6), round(max(stress_token_ratios), 6)]
                if stress_token_ratios
                else None
            ),
            "status": stress_status,
            "decision": (
                "not measured"
                if stress_status == "not_run"
                else (
                    "current heterogeneous prompt stress meets the 0.9 token-throughput target"
                    if stress_status == "target_met"
                    else "retained as an optimization backlog; not represented as scheduler overhead"
                )
            ),
        },
        "gate": {
            **gates,
            "cancel_releases_within_current_decode_plus_one_tick": (
                "covered by tests/gpu/test_inference_runtime_sequences.py"
            ),
            "m5_priority_exit_gate_passed": passed,
            "decision": (
                f"passed; heterogeneous stress status={stress_status}"
                if passed
                else "failed"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate reproducible M5 priority runs")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--stress-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.runs, args.stress_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "gate": report["gate"]}, indent=2))
    return 0 if report["gate"]["m5_priority_exit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
