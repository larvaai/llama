from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference_runtime import ContinuousBatchScheduler, SchedulingMetadata  # noqa: E402
from inference_runtime.adapters import LlamaCppSteppableBackend  # noqa: E402
from model_worker.manifest import load_manifest  # noqa: E402
from model_worker.output_contract import validate_output  # noqa: E402
from model_worker.preflight import preflight  # noqa: E402
from model_worker.strict_json import loads  # noqa: E402


class TimestampSink:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.events = []
        self.lock = threading.Lock()

    def publish(self, event) -> None:
        with self.lock:
            self.events.append(
                {
                    "kind": event.kind.value,
                    "elapsed_ms": round((time.monotonic() - self.started) * 1000, 3),
                    "tokens": event.tokens,
                }
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def body(model_id: str, expected: str) -> dict[str, Any]:
    return {
        "protocol_version": "model-worker.v1",
        "model_id": model_id,
        "messages": [
            {
                "role": "user",
                "content": f"Return a JSON object whose result field is exactly {expected}.",
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
        "benchmark-workflow",
        f"agent-{request_id}",
        "throughput",
        1,
        None,
    )


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile_value)))
    return round(ordered[index], 3)


def summarize(records: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    sampled = sum(record["sampled_tokens"] for record in records)
    decode_intervals = []
    first_decode = []
    first_sample = []
    first_final = []
    sample_itl = []
    final_itl = []
    for record in records:
        decode = [
            event["elapsed_ms"]
            for event in record.get("events", [])
            if event["kind"] == "decode_completed"
        ]
        if decode:
            first_decode.append(decode[0])
            decode_intervals.extend(
                later - earlier for earlier, later in zip(decode, decode[1:])
            )
        timing = record.get("timing", {})
        if type(timing) is dict:
            if type(timing.get("first_sample_ms")) in {int, float}:
                first_sample.append(float(timing["first_sample_ms"]))
            if type(timing.get("first_final_ms")) in {int, float}:
                first_final.append(float(timing["first_final_ms"]))
            if type(timing.get("sample_itl_ms")) is list:
                sample_itl.extend(float(value) for value in timing["sample_itl_ms"])
            if type(timing.get("final_itl_ms")) is list:
                final_itl.extend(float(value) for value in timing["final_itl_ms"])
    return {
        "wall_seconds": round(wall_seconds, 6),
        "completed_requests": len(records),
        "sampled_tokens": sampled,
        "requests_per_second": round(len(records) / wall_seconds, 6),
        "sampled_tokens_per_second": round(sampled / wall_seconds, 6),
        "mean_request_seconds": round(
            statistics.mean(record["elapsed_seconds"] for record in records),
            6,
        ),
        "p95_first_decode_event_ms": percentile(first_decode, 0.95),
        "p95_decode_quantum_interval_ms": percentile(decode_intervals, 0.95),
        "p95_first_sample_ms": percentile(first_sample, 0.95),
        "p95_first_final_ms": percentile(first_final, 0.95),
        "p95_sample_itl_ms": percentile(sample_itl, 0.95),
        "p95_final_itl_ms": percentile(final_itl, 0.95),
        "token_level_itl_samples": len(sample_itl),
        "token_level_final_itl_samples": len(final_itl),
        "latency_note": (
            "Native sample/final ITL arrays are token-level. Decode event intervals "
            "remain quantum-level diagnostics only."
            if sample_itl
            else "This backend did not expose token-level timing samples."
        ),
        "records": records,
    }


def run_parallel(
    count: int,
    invoke: Callable[[int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    records: dict[int, dict[str, Any]] = {}
    failures: dict[int, BaseException] = {}

    def run(index: int) -> None:
        try:
            records[index] = invoke(index)
        except BaseException as exc:
            failures[index] = exc

    threads = [threading.Thread(target=run, args=(index,)) for index in range(count)]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(180)
    wall = time.perf_counter() - started
    if any(thread.is_alive() for thread in threads):
        raise TimeoutError("parallel benchmark thread did not terminate")
    if failures:
        detail = ", ".join(
            f"{index}={type(error).__name__}:{error}"
            for index, error in sorted(failures.items())
        )
        raise RuntimeError(f"parallel benchmark failed: {detail}")
    return [records[index] for index in range(count)], wall


def custom_record(scheduler, manifest, index: int, *, prefix: str) -> dict[str, Any]:
    expected = f"ok-{index}"
    prepared = preflight(body(manifest.id, expected), manifest)
    sink = TimestampSink()
    started = time.perf_counter()
    result = scheduler.infer(
        prepared,
        scheduling=metadata(f"{prefix}-{index}"),
        events=sink,
    )
    elapsed = time.perf_counter() - started
    if result.output != {"result": expected}:
        raise RuntimeError(f"custom runtime semantic mismatch for request {index}")
    return {
        "index": index,
        "elapsed_seconds": round(elapsed, 6),
        "output": result.output,
        "sampled_tokens": result.usage["sampled_tokens"],
        "prompt_tokens": result.usage["prompt_tokens"],
        "timing": result.timing,
        "events": sink.events,
    }


def benchmark_custom(args, manifest) -> dict[str, Any]:
    backend = LlamaCppSteppableBackend(
        args.runtime_executable,
        args.runtime_manifest,
        startup_timeout=180,
        command_timeout=180,
    )
    scheduler = ContinuousBatchScheduler(backend, tick_token_budget=512)
    try:
        custom_record(scheduler, manifest, 99, prefix="warmup")
        serial_records = []
        serial_started = time.perf_counter()
        for index in range(args.concurrency):
            serial_records.append(
                custom_record(scheduler, manifest, index, prefix="serial")
            )
        serial_wall = time.perf_counter() - serial_started
        concurrent_records, concurrent_wall = run_parallel(
            args.concurrency,
            lambda index: custom_record(
                scheduler,
                manifest,
                index,
                prefix="concurrent",
            ),
        )
        return {
            "serial": summarize(serial_records, serial_wall),
            "concurrent": summarize(concurrent_records, concurrent_wall),
            "cache_stats": backend.cache_stats(),
        }
    finally:
        scheduler.shutdown(timeout=20)


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    encoded = None
    method = "GET"
    if payload is not None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        method = "POST"
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_server(process: subprocess.Popen, base_url: str, timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}")
        try:
            health = request_json(base_url + "/health", None, 1)
            if health.get("status") == "ok":
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    raise TimeoutError("llama-server readiness timed out")


def server_record(base_url: str, manifest, index: int) -> dict[str, Any]:
    expected = f"ok-{index}"
    prepared = preflight(body(manifest.id, expected), manifest)
    payload = {
        "model": Path(manifest.raw["gguf_path"]).name,
        "messages": [
            {"role": message.role, "content": message.content}
            for message in prepared.model_messages
        ],
        "temperature": 0,
        "max_tokens": prepared.limits.total_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "model_worker_output",
                "strict": True,
                "schema": prepared.request.output_contract.schema,
            },
        },
    }
    started = time.perf_counter()
    raw = request_json(base_url + "/v1/chat/completions", payload, 180)
    elapsed = time.perf_counter() - started
    try:
        message = raw["choices"][0]["message"]
        output = loads(message["content"])
        if validate_output(output, prepared.contract):
            raise ValueError("contract mismatch")
        if output != {"result": expected}:
            raise ValueError("semantic mismatch")
        usage = raw["usage"]
        completion_tokens = usage["completion_tokens"]
        prompt_tokens = usage["prompt_tokens"]
        if type(completion_tokens) is not int or type(prompt_tokens) is not int:
            raise TypeError("usage")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid llama-server response for {index}: {exc}") from exc
    return {
        "index": index,
        "elapsed_seconds": round(elapsed, 6),
        "output": output,
        "sampled_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "timing": raw.get("timings", {}),
        "events": [],
    }


def benchmark_server(args, manifest) -> dict[str, Any]:
    runtime_dir = Path(manifest.raw["runtime"]["directory"])
    executable = args.llama_server or runtime_dir / "llama-server.exe"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    model = Path(manifest.raw["gguf_path"])
    base_url = f"http://127.0.0.1:{args.port}"
    environment = os.environ.copy()
    environment["PATH"] = str(runtime_dir) + os.pathsep + environment.get("PATH", "")
    command = [
        str(executable),
        "-m",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "-c",
        "4096",
        "-np",
        str(args.concurrency),
        "-b",
        "1024",
        "-ub",
        "512",
        "-ngl",
        "99",
        "--threads",
        "12",
        "--jinja",
        "--reasoning",
        "on",
        "--reasoning-format",
        "deepseek",
        "--reasoning-budget",
        "256",
        "--reasoning-budget-message",
        "Stop thinking and return only the required JSON object now.",
        "--log-verbosity",
        "1",
        "--log-colors",
        "off",
    ]
    process = subprocess.Popen(
        command,
        cwd=runtime_dir,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        wait_server(process, base_url)
        server_record(base_url, manifest, 99)
        records, wall = run_parallel(
            args.concurrency,
            lambda index: server_record(base_url, manifest, index),
        )
        return {
            "executable": str(executable.resolve()),
            "executable_sha256": sha256(executable),
            "command": command,
            "concurrent": summarize(records, wall),
        }
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(5)


def git_identity(root: Path) -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).splitlines()
    return {"revision": revision, "dirty": bool(dirty), "changed_paths": len(dirty)}


def gpu_identity() -> dict[str, Any] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).strip().splitlines()[0]
        name, driver, memory = [part.strip() for part in output.split(",")]
        return {"name": name, "driver": driver, "memory_mib": int(memory)}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-executable", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--port", type=int, default=18123)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.concurrency < 2 or args.concurrency > 8:
        parser.error("--concurrency must be in 2..8")
    return args


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(args.model_manifest)
    custom = benchmark_custom(args, manifest)
    server = benchmark_server(args, manifest)
    serial = custom["serial"]
    concurrent = custom["concurrent"]
    baseline = server["concurrent"]
    serial_speedup = concurrent["requests_per_second"] / serial["requests_per_second"]
    request_ratio = concurrent["requests_per_second"] / baseline["requests_per_second"]
    token_ratio = (
        concurrent["sampled_tokens_per_second"]
        / baseline["sampled_tokens_per_second"]
    )
    token_level_itl_evidence = concurrent["token_level_itl_samples"] > 0
    p95_sample_itl_slo_ms = 50.0
    itl_slo_pass = (
        token_level_itl_evidence
        and concurrent["p95_sample_itl_ms"] is not None
        and concurrent["p95_sample_itl_ms"] <= p95_sample_itl_slo_ms
    )
    m4_pass = (
        serial_speedup >= 1.5
        and request_ratio >= 0.9
        and itl_slo_pass
    )
    artifact = {
        "artifact_version": "inference-runtime-benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "git": git_identity(root),
            "gpu": gpu_identity(),
            "model_manifest": str(args.model_manifest.resolve()),
            "model_manifest_file_sha256": sha256(args.model_manifest),
            "model_manifest_digest": manifest.digest,
            "runtime_manifest": str(args.runtime_manifest.resolve()),
            "runtime_manifest_file_sha256": sha256(args.runtime_manifest),
            "runtime_executable": str(args.runtime_executable.resolve()),
            "runtime_executable_sha256": sha256(args.runtime_executable),
            "benchmark_source_sha256": sha256(Path(__file__).resolve()),
            "scheduler_source_sha256": sha256(
                root / "inference_runtime" / "continuous_scheduler.py"
            ),
            "governance_source_sha256": sha256(
                root / "inference_runtime" / "governance.py"
            ),
            "adapter_source_sha256": sha256(
                root / "inference_runtime" / "adapters" / "llama_cpp.py"
            ),
            "native_source_sha256": sha256(
                root / "native" / "inference_runtime_main.cpp"
            ),
        },
        "workload": {
            "concurrency": args.concurrency,
            "requests": args.concurrency,
            "reasoning_tokens": 256,
            "final_tokens": 64,
            "total_tokens": 320,
            "per_sequence_context_tokens": 1024,
            "cache_scope": None,
        },
        "custom_runtime": custom,
        "llama_server": server,
        "gate": {
            "serial_speedup": round(serial_speedup, 6),
            "request_throughput_ratio_to_llama_server": round(request_ratio, 6),
            "token_throughput_ratio_to_llama_server": round(token_ratio, 6),
            "requires_serial_speedup": 1.5,
            "requires_request_ratio": 0.9,
            "requires_p95_sample_itl_ms": p95_sample_itl_slo_ms,
            "token_level_itl_evidence": token_level_itl_evidence,
            "itl_slo_passed": itl_slo_pass,
            "m4_exit_gate_passed": m4_pass,
            "decision": (
                "passed: throughput and token-level ITL targets met"
                if m4_pass
                else (
                    "blocked: token-level ITL evidence is absent"
                    if not token_level_itl_evidence
                    else "blocked: throughput and token-level ITL targets are not all met"
                )
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["gate"], ensure_ascii=False, indent=2))
    return 0 if m4_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
