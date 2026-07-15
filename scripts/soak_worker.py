from __future__ import annotations

import argparse
import http.client
import json
import time
from pathlib import Path


CASES = (
    ("Count the labels that start with A, ignoring case: Alpha, beta, atlas, Gamma.", 2),
    ("Count the even integers in this list: 1, 2, 4, 7, 8.", 3),
    ("Đếm số lần từ 'đỏ' xuất hiện: đỏ, xanh, đỏ, vàng.", 2),
    ("How many Unicode check marks are here: ✓ x ✓ ✓?", 3),
)

IDENTITY_FIELDS = (
    "revision",
    "manifest_digest",
    "runtime_build",
    "model_digest",
    "native_executable_sha256",
)
DEFAULT_FINAL_TOKENS = 64


def resolve_generation_limits(
    manifest_limits: object,
    reasoning_tokens: int | None = None,
) -> dict[str, int]:
    if type(manifest_limits) is not dict:
        raise ValueError("manifest limits must be an object")
    names = ("max_reasoning_tokens", "max_final_tokens", "max_total_tokens")
    caps: dict[str, int] = {}
    for name in names:
        value = manifest_limits.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"manifest {name} must be a positive integer")
        caps[name] = value

    max_total = caps["max_total_tokens"]
    if max_total < 2:
        raise ValueError("manifest max_total_tokens must reserve reasoning and final output")
    final_tokens = min(
        DEFAULT_FINAL_TOKENS,
        caps["max_final_tokens"],
        max_total - 1,
    )
    safe_reasoning_max = min(
        caps["max_reasoning_tokens"],
        max_total - final_tokens,
    )
    if reasoning_tokens is None:
        reasoning_tokens = safe_reasoning_max
    if type(reasoning_tokens) is not int or reasoning_tokens <= 0:
        raise ValueError("--reasoning-tokens must be a positive integer")
    if reasoning_tokens > safe_reasoning_max:
        raise ValueError(
            "--reasoning-tokens exceeds the manifest envelope after reserving final output"
        )
    return {
        "reasoning_tokens": reasoning_tokens,
        "final_tokens": final_tokens,
        "total_tokens": reasoning_tokens + final_tokens,
    }


def extract_runtime_identity(payload: object) -> tuple[str, ...] | None:
    if type(payload) is not dict:
        return None
    model = payload.get("model")
    if type(model) is not dict:
        return None
    identity = tuple(model.get(name) for name in IDENTITY_FIELDS)
    if any(type(value) is not str or not value for value in identity):
        return None
    return identity


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reasoning-tokens", type=int)
    args = parser.parse_args()
    if args.requests <= 0:
        parser.error("--requests must be positive")

    manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    try:
        generation_limits = resolve_generation_limits(
            manifest.get("limits"),
            args.reasoning_tokens,
        )
    except ValueError as exc:
        parser.error(str(exc))
    schema = {"type": "object", "properties": {"result": {"type": "integer"}}, "required": ["result"], "additionalProperties": False}
    started = time.monotonic()
    failures = []
    durations = []
    process_generations = set()
    identities = set()
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_prompt_decode_ms = 0.0
    total_generation_ms = 0.0

    for index in range(args.requests):
        prompt, expected = CASES[index % len(CASES)]
        body = {
            "protocol_version": "model-worker.v1",
            "model_id": manifest["id"],
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "output_contract": {"version": "structured-output.v1", "schema": schema, "instructions": "The result field is the integer count requested by the user."},
            "limits": {**generation_limits, "queue_timeout_ms": 5000, "execution_timeout_ms": 180000},
            "stream": {"enabled": False, "include_reasoning": False},
            "metadata": {"client_request_id": f"soak-{index}"},
        }
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request_started = time.monotonic()
        try:
            connection = http.client.HTTPConnection(args.host, args.port, timeout=190)
            connection.request("POST", "/v1/model/generate", body=raw, headers={"Content-Type": "application/json", "Content-Length": str(len(raw))})
            response = connection.getresponse()
            payload = json.loads(response.read())
            connection.close()
            durations.append(time.monotonic() - request_started)
            model = payload.get("model")
            generation = model.get("process_generation") if type(model) is dict else None
            if generation is not None:
                process_generations.add(generation)
            identity = extract_runtime_identity(payload)
            if identity is not None:
                identities.add(identity)
            usage = payload.get("usage") or {}
            timing = payload.get("timing") or {}
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_generated_tokens += usage.get("sampled_tokens", 0)
            total_prompt_decode_ms += timing.get("prompt_decode_ms", 0)
            total_generation_ms += timing.get("generation_ms", 0)
            actual = (payload.get("output") or {}).get("result")
            successful = response.status == 200 and payload.get("protocol_valid") and payload.get("output_valid") and actual == expected
            if successful and identity is None:
                failures.append({"index": index, "error": "missing_runtime_identity"})
            elif not successful:
                failures.append({"index": index, "status": response.status, "expected": expected, "actual": actual, "error": payload.get("error")})
        except Exception as exc:
            failures.append({"index": index, "exception": type(exc).__name__, "message": str(exc)})

    if len(process_generations) > 1:
        failures.append({"error": "unexpected_process_restart", "process_generations": sorted(process_generations)})
    if len(identities) > 1:
        failures.append({"error": "identity_changed_during_soak", "identity_count": len(identities)})
    result = {
        "requests": args.requests,
        "cases": len(CASES),
        "generation_limits": generation_limits,
        "failures": failures,
        "process_generations": sorted(process_generations),
        "identity": list(next(iter(identities), ())),
        "throughput": {
            "prompt_tokens_per_second": total_prompt_tokens / (total_prompt_decode_ms / 1000) if total_prompt_decode_ms else 0.0,
            "generation_tokens_per_second": total_generated_tokens / (total_generation_ms / 1000) if total_generation_ms else 0.0,
            "prompt_tokens": total_prompt_tokens,
            "generated_tokens": total_generated_tokens,
        },
        "latency_seconds": {"p50": percentile(durations, 0.50), "p95": percentile(durations, 0.95), "max": max(durations, default=0.0)},
        "elapsed_seconds": time.monotonic() - started,
    }
    rendered = json.dumps(result, ensure_ascii=False)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
