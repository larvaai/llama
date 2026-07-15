from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_worker.errors import WorkerError
from model_worker.manifest import ModelManifest, load_manifest
from model_worker.strict_json import loads


RUNTIME_MANIFEST_VERSION = "inference-runtime.v1"
_TOP_LEVEL = {
    "runtime_manifest_version",
    "backend_id",
    "model_manifest",
    "model_manifest_digest",
    "scheduler",
    "cache",
}
_SCHEDULER_KEYS = {
    "max_sequences",
    "cpu_threads",
    "kv_tokens",
    "prefill_chunk_tokens",
    "max_decode_batch",
    "decode_quantum_tokens",
    "tick_token_budget",
}
_CACHE_KEYS = {"enabled", "byte_budget", "max_entries", "ttl_seconds"}


@dataclass(frozen=True, slots=True)
class RuntimeSchedulerConfig:
    max_sequences: int
    cpu_threads: int
    kv_tokens: int
    prefill_chunk_tokens: int
    max_decode_batch: int
    decode_quantum_tokens: int
    tick_token_budget: int


@dataclass(frozen=True, slots=True)
class RuntimeCacheConfig:
    enabled: bool
    byte_budget: int
    max_entries: int
    ttl_seconds: int


@dataclass(frozen=True, slots=True)
class InferenceRuntimeManifest:
    path: Path
    raw: dict[str, Any]
    digest: str
    backend_id: str
    model_manifest: ModelManifest
    scheduler: RuntimeSchedulerConfig
    cache: RuntimeCacheConfig


def _exact_object(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise WorkerError(
            "worker_not_ready",
            f"{name} has missing or unknown fields",
        )
    return value


def _bounded_id(value: Any, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value.encode("utf-8")) > 128
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise WorkerError("worker_not_ready", f"{name} is invalid")
    return value


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if type(value) is not int or value <= 0 or value > maximum:
        raise WorkerError(
            "worker_not_ready",
            f"{name} must be an integer in [1, {maximum}]",
        )
    return value


def load_inference_runtime_manifest(
    path: Path,
    *,
    verify_model_files: bool = True,
) -> InferenceRuntimeManifest:
    resolved = path.resolve()
    try:
        raw_bytes = resolved.read_bytes()
    except OSError as exc:
        raise WorkerError("worker_not_ready", "cannot read inference runtime manifest") from exc
    data = _exact_object(loads(raw_bytes), _TOP_LEVEL, "runtime manifest")
    if data["runtime_manifest_version"] != RUNTIME_MANIFEST_VERSION:
        raise WorkerError("worker_not_ready", "unsupported inference runtime manifest")
    backend_id = _bounded_id(data["backend_id"], "backend_id")
    model_path_value = data["model_manifest"]
    if type(model_path_value) is not str or not model_path_value:
        raise WorkerError("worker_not_ready", "model_manifest must be a path string")
    model_path = Path(model_path_value)
    if not model_path.is_absolute():
        model_path = resolved.parent / model_path
    model = load_manifest(model_path.resolve(), verify_files=verify_model_files)
    if data["model_manifest_digest"] != model.digest:
        raise WorkerError("worker_not_ready", "model manifest digest mismatch")

    scheduler_raw = _exact_object(data["scheduler"], _SCHEDULER_KEYS, "scheduler")
    max_sequences = _positive_int(
        scheduler_raw["max_sequences"],
        "scheduler.max_sequences",
        maximum=256,
    )
    cpu_threads = _positive_int(
        scheduler_raw["cpu_threads"],
        "scheduler.cpu_threads",
        maximum=256,
    )
    kv_tokens = _positive_int(
        scheduler_raw["kv_tokens"],
        "scheduler.kv_tokens",
        maximum=1_048_576,
    )
    prefill_chunk = _positive_int(
        scheduler_raw["prefill_chunk_tokens"],
        "scheduler.prefill_chunk_tokens",
        maximum=model.context["n_batch"],
    )
    decode_batch = _positive_int(
        scheduler_raw["max_decode_batch"],
        "scheduler.max_decode_batch",
        maximum=max_sequences,
    )
    decode_quantum = _positive_int(
        scheduler_raw["decode_quantum_tokens"],
        "scheduler.decode_quantum_tokens",
        maximum=model.limits["max_total_tokens"],
    )
    tick_budget = _positive_int(
        scheduler_raw["tick_token_budget"],
        "scheduler.tick_token_budget",
        maximum=model.context["n_batch"],
    )
    if kv_tokens < max_sequences * 2:
        raise WorkerError("worker_not_ready", "kv_tokens is too small for max_sequences")
    if tick_budget < decode_batch * decode_quantum:
        raise WorkerError(
            "worker_not_ready",
            "tick_token_budget must cover one quantum per decode sequence",
        )
    if prefill_chunk > tick_budget:
        raise WorkerError(
            "worker_not_ready",
            "prefill_chunk_tokens must not exceed tick_token_budget",
        )

    cache_raw = _exact_object(data["cache"], _CACHE_KEYS, "cache")
    if type(cache_raw["enabled"]) is not bool:
        raise WorkerError("worker_not_ready", "cache.enabled must be a boolean")
    cache_byte_budget = _positive_int(
        cache_raw["byte_budget"],
        "cache.byte_budget",
        maximum=64 * 1024**3,
    )
    cache_max_entries = _positive_int(
        cache_raw["max_entries"],
        "cache.max_entries",
        maximum=65_536,
    )
    cache_ttl_seconds = _positive_int(
        cache_raw["ttl_seconds"],
        "cache.ttl_seconds",
        maximum=31_536_000,
    )

    canonical = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return InferenceRuntimeManifest(
        path=resolved,
        raw=data,
        digest="sha256:" + hashlib.sha256(canonical).hexdigest(),
        backend_id=backend_id,
        model_manifest=model,
        scheduler=RuntimeSchedulerConfig(
            max_sequences=max_sequences,
            cpu_threads=cpu_threads,
            kv_tokens=kv_tokens,
            prefill_chunk_tokens=prefill_chunk,
            max_decode_batch=decode_batch,
            decode_quantum_tokens=decode_quantum,
            tick_token_budget=tick_budget,
        ),
        cache=RuntimeCacheConfig(
            enabled=cache_raw["enabled"],
            byte_budget=cache_byte_budget,
            max_entries=cache_max_entries,
            ttl_seconds=cache_ttl_seconds,
        ),
    )
