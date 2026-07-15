from __future__ import annotations

import json

import pytest

from inference_runtime import load_inference_runtime_manifest
from model_worker.errors import WorkerError
from model_worker.manifest import load_manifest


def _runtime_data(model_path, model_digest):
    return {
        "runtime_manifest_version": "inference-runtime.v1",
        "backend_id": "llama-cpp-test",
        "model_manifest": str(model_path),
        "model_manifest_digest": model_digest,
        "scheduler": {
            "max_sequences": 8,
            "cpu_threads": 12,
            "kv_tokens": 8192,
            "prefill_chunk_tokens": 256,
            "max_decode_batch": 8,
            "decode_quantum_tokens": 8,
            "tick_token_budget": 512,
        },
        "cache": {
            "enabled": True,
            "byte_budget": 1024,
            "max_entries": 4,
            "ttl_seconds": 60,
        },
    }


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_runtime_manifest_binds_exact_model_and_scheduler(manifest, tmp_path):
    data = _runtime_data(manifest.path, manifest.digest)
    loaded = load_inference_runtime_manifest(
        _write(tmp_path / "runtime.json", data),
        verify_model_files=False,
    )

    assert loaded.backend_id == "llama-cpp-test"
    assert loaded.model_manifest.digest == manifest.digest
    assert loaded.scheduler.max_sequences == 8
    assert loaded.scheduler.cpu_threads == 12
    assert loaded.scheduler.kv_tokens == 8192
    assert loaded.scheduler.decode_quantum_tokens == 8
    assert loaded.cache.enabled
    assert loaded.digest.startswith("sha256:")


def test_runtime_manifest_resolves_relative_model_path(tmp_path):
    model_data = json.loads(open("config/model.example.json", encoding="utf-8").read())
    model_path = _write(tmp_path / "model.json", model_data)
    model = load_manifest(model_path, verify_files=False)
    data = _runtime_data("model.json", model.digest)

    loaded = load_inference_runtime_manifest(
        _write(tmp_path / "runtime.json", data),
        verify_model_files=False,
    )

    assert loaded.model_manifest.path == model_path.resolve()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.update(extra=True),
        lambda data: data.update(runtime_manifest_version="wrong"),
        lambda data: data.update(backend_id="bad\nbackend"),
        lambda data: data.update(model_manifest_digest="sha256:wrong"),
        lambda data: data["scheduler"].update(max_sequences=0),
        lambda data: data["scheduler"].update(cpu_threads=0),
        lambda data: data["scheduler"].update(kv_tokens=8),
        lambda data: data["scheduler"].update(max_decode_batch=9),
        lambda data: data["scheduler"].update(decode_quantum_tokens=0),
        lambda data: data["scheduler"].update(decode_quantum_tokens=128),
        lambda data: data["scheduler"].update(tick_token_budget=4),
        lambda data: data["scheduler"].update(prefill_chunk_tokens=513),
        lambda data: data["scheduler"].update(extra=1),
        lambda data: data["cache"].update(enabled="true"),
        lambda data: data["cache"].update(byte_budget=0),
        lambda data: data["cache"].update(extra=1),
    ],
)
def test_runtime_manifest_rejects_unbound_or_impossible_config(
    manifest,
    tmp_path,
    mutate,
):
    data = _runtime_data(manifest.path, manifest.digest)
    mutate(data)
    with pytest.raises(WorkerError) as raised:
        load_inference_runtime_manifest(
            _write(tmp_path / "runtime.json", data),
            verify_model_files=False,
        )
    assert raised.value.code == "worker_not_ready"
