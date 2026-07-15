from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_worker.manifest import load_manifest


def pytest_addoption(parser):
    parser.addoption("--model-manifest", action="store", default=None, help="local verified model manifest for GPU tests")
    parser.addoption("--worker-url", action="store", default=None, help="running released worker URL for real-model tests")
    parser.addoption("--require-gpu", action="store_true", default=False, help="fail instead of skip when GPU evidence is unavailable")


@pytest.fixture
def manifest(tmp_path: Path):
    data = json.loads(Path("config/model.example.json").read_text(encoding="utf-8"))
    return load_manifest(_write(tmp_path / "model.json", data), verify_files=False)


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def request_body():
    return {
        "protocol_version": "model-worker.v1",
        "model_id": "qwen35-9b-local",
        "messages": [{"role": "user", "content": "Return a structured answer."}],
        "output_contract": {"version": "structured-output.v1", "schema": {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"], "additionalProperties": False}},
        "limits": {"reasoning_tokens": 16, "final_tokens": 8, "total_tokens": 20, "queue_timeout_ms": 100, "execution_timeout_ms": 200},
        "stream": {"enabled": False, "include_reasoning": False},
        "metadata": {"client_request_id": "opaque"},
    }
