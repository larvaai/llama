from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from model_worker.context import decode_prompt, preflight_context, prompt_chunks
from model_worker.errors import WorkerError
from model_worker.contracts import GenerateRequest
from model_worker.manifest import ModelManifest, default_runtime_library, enforce_request_envelope, load_manifest, verify_capabilities
from model_worker.preflight import preflight
from model_worker.prompt import PROMPT_CONTRACT_VERSION


class Capabilities:
    def __init__(self, mapping=None, context=4096, template=True): self.mapping, self._context, self.template = mapping or {}, context, template
    def tokenize(self, text): return self.mapping.get(text, [ord(char) for char in text])
    def has_chat_template(self): return self.template
    def model_context(self): return self._context


def test_multi_token_markers_and_capabilities(manifest):
    start, end = verify_capabilities(manifest, Capabilities({"<think>": [1, 2], "</think>": [3, 4]}))
    assert start == (1, 2) and end == (3, 4)


def test_bad_capabilities_fail_readiness(manifest):
    with pytest.raises(WorkerError): verify_capabilities(manifest, Capabilities(template=False))
    with pytest.raises(WorkerError): verify_capabilities(manifest, Capabilities({"<think>": [1], "</think>": [1]}))


def test_envelope_and_model_identity(manifest, request_body):
    prepared = preflight(request_body, manifest)
    enforce_request_envelope(prepared.request, manifest)
    assert prepared.model_messages[0].role == "system"
    assert prepared.model_messages[-1].content == request_body["messages"][-1]["content"]
    assert prepared.prompt_hash.startswith("sha256:")
    assert prepared.prompt_version == PROMPT_CONTRACT_VERSION
    bad = copy.deepcopy(request_body); bad["model_id"] = "other"
    with pytest.raises(WorkerError): preflight(bad, manifest)


def test_context_chunking_and_cancellation():
    assert list(prompt_chunks(list(range(9)), 4)) == [[0,1,2,3],[4,5,6,7],[8]]
    assert preflight_context(100, 20, 128, 8) == 0
    with pytest.raises(WorkerError) as error: preflight_context(101, 20, 128, 8)
    assert error.value.code == "context_overflow"
    decoded = []
    with pytest.raises(WorkerError) as cancelled:
        decode_prompt(list(range(7)), 3, decoded.append, lambda: len(decoded) == 1)
    assert cancelled.value.code == "cancelled" and decoded == [[0, 1, 2]]


def test_context_input_validation_and_full_decode():
    for values in [(-1, 0, 1, 0), (1.0, 0, 1, 0), (0, 0, True, 0)]:
        with pytest.raises(WorkerError) as captured:
            preflight_context(*values)
        assert captured.value.code == "invalid_request"
    with pytest.raises(ValueError):
        list(prompt_chunks([1], 0))
    decoded = []
    decode_prompt([1, 2, 3], 2, decoded.append, lambda: False)
    assert decoded == [[1, 2], [3]]


def write_manifest(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.pop("id"),
        lambda data: data.update(runtime_build="wrong"),
        lambda data: data.update(id=""),
        lambda data: data["context"].update(n_ctx=0),
        lambda data: data["context"].update(n_batch=data["context"]["n_ctx"] + 1),
        lambda data: data["context"].update(n_ctx=data["context"]["training_context"] + 1, rope_scaling=None),
        lambda data: data.update(sampling={"profile": "random"}),
        lambda data: data.update(reasoning={"mode": "unknown"}),
        lambda data: data.update(reasoning={"mode": "required_marker_sequence", "start_text": "", "end_text": "x"}),
        lambda data: data["reasoning"].pop("require_start"),
        lambda data: data["reasoning"].update(require_start=False),
        lambda data: data["reasoning"].update(require_start=1),
    ],
)
def test_manifest_rejects_each_invalid_readiness_invariant(tmp_path, mutate):
    data = json.loads(Path("config/model.example.json").read_text(encoding="utf-8"))
    mutate(data)
    with pytest.raises(WorkerError) as captured:
        load_manifest(write_manifest(tmp_path, data), verify_files=False)
    assert captured.value.code == "worker_not_ready"


def test_manifest_verifies_model_and_runtime_hashes(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    library = runtime / default_runtime_library()
    library.write_bytes(b"runtime")
    data = json.loads(Path("config/model.example.json").read_text(encoding="utf-8"))
    data["gguf_path"] = str(model)
    data["gguf_sha256"] = "sha256:" + hashlib.sha256(model.read_bytes()).hexdigest()
    data["runtime"]["directory"] = str(runtime)
    data["runtime"]["library_sha256"] = "sha256:" + hashlib.sha256(library.read_bytes()).hexdigest()
    loaded = load_manifest(write_manifest(tmp_path, data))
    assert loaded.id == data["id"]
    assert loaded.digest.startswith("sha256:")

    data["gguf_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(WorkerError, match="model hash mismatch"):
        load_manifest(write_manifest(tmp_path, data))
    model.unlink()
    with pytest.raises(WorkerError, match="model file not found"):
        load_manifest(write_manifest(tmp_path, data))


def test_capability_context_and_unsupported_reasoning_mode(manifest, tmp_path):
    with pytest.raises(WorkerError, match="declared context"):
        verify_capabilities(manifest, Capabilities(context=1))
    raw = copy.deepcopy(manifest.raw)
    raw["reasoning"] = {"mode": "none"}
    with pytest.raises(WorkerError, match="requires reasoning marker sequences"):
        load_manifest(write_manifest(tmp_path, raw), verify_files=False)


def limited_manifest(manifest, **limits):
    raw = copy.deepcopy(manifest.raw)
    raw["limits"].update(limits)
    return ModelManifest(manifest.path, raw, manifest.digest)


@pytest.mark.parametrize(
    ("limit", "value", "expected_code"),
    [
        ("max_messages", 0, "request_too_large"),
        ("message_bytes", 1, "request_too_large"),
        ("input_bytes", 1, "request_too_large"),
        ("instructions_bytes", 0, "request_too_large"),
        ("schema_bytes", 1, "request_too_large"),
        ("client_request_id_bytes", 1, "request_too_large"),
        ("max_total_tokens", 1, "invalid_request"),
    ],
)
def test_request_envelope_enforces_every_resource_limit(manifest, request_body, limit, value, expected_code):
    body = copy.deepcopy(request_body)
    body["output_contract"]["instructions"] = "instruction"
    body["metadata"]["client_request_id"] = "client"
    request = GenerateRequest.parse(body)
    with pytest.raises(WorkerError) as captured:
        enforce_request_envelope(request, limited_manifest(manifest, **{limit: value}))
    assert captured.value.code == expected_code


@pytest.mark.parametrize("limit", ["max_messages", "message_bytes", "input_bytes"])
def test_preflight_counts_internal_system_instruction_in_message_envelope(
    manifest,
    request_body,
    limit,
):
    prepared = preflight(request_body, manifest)
    internal_bytes = len(prepared.model_messages[0].content.encode("utf-8"))
    request_bytes = [len(message["content"].encode("utf-8")) for message in request_body["messages"]]
    if limit == "max_messages":
        value = len(request_body["messages"])
    elif limit == "message_bytes":
        assert internal_bytes > max(request_bytes)
        value = internal_bytes - 1
    else:
        value = sum(request_bytes)

    with pytest.raises(WorkerError) as captured:
        preflight(request_body, limited_manifest(manifest, **{limit: value}))
    assert captured.value.code == "request_too_large"
