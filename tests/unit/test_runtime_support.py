from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from model_worker import cli
from model_worker.artifacts import ArtifactStore
from model_worker.contracts import GenerateRequest
from model_worker.errors import ErrorDetail, WorkerError
from model_worker.ipc import FrameVerifier, encode_frame
from model_worker.metrics import Metrics
from model_worker.prompt import (
    BEGIN_CONTRACT,
    END_CONTRACT,
    PROMPT_CONTRACT_VERSION,
    build_model_messages,
    contract_instruction,
)
from model_worker.security import ExposurePolicy


def test_worker_error_mapping_and_serialization():
    error = WorkerError("queue_full", "busy", details=[ErrorDetail("$.queue", "full")])
    assert error.http_status == 429
    assert error.retryable is True
    assert error.as_dict() == {
        "code": "queue_full",
        "message": "busy",
        "retryable": True,
        "details": [{"path": "$.queue", "message": "full"}],
    }
    assert WorkerError("invalid_request", "bad").retryable is False
    with pytest.raises(ValueError, match="unknown worker error code"):
        WorkerError("invented", "bad")


def test_metrics_are_low_cardinality_and_render_all_sample_types():
    metrics = Metrics()
    metrics.inc("requests_total", termination="completed", request_id="must-not-be-a-label")
    metrics.observe("latency_ms", 1)
    metrics.observe("latency_ms", 2.5)
    metrics.gauge("queue_depth", 3)
    rendered = metrics.render()
    assert 'model_worker_requests_total{termination="completed"} 1' in rendered
    assert "request_id" not in rendered
    assert "model_worker_latency_ms_count 2" in rendered
    assert "model_worker_latency_ms_sum 3.500000000" in rendered
    assert "model_worker_queue_depth 3.0" in rendered


def test_contract_prompt_only_uses_normalized_fields_and_optional_instructions(request_body):
    request = GenerateRequest.parse(request_body)
    prompt = contract_instruction(request, ("result",))
    assert "Canonical fields: result" in prompt
    assert "Field semantics" not in prompt
    assert prompt.count(BEGIN_CONTRACT) == 1 and prompt.count(END_CONTRACT) == 1

    request_body["output_contract"]["instructions"] = "Use a concise value."
    instructed_request = GenerateRequest.parse(request_body)
    instructed = contract_instruction(instructed_request, ("result", "reason"))
    assert "Field semantics:\nUse a concise value." in instructed
    messages = build_model_messages(instructed_request, ("result", "reason"))
    assert messages[0].role == "system"
    assert sum(message.content.count("Use a concise value.") for message in messages) == 1


def test_ipc_verifier_rejects_every_identity_and_protocol_violation():
    verifier = FrameVerifier("request", "attempt")
    first = encode_frame("started", "request", "attempt", 0)
    assert verifier.verify(first)["type"] == "started"
    verifier.verify(
        encode_frame("phase", "request", "attempt", 1, phase="final")
    )
    verifier.verify(
        encode_frame(
            "final_delta",
            "request",
            "attempt",
            2,
            delta='{"result":"ok"}',
        )
    )
    completed = encode_frame(
        "completed",
        "request",
        "attempt",
        3,
        final_text='{"result":"ok"}',
        usage={},
        timing={},
    )
    assert verifier.verify(json.loads(completed))["sequence"] == 3

    invalid_frames = [
        "[]",
        {"protocol_version": "wrong"},
        {"protocol_version": "model-worker-ipc.v1", "request_id": "request", "attempt_id": "wrong", "sequence": 2, "type": "completed"},
        {"protocol_version": "model-worker-ipc.v1", "request_id": "request", "attempt_id": "attempt", "sequence": 3, "type": "completed"},
        {"protocol_version": "model-worker-ipc.v1", "request_id": "request", "attempt_id": "attempt", "sequence": 2, "type": "unknown"},
    ]
    for frame in invalid_frames:
        with pytest.raises(WorkerError) as captured:
            verifier.verify(frame)
        assert captured.value.code == "worker_crashed"
    with pytest.raises(ValueError, match="unknown frame type"):
        encode_frame("unknown", "request", "attempt", 0)


def test_security_policy_handles_localhost_and_valid_bearer():
    ExposurePolicy("localhost").validate()
    policy = ExposurePolicy("192.0.2.1", "secret", trusted_reverse_proxy=True)
    policy.validate()
    assert policy.authorized("Bearer secret") is True
    assert policy.authorized(None) is False


def test_artifact_quota_terminal_state_safe_ids_and_active_cleanup(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", total_quota=0, retention_seconds=0)
    with pytest.raises(ValueError, match="path-safe"):
        store.begin("../request", "attempt", 100)

    artifact = store.begin("request", "attempt", 1000)
    artifact.write_manifest(
        {},
        {},
        {
            "manifest_digest": "sha256:x",
            "runtime_build": "b10012",
            "prompt_hash": "sha256:prompt",
            "prompt_version": PROMPT_CONTRACT_VERSION,
        },
        {},
        {},
    )
    stored_manifest = json.loads((artifact.path / "manifest.json").read_text(encoding="utf-8"))
    assert stored_manifest["prompt_hash"] == "sha256:prompt"
    assert stored_manifest["prompt_version"] == PROMPT_CONTRACT_VERSION
    artifact.write_result({"termination": "completed"})
    with pytest.raises(RuntimeError, match="terminal artifact"):
        artifact.write_result({"termination": "failed"})
    assert store.cleanup(time.time() + 1) == 0
    store.finish(artifact)
    assert store.cleanup(time.time() + 1) == 1

    limited = store.begin("requesttwo", "attempttwo", 1)
    with pytest.raises(WorkerError) as captured:
        limited.write_manifest({}, {}, {"manifest_digest": "sha256:x", "runtime_build": "b10012"}, {}, {})
    assert captured.value.code == "request_too_large"
    store.finish(limited)


def test_validate_manifest_cli_success_and_failure(monkeypatch, capsys, tmp_path):
    manifest = SimpleNamespace(id="model", digest="sha256:digest", raw={"runtime_build": "b10012"})
    monkeypatch.setattr(cli, "load_manifest", lambda path: manifest)
    assert cli.validate_manifest_main([str(tmp_path / "manifest.json")]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    def fail(path):
        raise WorkerError("worker_not_ready", "bad manifest")

    monkeypatch.setattr(cli, "load_manifest", fail)
    assert cli.validate_manifest_main([str(tmp_path / "manifest.json")]) == 1
    assert json.loads(capsys.readouterr().out) == {"valid": False, "error": "bad manifest"}


def test_service_cli_wires_components_and_closes_on_interrupt(monkeypatch, capsys, manifest, tmp_path):
    events = []

    class Worker:
        def __init__(self, executable, loaded_manifest):
            events.append(("worker", executable, loaded_manifest))

        def start(self):
            events.append(("worker_start",))

    class Dispatcher:
        def __init__(self, worker, capacity):
            events.append(("dispatcher", worker, capacity))

    class Server:
        def __init__(self, address, dispatcher, loaded_manifest, policy):
            events.append(("server", address, dispatcher, loaded_manifest, policy))

        def serve_forever(self, poll_interval):
            events.append(("serve", poll_interval))
            raise KeyboardInterrupt

        def server_close(self):
            events.append(("close",))

    monkeypatch.setattr(cli, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(cli, "NativeWorkerProcess", Worker)
    monkeypatch.setattr(cli, "Dispatcher", Dispatcher)
    monkeypatch.setattr(cli, "ModelWorkerHTTPServer", Server)
    monkeypatch.delenv("MODEL_WORKER_BEARER_TOKEN", raising=False)
    result = cli.main(
        [
            "--model-manifest", str(tmp_path / "manifest.json"),
            "--native-executable", str(tmp_path / "native.exe"),
            "--port", "8123",
        ]
    )
    assert result == 0
    assert ("worker_start",) in events
    assert ("serve", 0.2) in events
    assert events[-1] == ("close",)
    assert json.loads(capsys.readouterr().out)["status"] == "ready"
