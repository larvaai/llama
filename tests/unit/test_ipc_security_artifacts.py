from __future__ import annotations

import os
import time

import pytest

from model_worker.artifacts import ArtifactStore
from model_worker.errors import WorkerError
from model_worker.ipc import FrameVerifier
from model_worker.security import ExposurePolicy


def test_ipc_identity_and_sequence():
    verifier = FrameVerifier("r", "a")
    assert verifier.verify({"protocol_version":"model-worker-ipc.v1","request_id":"r","attempt_id":"a","sequence":0,"type":"started"})["type"] == "started"
    with pytest.raises(WorkerError): verifier.verify({"protocol_version":"model-worker-ipc.v1","request_id":"wrong","attempt_id":"a","sequence":1,"type":"completed"})


def test_ipc_request_frame_state_machine_fails_closed():
    envelope = {
        "protocol_version": "model-worker-ipc.v1",
        "request_id": "r",
        "attempt_id": "a",
    }
    with pytest.raises(WorkerError, match="did not start"):
        FrameVerifier("r", "a").verify(
            {
                **envelope,
                "sequence": 0,
                "type": "completed",
                "final_text": "{}",
                "usage": {},
                "timing": {},
            }
        )

    verifier = FrameVerifier("r", "a")
    verifier.verify({**envelope, "sequence": 0, "type": "started"})
    with pytest.raises(WorkerError, match="duplicate IPC started"):
        verifier.verify({**envelope, "sequence": 1, "type": "started"})

    verifier = FrameVerifier("r", "a")
    verifier.verify({**envelope, "sequence": 0, "type": "started"})
    with pytest.raises(WorkerError, match="invalid IPC final delta"):
        verifier.verify({**envelope, "sequence": 1, "type": "final_delta", "delta": "x"})

    verifier = FrameVerifier("r", "a")
    verifier.verify({**envelope, "sequence": 0, "type": "started"})
    with pytest.raises(WorkerError, match="before final phase"):
        verifier.verify(
            {
                **envelope,
                "sequence": 1,
                "type": "completed",
                "final_text": "{}",
                "usage": {},
                "timing": {},
            }
        )


def test_ipc_request_frame_state_machine_accepts_valid_terminal_flow():
    envelope = {
        "protocol_version": "model-worker-ipc.v1",
        "request_id": "r",
        "attempt_id": "a",
    }
    verifier = FrameVerifier("r", "a")
    verifier.verify({**envelope, "sequence": 0, "type": "started"})
    verifier.verify({**envelope, "sequence": 1, "type": "progress", "phase": "reasoning", "tokens": 16})
    verifier.verify({**envelope, "sequence": 2, "type": "phase", "phase": "final"})
    verifier.verify({**envelope, "sequence": 3, "type": "final_delta", "delta": "{}"})
    terminal = verifier.verify(
        {
            **envelope,
            "sequence": 4,
            "type": "completed",
            "final_text": "{}",
            "usage": {},
            "timing": {},
        }
    )
    assert terminal["type"] == "completed"


def test_external_exposure_fails_closed():
    with pytest.raises(WorkerError): ExposurePolicy("0.0.0.0").validate()
    ExposurePolicy("0.0.0.0", "secret", tls_terminated=True).validate()
    assert not ExposurePolicy("127.0.0.1", "secret").authorized("Bearer wrong")


def test_immutable_atomic_artifacts_and_duplicate_client_correlation(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", total_quota=10_000, retention_seconds=1)
    first = store.begin("requestone", "attemptone", 1000)
    first.write_manifest({"messages":["private"]}, {"schema":1}, {"manifest_digest":"sha256:x","runtime_build":"b10012"}, {}, {})
    first.write_result({"termination":"completed"}); store.finish(first)
    second = store.begin("requesttwo", "attempttwo", 1000)
    assert first.path != second.path
    assert "private" not in (first.path / "manifest.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError): first.write_manifest({}, {}, {"manifest_digest":"x","runtime_build":"b10012"}, {}, {})
    os.utime(first.path / "result.json", (0,0))
    assert store.cleanup(time.time()) == 1
