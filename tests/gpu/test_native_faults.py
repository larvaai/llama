from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import pytest

from model_worker.ipc import FrameVerifier, encode_frame
from model_worker.manifest import load_manifest
from model_worker.output_contract import validate_output
from model_worker.preflight import preflight
from model_worker.request_registry import RequestRegistry
from model_worker.strict_json import loads
from model_worker.worker_process import NativeWorkerProcess, SupervisorState


class NativeSession:
    def __init__(self, executable: Path, manifest_path: Path):
        self.executable = executable
        self.manifest_path = manifest_path
        self.process: subprocess.Popen[str] | None = None
        self.frames: queue.Queue[str | None] = queue.Queue()
        self.generation = 0

    def start(self, timeout: float = 180) -> None:
        env = os.environ.copy()
        env["PATH"] = str(self.executable.parent) + os.pathsep + env.get("PATH", "")
        self.process = subprocess.Popen(
            [str(self.executable), str(self.manifest_path)],
            cwd=self.executable.parent,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.frames = queue.Queue()

        def read_frames():
            assert self.process and self.process.stdout
            for line in self.process.stdout:
                self.frames.put(line)
            self.frames.put(None)

        threading.Thread(target=read_frames, daemon=True).start()
        ready = self.next_frame(timeout)
        payload = loads(ready)
        assert payload["type"] == "ready"
        assert payload["protocol_version"] == "model-worker-ipc.v1"
        self.generation += 1

    def next_frame(self, timeout: float = 180) -> str:
        try:
            line = self.frames.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("native worker emitted no frame before timeout") from exc
        if line is None:
            raise RuntimeError("native worker pipe closed")
        return line

    def send(self, frame: str) -> None:
        assert self.process and self.process.stdin
        self.process.stdin.write(frame + "\n")
        self.process.stdin.flush()

    def generate(
        self,
        prepared,
        *,
        cancel_phase: str | None = None,
        request_id: str | None = None,
        attempt_id: str | None = None,
    ):
        request_id = request_id or uuid.uuid4().hex
        attempt_id = attempt_id or uuid.uuid4().hex
        self.send(
            encode_frame(
                "generate",
                request_id,
                attempt_id,
                0,
                request=asdict(prepared),
            )
        )
        verifier = FrameVerifier(request_id, attempt_id)
        cancelled = False
        observed = []
        while True:
            frame = verifier.verify(self.next_frame())
            observed.append(frame["type"])
            phase = frame.get("phase")
            if cancel_phase == "prompt" and frame["type"] == "progress" and phase == "prompt_decode":
                self.send(encode_frame("cancel", request_id, attempt_id, 0))
                cancelled = True
            elif cancel_phase == "reasoning" and frame["type"] == "progress" and phase == "reasoning":
                self.send(encode_frame("cancel", request_id, attempt_id, 0))
                cancelled = True
            elif cancel_phase == "final" and frame["type"] == "phase" and phase == "final":
                self.send(encode_frame("cancel", request_id, attempt_id, 0))
                cancelled = True
            if frame["type"] in {"completed", "failed"}:
                if cancel_phase is not None:
                    assert cancelled, f"never observed cancellable {cancel_phase} phase: {observed}"
                return frame

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                self.send(encode_frame("shutdown", "", "", 0))
                process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        self.process = None


def request_body(model_id: str, *, phase: str | None = None) -> dict:
    if phase == "prompt":
        prompt = "Read this bounded input, then count the final labels: " + "token " * 1800 + " Alpha, beta, atlas."
        schema = {"type": "object", "properties": {"result": {"type": "integer"}}, "required": ["result"], "additionalProperties": False}
        instructions = "The result field is the count requested at the end of the user message."
    elif phase == "final":
        prompt = "Return a long lowercase string value."
        schema = {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"], "additionalProperties": False}
        instructions = "The result field should contain at least 300 lowercase x characters."
    else:
        prompt = "Count labels starting with A, ignoring case: Alpha, beta, atlas, Gamma."
        schema = {"type": "object", "properties": {"result": {"type": "integer"}}, "required": ["result"], "additionalProperties": False}
        instructions = "The result field is the integer count requested by the user."
    return {
        "protocol_version": "model-worker.v1",
        "model_id": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "output_contract": {
            "version": "structured-output.v1",
            "schema": schema,
            "instructions": instructions,
        },
        "limits": {
            "reasoning_tokens": 768 if phase == "final" else 256,
            "final_tokens": 400 if phase == "final" else 64,
            "total_tokens": 1000 if phase == "final" else 300,
            "queue_timeout_ms": 5000,
            "execution_timeout_ms": 180000,
        },
        "stream": {"enabled": False, "include_reasoning": False},
    }


@pytest.mark.gpu
def test_real_native_cancel_corruption_and_crash_recovery(request):
    manifest_option = request.config.getoption("--model-manifest")
    executable_option = request.config.getoption("--native-executable")
    if not manifest_option or not executable_option:
        if request.config.getoption("--require-gpu"):
            pytest.fail("real native fault gate requires --model-manifest and --native-executable")
        pytest.skip("pass --model-manifest and --native-executable for real native fault evidence")

    manifest = load_manifest(Path(manifest_option))
    executable = Path(executable_option).resolve()
    assert executable.is_file()
    session = NativeSession(executable, manifest.path)
    session.start()
    try:
        # Malformed and stale control frames must not poison the next valid job.
        session.send("{malformed-control")
        session.send(encode_frame("cancel", "stale-request", "stale-attempt", 0))
        baseline = preflight(request_body(manifest.id), manifest)
        completed = session.generate(baseline)
        assert completed["type"] == "completed"
        output = loads(completed["final_text"])
        assert validate_output(output, baseline.contract) == []

        shared_request = uuid.uuid4().hex
        stale_attempt = uuid.uuid4().hex
        current_attempt = uuid.uuid4().hex
        session.send(encode_frame("cancel", shared_request, stale_attempt, 0))
        assert session.generate(
            baseline,
            request_id=shared_request,
            attempt_id=current_attempt,
        )["type"] == "completed"

        for phase in ("prompt", "reasoning", "final"):
            prepared = preflight(request_body(manifest.id, phase=phase), manifest)
            cancel_started = time.monotonic()
            cancelled = session.generate(prepared, cancel_phase=phase)
            cancel_elapsed = time.monotonic() - cancel_started
            assert cancelled["type"] == "failed"
            assert cancelled["error"] == "cancelled"
            assert cancel_elapsed < 30, f"{phase} cancellation took {cancel_elapsed:.3f}s"
            recovery = session.generate(baseline)
            assert recovery["type"] == "completed"

        # A real process crash is followed by a clean model reload/generation.
        assert session.process is not None
        session.process.kill()
        session.process.wait(timeout=10)
        session.start()
        assert session.generation == 2
        assert session.generate(baseline)["type"] == "completed"
    finally:
        session.stop()


@pytest.mark.gpu
def test_real_supervisor_observes_crash_and_reloads_clean_generation(request):
    manifest_option = request.config.getoption("--model-manifest")
    executable_option = request.config.getoption("--native-executable")
    if not manifest_option or not executable_option:
        if request.config.getoption("--require-gpu"):
            pytest.fail("real supervisor gate requires --model-manifest and --native-executable")
        pytest.skip("pass --model-manifest and --native-executable for real supervisor evidence")

    manifest = load_manifest(Path(manifest_option))
    executable = Path(executable_option).resolve()
    prepared = preflight(request_body(manifest.id), manifest)
    registry = RequestRegistry()

    def record():
        return registry.create(
            prepared,
            prepared.limits.queue_timeout_ms,
            prepared.limits.execution_timeout_ms,
        )

    worker = NativeWorkerProcess(executable, manifest, startup_timeout=180, shutdown_grace=10)
    try:
        assert worker.execute(record()).output == {"result": 2}
        assert worker.process_generation == 1
        crashed = worker.process
        assert crashed is not None
        crashed.kill()
        crashed.wait(timeout=10)
        assert worker.supervisor_state == SupervisorState.DEGRADED

        assert worker.execute(record()).output == {"result": 2}
        assert worker.process_generation == 2
        assert worker.supervisor_state == SupervisorState.READY
    finally:
        worker.shutdown()
