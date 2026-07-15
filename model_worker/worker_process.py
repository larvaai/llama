from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import GenerateResult
from .errors import WorkerError
from .ipc import FrameVerifier, encode_frame
from .manifest import ModelManifest
from .output_contract import validate_output
from .request_registry import RequestRecord
from .strict_json import loads


class NativeWorkerProcess:
    """Serial NDJSON data plane plus an out-of-band cancellation control pipe abstraction."""

    def __init__(self, executable: Path, manifest: ModelManifest) -> None:
        self.command = [str(executable), str(manifest.path)]
        self.manifest = manifest
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.RLock()
        self.write_lock = threading.Lock()
        self.process_generation = 0
        self.model_loads_total = 0

    def start(self) -> None:
        with self.lock:
            if self.process and self.process.poll() is None:
                return
            # llama.cpp can emit enough startup diagnostics to fill an unread pipe;
            # discard it here and rely on typed readiness/metrics from the supervisor.
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
            ready = loads(self.process.stdout.readline())
            if ready.get("type") != "ready" or ready.get("protocol_version") != "model-worker-ipc.v1":
                self.process.kill()
                raise WorkerError("worker_not_ready", "native worker did not emit a valid ready frame")
            self.process_generation += 1
            self.model_loads_total += 1

    def execute(self, record: RequestRecord) -> Any:
        self.start()
        assert self.process and self.process.stdin and self.process.stdout
        verifier = FrameVerifier(record.request_id, record.attempt_id)
        started = time.monotonic()
        with self.write_lock:
            self.process.stdin.write(encode_frame("generate", record.request_id, record.attempt_id, 0, request=asdict(record.request)) + "\n")
            self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise WorkerError("worker_crashed", "worker pipe closed before completion")
            frame = verifier.verify(line)
            if frame["type"] == "completed":
                try:
                    output = loads(frame["final_text"])
                except WorkerError as exc:
                    raise WorkerError("output_invalid", "final output is not strict JSON") from exc
                errors = validate_output(output, record.request.contract)
                if errors:
                    raise WorkerError("output_invalid", "final output violates normalized contract", details=errors)
                elapsed = (time.monotonic() - started) * 1000
                return GenerateResult(record.request_id, record.attempt_id, "completed", True, True, output, frame["usage"], {"queue_ms": 0, "prompt_decode_ms": 0, "generation_ms": elapsed, "total_ms": elapsed}, {"id": self.manifest.id, "manifest_digest": self.manifest.digest, "runtime_build": self.manifest.raw["runtime_build"], "process_generation": self.process_generation})
            if frame["type"] == "failed":
                code = frame.get("error", "decode_failed")
                if code not in {"cancelled", "context_overflow", "protocol_violation", "decode_failed"}: code = "decode_failed"
                raise WorkerError(code, frame.get("detail", "native request failed"))

    def cancel(self, record: RequestRecord) -> None:
        record.cancel_event.set()
        with self.lock:
            process = self.process
        if process and process.poll() is None and process.stdin:
            try:
                with self.write_lock:
                    process.stdin.write(encode_frame("cancel", record.request_id, record.attempt_id, 0) + "\n")
                    process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def kill_and_restart(self) -> bool:
        with self.lock:
            if self.process and self.process.poll() is None: self.process.kill()
            self.process = None
        try:
            self.start(); return True
        except WorkerError:
            return False

    def shutdown(self) -> None:
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
            self.process = None
