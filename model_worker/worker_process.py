from __future__ import annotations

import hashlib
import os
import queue
import subprocess
import threading
import time
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from .contracts import GenerateResult
from .errors import WorkerError
from .events import (
    EventValidationError,
    InferenceEvent,
    InferenceEventType,
    PublishResult,
)
from .ipc import FrameVerifier, encode_frame
from .manifest import ModelManifest
from .output_contract import validate_output
from .request_registry import RequestRecord
from .strict_json import loads


_EVENT_FRAME_TYPES = frozenset(event_type.value for event_type in InferenceEventType)


class SupervisorState(str, Enum):
    STARTING = "STARTING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    RESTARTING = "RESTARTING"
    DRAINING = "DRAINING"
    STOPPED = "STOPPED"


class NativeWorkerProcess:
    """Serial NDJSON data plane plus an out-of-band cancellation control pipe abstraction."""

    def __init__(
        self,
        executable: Path,
        manifest: ModelManifest,
        *,
        startup_timeout: float = 30.0,
        terminate_grace: float = 2.0,
        terminal_frame_grace: float = .01,
        shutdown_grace: float = 2.0,
    ) -> None:
        if startup_timeout <= 0 or min(terminate_grace, terminal_frame_grace, shutdown_grace) < 0:
            raise ValueError("worker process timeouts must be non-negative and startup positive")
        self.command = [str(executable), str(manifest.path)]
        self.executable = executable.resolve()
        self._executable_digest = (
            "sha256:" + hashlib.sha256(self.executable.read_bytes()).hexdigest()
            if self.executable.is_file()
            else None
        )
        self.manifest = manifest
        self.startup_timeout = startup_timeout
        self.terminate_grace = terminate_grace
        self.terminal_frame_grace = terminal_frame_grace
        self.shutdown_grace = shutdown_grace
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.RLock()
        self.write_lock = threading.Lock()
        self._state = SupervisorState.STOPPED
        self._frames: queue.Queue[str | None] | None = None
        self._reader_thread: threading.Thread | None = None
        self.process_generation = 0
        self.model_loads_total = 0
        self._last_process_heartbeat: float | None = None
        self._last_request_progress: float | None = None

    @property
    def runtime_identity(self) -> dict[str, Any]:
        return {
            "revision": os.environ.get("MODEL_WORKER_REVISION", "unknown"),
            "model_digest": self.manifest.raw["gguf_sha256"],
            "native_executable_sha256": self._executable_digest,
            "process_generation": self.process_generation,
        }

    @property
    def supervisor_state(self) -> SupervisorState:
        with self.lock:
            if (
                self._state == SupervisorState.READY
                and (self.process is None or self.process.poll() is not None)
            ):
                self._state = SupervisorState.DEGRADED
            return self._state

    @property
    def last_process_heartbeat(self) -> float | None:
        with self.lock:
            return self._last_process_heartbeat

    @property
    def last_request_progress(self) -> float | None:
        with self.lock:
            return self._last_request_progress

    def start(self) -> None:
        with self.lock:
            if (
                self.process
                and self.process.poll() is None
                and self._state == SupervisorState.READY
            ):
                return
            self._start_locked(SupervisorState.STARTING)

    def _start_locked(self, state: SupervisorState) -> None:
        self._state = state
        process = None
        try:
            # llama.cpp can emit enough startup diagnostics to fill an unread pipe;
            # discard it here and rely on typed readiness/metrics from the supervisor.
            process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
            self.process = process
            line = self._readline_with_timeout(process, self.startup_timeout)
            if not line:
                raise WorkerError("worker_not_ready", "native worker exited before ready")
            try:
                ready = loads(line)
            except WorkerError as exc:
                raise WorkerError("worker_not_ready", "native worker emitted invalid ready JSON") from exc
            if (
                ready.get("type") != "ready"
                or ready.get("protocol_version") != "model-worker-ipc.v1"
                or ready.get("sequence") != 0
                or ready.get("model_id") != self.manifest.id
            ):
                raise WorkerError("worker_not_ready", "native worker did not emit a valid ready frame")
            self.process_generation += 1
            self.model_loads_total += 1
            frames: queue.Queue[str | None] = queue.Queue()
            generation = self.process_generation
            reader = threading.Thread(
                target=self._read_process_frames,
                args=(process, generation, frames),
                name=f"model-worker-frame-reader-{generation}",
                daemon=True,
            )
            self._frames = frames
            self._reader_thread = reader
            self._state = SupervisorState.READY
            self._last_process_heartbeat = time.monotonic()
            self._last_request_progress = None
            reader.start()
        except BaseException as exc:
            self._state = SupervisorState.DEGRADED
            if process is not None:
                if isinstance(exc, WorkerError) and "timed out" in exc.message:
                    self._terminate_then_kill(process)
                else:
                    self._kill(process)
            if self.process is process:
                self.process = None
            raise

    @staticmethod
    def _readline_with_timeout(process: subprocess.Popen[str], timeout: float) -> str:
        assert process.stdout is not None
        result: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

        def read() -> None:
            try:
                result.put(process.stdout.readline())
            except BaseException as exc:
                result.put(exc)

        reader = threading.Thread(target=read, name="model-worker-ready-reader", daemon=True)
        reader.start()
        try:
            value = result.get(timeout=timeout)
        except queue.Empty as exc:
            raise WorkerError("worker_not_ready", "native worker startup timed out") from exc
        if isinstance(value, BaseException):
            raise WorkerError("worker_not_ready", "native worker ready pipe failed") from value
        return value

    def _read_process_frames(
        self,
        process: subprocess.Popen[str],
        generation: int,
        frames: queue.Queue[str | None],
    ) -> None:
        assert process.stdout is not None
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                frames.put(line)
        finally:
            frames.put(None)
            with self.lock:
                if (
                    self.process is process
                    and self.process_generation == generation
                    and self._state == SupervisorState.READY
                ):
                    self._state = SupervisorState.DEGRADED

    @staticmethod
    def _wait_for_exit(process: subprocess.Popen[str], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(min(.01, max(0, deadline - time.monotonic())))
        return process.poll() is not None

    def _kill(self, process: subprocess.Popen[str]) -> bool:
        if process.poll() is None:
            process.kill()
        return self._wait_for_exit(process, self.terminate_grace)

    def _terminate_then_kill(self, process: subprocess.Popen[str]) -> bool:
        if process.poll() is not None:
            return True
        process.terminate()
        if self._wait_for_exit(process, self.terminate_grace):
            return True
        return self._kill(process)

    def execute(self, record: RequestRecord) -> Any:
        if record.cancel_event.is_set():
            raise WorkerError("cancelled", "request cancelled before native send")
        self.start()
        with self.lock:
            process = self.process
            frames = self._frames
            generation = self.process_generation
        assert process and process.stdin and frames
        verifier = FrameVerifier(record.request_id, record.attempt_id)
        started = time.monotonic()
        cancel_sent = False
        slow_consumer = False
        final_deltas: list[str] = []
        try:
            self._write_control(
                process,
                encode_frame(
                    "generate",
                    record.request_id,
                    record.attempt_id,
                    0,
                    request=asdict(record.request),
                ),
            )
            if record.cancel_event.is_set():
                self._send_cancel(process, record)
                cancel_sent = True
            while True:
                if record.cancel_event.is_set() and not cancel_sent:
                    self._send_cancel(process, record)
                    cancel_sent = True
                try:
                    line = frames.get(timeout=.05)
                except queue.Empty:
                    with self.lock:
                        generation_changed = (
                            self.process is not process
                            or self.process_generation != generation
                        )
                    if generation_changed or process.poll() is not None:
                        raise WorkerError("worker_crashed", "worker generation changed before completion")
                    continue
                if line is None:
                    raise WorkerError("worker_crashed", "worker pipe closed before completion")
                frame = verifier.verify(line)
                if frame["type"] == "final_delta":
                    final_deltas.append(frame["delta"])
                if frame["type"] in _EVENT_FRAME_TYPES:
                    try:
                        publish_result = record.event_sink.publish(
                            InferenceEvent.from_ipc_frame(frame)
                        )
                    except EventValidationError as exc:
                        raise WorkerError(
                            "worker_crashed",
                            "native worker emitted an invalid inference event",
                        ) from exc
                    if publish_result is PublishResult.SLOW_CONSUMER and not slow_consumer:
                        slow_consumer = True
                        record.cancel_event.set()
                        if not cancel_sent:
                            self._send_cancel(process, record)
                            cancel_sent = True
                observed_at = time.monotonic()
                with self.lock:
                    if frame["type"] == "heartbeat":
                        self._last_process_heartbeat = observed_at
                    if frame["type"] in {"started", "phase", "final_delta", "progress"}:
                        self._last_request_progress = observed_at
                if frame["type"] == "completed":
                    self._reject_extra_terminal_frame(frames)
                    if "".join(final_deltas) != frame["final_text"]:
                        raise WorkerError(
                            "worker_crashed",
                            "IPC final deltas do not match completed final_text",
                        )
                    if slow_consumer:
                        raise WorkerError(
                            "slow_consumer",
                            "stream consumer did not keep up with model output",
                        )
                    try:
                        output = loads(frame["final_text"])
                    except WorkerError as exc:
                        raise WorkerError("output_invalid", "final output is not strict JSON") from exc
                    errors = validate_output(output, record.request.contract)
                    if errors:
                        raise WorkerError("output_invalid", "final output violates normalized contract", details=errors)
                    elapsed = (time.monotonic() - started) * 1000
                    native_timing = frame.get("timing", {})
                    timing = {
                        "queue_ms": 0,
                        "prompt_decode_ms": native_timing.get("prompt_decode_ms", 0),
                        "generation_ms": native_timing.get("generation_ms", elapsed),
                        "total_ms": elapsed,
                    }
                    model = {
                        "id": self.manifest.id,
                        "manifest_digest": self.manifest.digest,
                        "runtime_build": self.manifest.raw["runtime_build"],
                        **self.runtime_identity,
                        "process_generation": generation,
                    }
                    return GenerateResult(record.request_id, record.attempt_id, "completed", True, True, output, frame["usage"], timing, model)
                if frame["type"] == "failed":
                    self._reject_extra_terminal_frame(frames)
                    if slow_consumer:
                        raise WorkerError(
                            "slow_consumer",
                            "stream consumer did not keep up with model output",
                        )
                    code = frame.get("error", "decode_failed")
                    if code not in {"cancelled", "context_overflow", "protocol_violation", "decode_failed"}: code = "decode_failed"
                    raise WorkerError(code, frame.get("detail", "native request failed"))
        except WorkerError as exc:
            if exc.code == "worker_crashed":
                self._recover_protocol_failure(process, generation)
            raise

    def _reject_extra_terminal_frame(self, frames: queue.Queue[str | None]) -> None:
        try:
            extra = frames.get(timeout=self.terminal_frame_grace)
        except queue.Empty:
            return
        if extra is not None:
            raise WorkerError("worker_crashed", "duplicate or trailing IPC frame after terminal")

    def _write_control(self, process: subprocess.Popen[str], frame: str) -> None:
        if process.stdin is None:
            raise WorkerError("worker_crashed", "worker control pipe is unavailable")
        try:
            with self.write_lock:
                process.stdin.write(frame + "\n")
                process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerError("worker_crashed", "worker control pipe write failed") from exc

    def _send_cancel(self, process: subprocess.Popen[str], record: RequestRecord) -> None:
        self._write_control(
            process,
            encode_frame("cancel", record.request_id, record.attempt_id, 0),
        )

    def cancel(self, record: RequestRecord) -> None:
        record.cancel_event.set()
        with self.lock:
            process = self.process
        if process and process.poll() is None and process.stdin:
            try:
                self._send_cancel(process, record)
            except WorkerError:
                pass

    def _recover_protocol_failure(
        self,
        process: subprocess.Popen[str],
        generation: int,
    ) -> bool:
        with self.lock:
            if self.process is not process or self.process_generation != generation:
                return self._state == SupervisorState.READY
            self._state = SupervisorState.RESTARTING
            if not self._kill(process):
                self._state = SupervisorState.DEGRADED
                return False
            self.process = None
            try:
                self._start_locked(SupervisorState.RESTARTING)
                return True
            except (WorkerError, OSError):
                self._state = SupervisorState.DEGRADED
                return False

    def kill_and_restart(self) -> bool:
        with self.lock:
            self._state = SupervisorState.RESTARTING
            if self.process and self.process.poll() is None:
                if not self._kill(self.process):
                    self._state = SupervisorState.DEGRADED
                    return False
            self.process = None
            try:
                self._start_locked(SupervisorState.RESTARTING)
                return True
            except (WorkerError, OSError):
                self._state = SupervisorState.DEGRADED
                return False

    def shutdown(self) -> None:
        reader = None
        process = None
        stopped = True
        with self.lock:
            if (
                self._state == SupervisorState.STOPPED
                and self.process is None
                and (self._reader_thread is None or not self._reader_thread.is_alive())
            ):
                return
            self._state = SupervisorState.DRAINING
            process = self.process
            reader = self._reader_thread
            if process and process.poll() is None:
                try:
                    self._write_control(process, encode_frame("shutdown", "", "", 0))
                except WorkerError:
                    pass
                deadline = time.monotonic() + self.shutdown_grace
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(min(.01, max(0, deadline - time.monotonic())))
                if process.poll() is None:
                    stopped = self._terminate_then_kill(process)
                else:
                    stopped = True
            if stopped:
                self.process = None
                self._frames = None
        if reader is not None:
            reader.join(self.terminate_grace)
        with self.lock:
            reader_stopped = reader is None or not reader.is_alive()
            if stopped and reader_stopped:
                self._reader_thread = None
                self._state = SupervisorState.STOPPED
            else:
                self._state = SupervisorState.DEGRADED
