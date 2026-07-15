from __future__ import annotations

import json
import select
import socket
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .dispatcher import Dispatcher
from .artifacts import ArtifactStore
from .errors import WorkerError
from .events import BoundedRequestEventBuffer, InferenceEvent
from .maintenance import MaintenanceRunner
from .manifest import ModelManifest
from .metrics import Metrics
from .preflight import preflight
from .request_registry import Lifecycle, RequestRecord, TERMINAL
from .security import ExposurePolicy
from .strict_json import loads
from .terminal_metrics import TerminalMetricsObserver


class _ClientDisconnected(ConnectionError):
    pass


class ModelWorkerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        dispatcher: Dispatcher,
        manifest: ModelManifest,
        policy: ExposurePolicy,
        *,
        max_handlers: int = 16,
        read_timeout: float = 10.0,
        artifact_root: Path = Path("artifacts"),
        maintenance_interval: float = 60.0,
        registry_terminal_ttl: float | None = None,
        max_registry_prune: int = 256,
        max_artifact_removals: int = 64,
    ):
        import threading
        policy.validate()
        self.dispatcher, self.manifest, self.policy = dispatcher, manifest, policy
        self.metrics = Metrics()
        self.artifacts = ArtifactStore(artifact_root, total_quota=manifest.limits["total_artifact_bytes"], retention_seconds=manifest.limits["artifact_retention_seconds"])
        self.handler_slots = threading.BoundedSemaphore(max_handlers)
        self.read_timeout = read_timeout
        super().__init__(address, ModelWorkerHandler)
        self._terminal_metrics = TerminalMetricsObserver(self.metrics)
        self.dispatcher.registry.add_terminal_observer(self._terminal_metrics)
        terminal_ttl = (
            float(manifest.limits["artifact_retention_seconds"])
            if registry_terminal_ttl is None
            else registry_terminal_ttl
        )
        self.maintenance = MaintenanceRunner(
            self.dispatcher.registry,
            self.artifacts,
            interval_seconds=maintenance_interval,
            terminal_ttl_seconds=terminal_ttl,
            max_registry_prune=max_registry_prune,
            max_artifact_removals=max_artifact_removals,
            metrics=self.metrics,
        )
        self.maintenance.start()

    def process_request_thread(self, request, client_address):
        if not self.handler_slots.acquire(blocking=False):
            try: request.close()
            finally: return
        try: super().process_request_thread(request, client_address)
        finally: self.handler_slots.release()

    def handle_error(self, request, client_address):
        # Disconnects and malformed sockets are contained at the HTTP boundary;
        # request handlers convert in-flight work to cancellation separately.
        return

    def server_close(self):
        self.maintenance.stop(timeout=2.0)
        self.dispatcher.shutdown()
        self.dispatcher.registry.remove_terminal_observer(self._terminal_metrics)
        super().server_close()


class ModelWorkerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ModelWorker/1"

    @property
    def app(self) -> ModelWorkerHTTPServer:
        return self.server  # type: ignore[return-value]

    def setup(self):
        super().setup()
        self.connection.settimeout(self.app.read_timeout)

    def log_message(self, fmt: str, *args: Any) -> None:
        return  # Structured logging is owned by the service launcher; never logs headers/body.

    def _json(self, status: int, payload: Any, **headers: str) -> None:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        for name, value in headers.items(): self.send_header(name, value)
        self.end_headers(); self.wfile.write(raw)

    def _error(self, error: WorkerError) -> None:
        self._json(error.http_status, {"protocol_version": "model-worker.v1", "error": error.as_dict()})

    def _start_sse(self) -> None:
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as exc:
            raise _ClientDisconnected from exc
        self.close_connection = True

    def _write_sse(self, event: str, payload: Any) -> None:
        raw = (
            f"event: {event}\n"
            f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'), default=str)}\n\n"
        ).encode("utf-8")
        try:
            self.wfile.write(raw)
            self.wfile.flush()
        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            socket.timeout,
            TimeoutError,
            OSError,
        ) as exc:
            raise _ClientDisconnected from exc

    def _client_disconnected(self) -> bool:
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            if not readable:
                return False
            return self.connection.recv(1, socket.MSG_PEEK) == b""
        except (BlockingIOError, InterruptedError, socket.timeout):
            return False
        except (ConnectionResetError, ConnectionAbortedError, OSError, ValueError):
            return True

    def _wait_non_stream(self, record: RequestRecord) -> None:
        while record.lifecycle not in TERMINAL:
            if self._client_disconnected():
                raise _ClientDisconnected
            self.app.dispatcher.wait(record, timeout=.02)

    def _drain_stream(self, record: RequestRecord) -> None:
        if not isinstance(record.event_sink, BoundedRequestEventBuffer):
            raise RuntimeError("stream request is missing its bounded event buffer")
        while True:
            if self._client_disconnected():
                raise _ClientDisconnected
            if record.lifecycle in TERMINAL:
                for event in record.event_sink.drain():
                    self._write_inference_event(event)
                return
            event = record.event_sink.get(timeout=.02)
            if event is not None:
                self._write_inference_event(event)

    def _write_inference_event(self, event: InferenceEvent) -> None:
        self._write_sse(event.event_type.value, event.as_dict())

    def _finalize_artifact(self, artifact, record: RequestRecord) -> tuple[dict[str, Any], int]:
        try:
            payload, status = self._record_response(record)
            artifact.write_result(payload)
            return payload, status
        finally:
            self.app.artifacts.finish(artifact)

    def _settle_request(
        self,
        record,
        artifact,
        artifact_finished: bool,
        *,
        disconnected: bool,
    ) -> bool:
        if disconnected:
            self.close_connection = True
        if record is None:
            return artifact_finished
        if record.lifecycle not in TERMINAL:
            try:
                self.app.dispatcher.cancel(record.request_id)
            except Exception:
                # Registry cancellation sets cancel_event before a backend hook
                # can fail. The lifecycle deadline remains the terminal bound.
                pass
            self.app.dispatcher.wait(record)
        if artifact is not None and not artifact_finished:
            try:
                self._finalize_artifact(artifact, record)
            except Exception:
                # Never synthesize a terminal result. ArtifactStore.finish still
                # runs in _finalize_artifact, leaving an honest incomplete attempt
                # if the true terminal payload cannot be persisted.
                pass
            artifact_finished = True
        return artifact_finished

    def _authorized(self) -> bool:
        if self.app.policy.authorized(self.headers.get("Authorization")): return True
        self._json(401, {"error": {"code": "unauthorized", "retryable": False}}, **{"WWW-Authenticate": "Bearer"})
        return False

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/live": self._json(200, {"status": "live"}); return
        if not self._authorized(): return
        if path == "/ready":
            state = self.app.dispatcher.supervisor_state
            ready = state == "READY"
            identity = self.app.dispatcher.runtime_identity
            identity["process_generation"] = self.app.dispatcher.process_generation
            self._json(200 if ready else 503, {"status": "ready" if ready else "not_ready", "supervisor_state": state, "model_id": self.app.manifest.id, "manifest_digest": self.app.manifest.digest, "runtime_build": self.app.manifest.raw["runtime_build"], **identity}); return
        if path == "/metrics":
            raw = self.app.metrics.render().encode()
            self.send_response(200); self.send_header("Content-Type", "text/plain; version=0.0.4"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw); return
        self._json(404, {"error": {"code": "not_found"}})

    def do_POST(self) -> None:
        if not self._authorized(): return
        path = urlsplit(self.path).path
        if path.startswith("/v1/model/cancel/"):
            request_id = path.rsplit("/", 1)[-1]
            changed = request_id.isalnum() and self.app.dispatcher.cancel(request_id)
            self._json(202 if changed else 404, {"request_id": request_id, "cancellation_requested": bool(changed)}); return
        if path == "/v1/controlled/generate":
            self._json(410, {"error": {"code": "legacy_endpoint", "message": "use /v1/model/generate"}}, Warning='299 - "legacy endpoint removed from release gate"'); return
        if path != "/v1/model/generate": self._json(404, {"error": {"code": "not_found"}}); return
        record = None
        artifact = None
        artifact_finished = False
        stream_started = False
        try:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None or not raw_length.isdecimal(): raise WorkerError("invalid_request", "valid Content-Length is required")
            length = int(raw_length)
            if length > self.app.manifest.limits["input_bytes"] + self.app.manifest.limits["schema_bytes"] + 65536:
                raise WorkerError("request_too_large", "HTTP body exceeds configured maximum")
            body = self.rfile.read(length)
            if len(body) != length: raise WorkerError("invalid_request", "incomplete request body")
            prepared = preflight(loads(body, too_large=length), self.app.manifest)
            record = self.app.dispatcher.submit(prepared)
            artifact = self.app.artifacts.begin(record.request_id, record.attempt_id, self.app.manifest.limits["max_artifact_bytes"])
            artifact.write_manifest(asdict(prepared.request), prepared.request.output_contract.schema, {"manifest_digest": self.app.manifest.digest, "runtime_build": self.app.manifest.raw["runtime_build"], "prompt_hash": prepared.prompt_hash, "prompt_version": prepared.prompt_version}, asdict(prepared.limits), record.timestamps)
            if prepared.request.stream.enabled:
                self._start_sse()
                stream_started = True
                self._write_sse(
                    "queued",
                    {"request_id": record.request_id, "attempt_id": record.attempt_id},
                )
                self._drain_stream(record)
            else:
                self._wait_non_stream(record)
            artifact_finished = True
            payload, status = self._finalize_artifact(artifact, record)
            if prepared.request.stream.enabled:
                event = "result" if record.lifecycle == Lifecycle.COMPLETED else "error"
                self._write_sse(event, payload)
            else: self._json(status, payload)
        except _ClientDisconnected:
            artifact_finished = self._settle_request(
                record,
                artifact,
                artifact_finished,
                disconnected=True,
            )
        except WorkerError as exc:
            if record is not None:
                artifact_finished = self._settle_request(
                    record,
                    artifact,
                    artifact_finished,
                    disconnected=False,
                )
            if record is None:
                self.app.metrics.inc("requests_total", error_class=exc.code)
            if stream_started:
                try:
                    self._write_sse(
                        "error",
                        {"protocol_version": "model-worker.v1", "error": exc.as_dict()},
                    )
                except _ClientDisconnected:
                    artifact_finished = self._settle_request(
                        record,
                        artifact,
                        artifact_finished,
                        disconnected=True,
                    )
            else:
                self._error(exc)
        except (socket.timeout, TimeoutError):
            if record is None:
                error = WorkerError("invalid_request", "request body read timeout")
                self.app.metrics.inc("requests_total", error_class=error.code)
                self._error(error)
            else:
                artifact_finished = self._settle_request(
                    record,
                    artifact,
                    artifact_finished,
                    disconnected=True,
                )
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            artifact_finished = self._settle_request(
                record,
                artifact,
                artifact_finished,
                disconnected=True,
            )
        except Exception:
            artifact_finished = self._settle_request(
                record,
                artifact,
                artifact_finished,
                disconnected=False,
            )
            error = WorkerError("worker_crashed", "internal request handling failure")
            if record is None:
                self.app.metrics.inc("requests_total", error_class=error.code)
            if stream_started:
                try:
                    self._write_sse(
                        "error",
                        {"protocol_version": "model-worker.v1", "error": error.as_dict()},
                    )
                except _ClientDisconnected:
                    self._settle_request(
                        record,
                        artifact,
                        artifact_finished,
                        disconnected=True,
                    )
            else:
                self._error(error)

    def _record_response(self, record) -> tuple[dict[str, Any], int]:
        if record.lifecycle not in TERMINAL:
            raise RuntimeError(f"cannot serialize non-terminal request {record.lifecycle}")
        if record.lifecycle == Lifecycle.COMPLETED:
            result = record.result
            payload = result.as_dict() if hasattr(result, "as_dict") else result
            return payload, 200
        if isinstance(record.error, WorkerError):
            error = record.error
            return {"protocol_version": "model-worker.v1", "request_id": record.request_id, "attempt_id": record.attempt_id, "termination": record.lifecycle.value.lower(), "protocol_valid": False, "output_valid": False, "output": None, "model": self._model_identity(), "error": error.as_dict()}, error.http_status
        code = record.error or "worker_crashed"
        if code == "queue_timeout": error = WorkerError("queue_timeout", "queue deadline exceeded")
        elif code == "deadline_exceeded": error = WorkerError("deadline_exceeded", "execution deadline exceeded")
        elif code == "cancelled": error = WorkerError("cancelled", "request cancelled")
        elif code in {"decode_failed", "worker_crashed", "context_overflow", "protocol_violation", "output_invalid", "slow_consumer", "shutdown", "queue_full"}:
            error = WorkerError(code, code.replace("_", " "))
        else: error = WorkerError("worker_crashed", "worker request failed")
        return {"protocol_version": "model-worker.v1", "request_id": record.request_id, "attempt_id": record.attempt_id, "termination": record.lifecycle.value.lower(), "protocol_valid": False, "output_valid": False, "output": None, "model": self._model_identity(), "error": error.as_dict()}, error.http_status

    def _model_identity(self) -> dict[str, Any]:
        identity = self.app.dispatcher.runtime_identity
        identity["process_generation"] = self.app.dispatcher.process_generation
        return {
            "id": self.app.manifest.id,
            "manifest_digest": self.app.manifest.digest,
            "runtime_build": self.app.manifest.raw["runtime_build"],
            **identity,
        }
