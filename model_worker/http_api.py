from __future__ import annotations

import json
import socket
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .dispatcher import Dispatcher
from .artifacts import ArtifactStore
from .errors import WorkerError
from .manifest import ModelManifest
from .metrics import Metrics
from .preflight import preflight
from .request_registry import Lifecycle
from .security import ExposurePolicy
from .strict_json import loads


class ModelWorkerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], dispatcher: Dispatcher, manifest: ModelManifest, policy: ExposurePolicy, *, max_handlers: int = 16, read_timeout: float = 10.0, artifact_root: Path = Path("artifacts")):
        import threading
        policy.validate()
        self.dispatcher, self.manifest, self.policy = dispatcher, manifest, policy
        self.metrics, self.ready = Metrics(), True
        self.artifacts = ArtifactStore(artifact_root, total_quota=manifest.limits["total_artifact_bytes"], retention_seconds=manifest.limits["artifact_retention_seconds"])
        self.handler_slots = threading.BoundedSemaphore(max_handlers)
        self.read_timeout = read_timeout
        super().__init__(address, ModelWorkerHandler)

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
        self.ready = False
        self.dispatcher.shutdown()
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

    def _authorized(self) -> bool:
        if self.app.policy.authorized(self.headers.get("Authorization")): return True
        self._json(401, {"error": {"code": "unauthorized", "retryable": False}}, **{"WWW-Authenticate": "Bearer"})
        return False

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/live": self._json(200, {"status": "live"}); return
        if not self._authorized(): return
        if path == "/ready":
            self._json(200 if self.app.ready else 503, {"status": "ready" if self.app.ready else "not_ready", "model_id": self.app.manifest.id, "manifest_digest": self.app.manifest.digest, "runtime_build": self.app.manifest.raw["runtime_build"]}); return
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
            artifact.write_manifest(asdict(prepared.request), prepared.request.output_contract.schema, {"manifest_digest": self.app.manifest.digest, "runtime_build": self.app.manifest.raw["runtime_build"]}, asdict(prepared.limits), record.timestamps)
            if prepared.request.stream.enabled:
                self.send_response(200); self.send_header("Content-Type", "text/event-stream; charset=utf-8"); self.send_header("Cache-Control", "no-store"); self.end_headers()
                self.wfile.write(f"event: queued\ndata: {json.dumps({'request_id': record.request_id, 'attempt_id': record.attempt_id})}\n\n".encode()); self.wfile.flush()
            self.app.dispatcher.wait(record, timeout=(prepared.limits.queue_timeout_ms + prepared.limits.execution_timeout_ms + 2000) / 1000)
            payload, status = self._record_response(record)
            artifact.write_result(payload)
            self.app.artifacts.finish(artifact)
            if prepared.request.stream.enabled:
                event = "result" if record.lifecycle == Lifecycle.COMPLETED else "error"
                self.wfile.write(f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n".encode("utf-8")); self.wfile.flush()
            else: self._json(status, payload)
        except WorkerError as exc:
            self.app.metrics.inc("requests_total", error_class=exc.code)
            self._error(exc)
        except (socket.timeout, TimeoutError):
            self._error(WorkerError("invalid_request", "request body read timeout"))
        except (BrokenPipeError, ConnectionResetError):
            if "record" in locals():
                self.app.dispatcher.cancel(record.request_id)
                self.app.dispatcher.wait(record, timeout=2)
            if "artifact" in locals():
                if not artifact._terminal: artifact.write_result({"termination":"cancelled","request_id":record.request_id,"attempt_id":record.attempt_id})
                self.app.artifacts.finish(artifact)
        except Exception:
            self._error(WorkerError("worker_crashed", "internal request handling failure"))

    def _record_response(self, record) -> tuple[dict[str, Any], int]:
        if record.lifecycle == Lifecycle.COMPLETED:
            self.app.metrics.inc("requests_total", termination="completed")
            result = record.result
            payload = result.as_dict() if hasattr(result, "as_dict") else result
            queued_at = record.timestamps.get("QUEUED", record.timestamps.get("RECEIVED", 0))
            running_at = record.timestamps.get("RUNNING", queued_at)
            self.app.metrics.observe("queue_wait_ms", max(0, running_at - queued_at) * 1000)
            for name in ("prompt_decode_ms", "generation_ms", "total_ms"):
                if name in payload.get("timing", {}): self.app.metrics.observe(name, payload["timing"][name])
            for name in ("prompt_tokens", "reasoning_tokens", "final_tokens", "sampled_tokens", "context_headroom"):
                if name in payload.get("usage", {}): self.app.metrics.observe(name, payload["usage"][name])
            return payload, 200
        if isinstance(record.error, WorkerError):
            error = record.error
            self.app.metrics.inc("requests_total", error_class=error.code)
            return {"protocol_version": "model-worker.v1", "request_id": record.request_id, "attempt_id": record.attempt_id, "termination": record.lifecycle.value.lower(), "protocol_valid": False, "output_valid": False, "output": None, "error": error.as_dict()}, error.http_status
        code = record.error or "worker_crashed"
        if code == "queue_timeout": error = WorkerError("queue_timeout", "queue deadline exceeded")
        elif code == "deadline_exceeded": error = WorkerError("deadline_exceeded", "execution deadline exceeded")
        elif code == "cancelled": error = WorkerError("cancelled", "request cancelled")
        elif code in {"decode_failed", "worker_crashed", "context_overflow", "protocol_violation", "output_invalid"}:
            error = WorkerError(code, code.replace("_", " "))
        else: error = WorkerError("worker_crashed", "worker request failed")
        self.app.metrics.inc("requests_total", error_class=error.code)
        return {"protocol_version": "model-worker.v1", "request_id": record.request_id, "attempt_id": record.attempt_id, "termination": record.lifecycle.value.lower(), "protocol_valid": False, "output_valid": False, "output": None, "error": error.as_dict()}, error.http_status
