from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from inference_runtime import SchedulingMetadata
from inference_runtime.adapters import (
    ManagedBackendError,
    SGLangManagedBackend,
    UrllibJSONTransport,
    VLLMManagedBackend,
)
from model_worker.preflight import preflight


class Sink:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)


class ConformanceHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.calls.append(
            {
                "path": self.path,
                "request_id": self.headers.get("X-Request-ID"),
                "payload": payload,
            }
        )
        if self.server.status != 200:
            self.send_response(self.server.status)
            self.end_headers()
            return
        body = json.dumps(
            {
                "id": "provider-http-1",
                "model": payload["model"],
                "system_fingerprint": "http-conformance",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": '{"result":"ok"}',
                        },
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        del format, args


def scheduling():
    return SchedulingMetadata(
        "http-conformance-1",
        "workflow",
        "agent",
        "throughput",
        1,
        None,
    )


@pytest.fixture
def conformance_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), ConformanceHandler)
    server.calls = []
    server.status = 200
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("adapter", [VLLMManagedBackend, SGLangManagedBackend])
def test_openai_compatible_backends_pass_real_http_boundary_conformance(
    adapter,
    conformance_server,
    manifest,
    request_body,
):
    host, port = conformance_server.server_address
    transport = UrllibJSONTransport(f"http://{host}:{port}")
    target = adapter(
        transport,
        models=("qwen35-9b-local",),
        max_context_tokens=1024,
        max_output_tokens=128,
        max_concurrent_requests=2,
    )

    result = target.generate(
        preflight(request_body, manifest),
        scheduling=scheduling(),
        events=Sink(),
    )

    assert result.output == {"result": "ok"}
    assert result.model["backend"] in {"vllm", "sglang"}
    call = conformance_server.calls[0]
    assert call["path"] == "/v1/chat/completions"
    assert call["request_id"] == "http-conformance-1"
    assert call["payload"]["response_format"]["json_schema"]["strict"] is True
    assert call["payload"]["stream"] is False
    assert not target.capabilities.supports_cancellation


def test_http_transport_classifies_provider_503_as_retryable(
    conformance_server,
    manifest,
    request_body,
):
    conformance_server.status = 503
    host, port = conformance_server.server_address
    target = VLLMManagedBackend(
        UrllibJSONTransport(f"http://{host}:{port}"),
        models=("qwen35-9b-local",),
        max_context_tokens=1024,
        max_output_tokens=128,
        max_concurrent_requests=1,
    )

    with pytest.raises(ManagedBackendError) as captured:
        target.generate(
            preflight(request_body, manifest),
            scheduling=scheduling(),
            events=Sink(),
        )

    assert captured.value.code == "provider_http_error"
    assert captured.value.retryable
