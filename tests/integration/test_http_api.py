from __future__ import annotations

import http.client
import json
import socket
import threading
import time

import pytest

from model_worker.dispatcher import Dispatcher
from model_worker.errors import WorkerError
from model_worker.events import InferenceEvent, InferenceEventType, PublishResult
from model_worker.http_api import ModelWorkerHTTPServer
from model_worker.request_registry import Lifecycle
from model_worker.security import ExposurePolicy
from model_worker.worker_process import SupervisorState


class Worker:
    def execute(self, record): return {"protocol_version":"model-worker.v1","request_id":record.request_id,"attempt_id":record.attempt_id,"termination":"completed","protocol_valid":True,"output_valid":True,"output":{"result":"ok"},"error":None}
    def cancel(self, record): record.cancel_event.set()
    def kill_and_restart(self): return True
    def shutdown(self): pass


def test_http_preflight_health_and_oversize(manifest, request_body, tmp_path):
    dispatcher = Dispatcher(Worker(), capacity=2)
    server = ModelWorkerHTTPServer(("127.0.0.1", 0), dispatcher, manifest, ExposurePolicy(), read_timeout=1, artifact_root=tmp_path / "artifacts")
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    connection = http.client.HTTPConnection(*server.server_address, timeout=2)
    connection.request("GET", "/ready"); ready = connection.getresponse()
    assert ready.status == 200 and json.loads(ready.read())["manifest_digest"].startswith("sha256:")
    body = json.dumps(request_body).encode()
    connection.request("POST", "/v1/model/generate", body=body, headers={"Content-Type":"application/json","Content-Length":str(len(body))})
    response = connection.getresponse(); payload=json.loads(response.read())
    assert response.status == 200 and payload["output_valid"] is True and "accepted" not in payload
    assert "model_worker_queue_wait_ms_count 1" in server.metrics.render()
    assert server.metrics.render().count(
        'model_worker_requests_total{termination="completed"} 1'
    ) == 1
    connection.request("POST", "/v1/model/generate", body=b"{}", headers={"Content-Length":str(manifest.limits["input_bytes"] + manifest.limits["schema_bytes"] + 65537)})
    oversized=connection.getresponse(); assert oversized.status == 413; oversized.read()
    connection.close(); server.shutdown(); server.server_close(); thread.join(1)


def test_preflight_failure_never_enqueues(manifest, request_body, tmp_path):
    worker=Worker(); dispatcher=Dispatcher(worker, capacity=1)
    server=ModelWorkerHTTPServer(("127.0.0.1",0),dispatcher,manifest,ExposurePolicy(),artifact_root=tmp_path / "artifacts")
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    bad=dict(request_body); bad["stream"]={"enabled":"false"}; raw=json.dumps(bad).encode()
    connection=http.client.HTTPConnection(*server.server_address,timeout=2)
    connection.request("POST","/v1/model/generate",body=raw,headers={"Content-Length":str(len(raw))})
    response=connection.getresponse(); assert response.status==400; response.read()
    assert dispatcher.registry.snapshot() == ()
    assert 'model_worker_requests_total{error_class="invalid_request"} 1' in server.metrics.render()
    connection.close(); server.shutdown(); server.server_close(); thread.join(1)


def running_server(manifest, tmp_path, worker=None, policy=None, **dispatcher_options):
    dispatcher = Dispatcher(
        worker or Worker(),
        capacity=2,
        watchdog_grace_ms=5,
        **dispatcher_options,
    )
    server = ModelWorkerHTTPServer(
        ("127.0.0.1", 0),
        dispatcher,
        manifest,
        policy or ExposurePolicy(),
        read_timeout=1,
        artifact_root=tmp_path / "artifacts",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def close_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(1)
    assert not thread.is_alive()


def read_json(response):
    return json.loads(response.read())


def read_sse(response):
    blocks = response.read().decode("utf-8").strip().split("\n\n")
    parsed = []
    for block in blocks:
        if not block:
            continue
        lines = block.splitlines()
        assert len(lines) == 2
        parsed.append(
            (
                lines[0].removeprefix("event: "),
                json.loads(lines[1].removeprefix("data: ")),
            )
        )
    return parsed


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.005)
    return predicate()


def test_http_auth_liveness_metrics_legacy_cancel_and_not_found(manifest, tmp_path):
    policy = ExposurePolicy("127.0.0.1", "secret")
    server, thread = running_server(manifest, tmp_path, policy=policy)
    connection = http.client.HTTPConnection(*server.server_address, timeout=2)
    try:
        connection.request("GET", "/live")
        live = connection.getresponse()
        assert live.status == 200 and read_json(live) == {"status": "live"}

        connection.request("GET", "/ready")
        unauthorized = connection.getresponse()
        assert unauthorized.status == 401
        assert unauthorized.getheader("WWW-Authenticate") == "Bearer"
        unauthorized.read()

        auth = {"Authorization": "Bearer secret"}
        connection.request("GET", "/metrics", headers=auth)
        metrics = connection.getresponse()
        assert metrics.status == 200
        assert metrics.getheader("Content-Type") == "text/plain; version=0.0.4"
        metrics.read()

        connection.request("GET", "/missing", headers=auth)
        missing = connection.getresponse()
        assert missing.status == 404 and read_json(missing)["error"]["code"] == "not_found"

        connection.request("POST", "/v1/controlled/generate", body=b"", headers={**auth, "Content-Length": "0"})
        legacy = connection.getresponse()
        assert legacy.status == 410 and legacy.getheader("Warning")
        legacy.read()

        connection.request("POST", "/v1/model/cancel/not-found", body=b"", headers={**auth, "Content-Length": "0"})
        cancelled = connection.getresponse()
        assert cancelled.status == 404 and read_json(cancelled)["cancellation_requested"] is False

        connection.request("POST", "/missing", body=b"", headers={**auth, "Content-Length": "0"})
        missing_post = connection.getresponse()
        assert missing_post.status == 404
        missing_post.read()
    finally:
        connection.close()
        close_server(server, thread)


class ErrorWorker(Worker):
    def __init__(self, error):
        self.error = error

    def execute(self, record):
        raise self.error


class StreamingEventWorker(Worker):
    def __init__(self, *, delta_count=1, include_telemetry=True):
        self.delta_count = delta_count
        self.include_telemetry = include_telemetry
        self.publish_results = []

    def execute(self, record):
        sequence = 0

        def publish(event_type, **payload):
            nonlocal sequence
            result = record.event_sink.publish(
                InferenceEvent(
                    event_type,
                    record.request_id,
                    record.attempt_id,
                    sequence,
                    **payload,
                )
            )
            self.publish_results.append(result)
            sequence += 1

        publish(InferenceEventType.STARTED)
        if self.include_telemetry:
            publish(
                InferenceEventType.PROGRESS,
                phase="prompt_decode",
                tokens=1,
            )
            publish(InferenceEventType.HEARTBEAT, tokens=1)
        publish(InferenceEventType.PHASE, phase="final")
        for index in range(self.delta_count):
            publish(InferenceEventType.FINAL_DELTA, delta=f"{index},")
        return super().execute(record)


class OverflowEventWorker(Worker):
    def execute(self, record):
        result = record.event_sink.publish(
            InferenceEvent(
                InferenceEventType.STARTED,
                record.request_id,
                record.attempt_id,
                0,
            )
        )
        assert result is PublishResult.SLOW_CONSUMER
        record.cancel_event.set()
        raise WorkerError("slow_consumer", "client did not drain stream")


def post_request(connection, request_body):
    raw = json.dumps(request_body).encode()
    connection.request(
        "POST",
        "/v1/model/generate",
        body=raw,
        headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
    )
    return connection.getresponse()


def test_http_preserves_typed_worker_error_and_contains_crash(manifest, request_body, tmp_path):
    cases = [
        (WorkerError("output_invalid", "bad output"), 422, "output_invalid"),
        (RuntimeError("boom"), 502, "worker_crashed"),
    ]
    for index, (error, expected_status, expected_code) in enumerate(cases):
        case_root = tmp_path / str(index)
        server, thread = running_server(manifest, case_root, worker=ErrorWorker(error))
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        try:
            response = post_request(connection, request_body)
            payload = read_json(response)
            assert response.status == expected_status
            assert payload["error"]["code"] == expected_code
            assert payload["protocol_valid"] is False and payload["output_valid"] is False
            rendered = server.metrics.render()
            assert rendered.count(
                f'model_worker_requests_total{{error_class="{expected_code}",termination="failed"}} 1'
            ) == 1
        finally:
            connection.close()
            close_server(server, thread)


def test_http_stream_emits_queued_and_terminal_events_without_reasoning(manifest, request_body, tmp_path):
    request_body["stream"] = {"enabled": True, "include_reasoning": False}
    server, thread = running_server(manifest, tmp_path)
    connection = http.client.HTTPConnection(*server.server_address, timeout=2)
    try:
        response = post_request(connection, request_body)
        assert response.status == 200
        assert response.getheader("Content-Type") == "text/event-stream; charset=utf-8"
        lines = [response.fp.readline().decode("utf-8").strip() for _ in range(6)]
        assert lines[0] == "event: queued"
        assert lines[3] == "event: result"
        assert all("reasoning" not in line for line in lines)
        result = json.loads(lines[4].removeprefix("data: "))
        assert result["output"] == {"result": "ok"}
    finally:
        connection.close()
        close_server(server, thread)


def test_http_stream_uses_error_event_after_headers(manifest, request_body, tmp_path):
    request_body["stream"] = {"enabled": True, "include_reasoning": False}
    server, thread = running_server(manifest, tmp_path, worker=ErrorWorker(WorkerError("decode_failed", "decode")))
    connection = http.client.HTTPConnection(*server.server_address, timeout=2)
    try:
        response = post_request(connection, request_body)
        assert response.status == 200
        lines = [response.fp.readline().decode("utf-8").strip() for _ in range(6)]
        assert lines[3] == "event: error"
        error = json.loads(lines[4].removeprefix("data: "))
        assert error["error"]["code"] == "decode_failed"
    finally:
        connection.close()
        close_server(server, thread)


def test_http_sse_forwards_verified_inference_events_before_terminal_result(
    manifest,
    request_body,
    tmp_path,
):
    body = json.loads(json.dumps(request_body))
    body["stream"] = {"enabled": True, "include_reasoning": False}
    worker = StreamingEventWorker()
    server, thread = running_server(manifest, tmp_path, worker=worker)
    connection = http.client.HTTPConnection(*server.server_address, timeout=5)
    try:
        response = post_request(connection, body)
        assert response.status == 200
        events = read_sse(response)
        assert [name for name, _ in events] == [
            "queued",
            "started",
            "progress",
            "heartbeat",
            "phase",
            "final_delta",
            "result",
        ]
        forwarded = events[1:-1]
        assert [payload["sequence"] for _, payload in forwarded] == list(range(5))
        assert events[-1][1]["output"] == {"result": "ok"}
        assert all(result is PublishResult.ENQUEUED for result in worker.publish_results)
    finally:
        connection.close()
        close_server(server, thread)


def test_http_sse_streams_one_thousand_deltas_in_linear_source_order(
    manifest,
    request_body,
    tmp_path,
):
    body = json.loads(json.dumps(request_body))
    body["stream"] = {"enabled": True, "include_reasoning": False}
    worker = StreamingEventWorker(delta_count=1000, include_telemetry=False)
    server, thread = running_server(manifest, tmp_path, worker=worker)
    connection = http.client.HTTPConnection(*server.server_address, timeout=5)
    try:
        response = post_request(connection, body)
        events = read_sse(response)
        deltas = [payload for name, payload in events if name == "final_delta"]
        assert len(deltas) == 1000
        assert [payload["sequence"] for payload in deltas] == list(range(2, 1002))
        assert "".join(payload["delta"] for payload in deltas).startswith("0,1,2,")
        assert events[-1][0] == "result"
        assert all(result is PublishResult.ENQUEUED for result in worker.publish_results)
        record = server.dispatcher.registry.snapshot()[0]
        assert record.event_sink.queued_events == 0
        assert record.event_sink.queued_bytes == 0
    finally:
        connection.close()
        close_server(server, thread)


def test_http_sse_surfaces_typed_slow_consumer_error(manifest, request_body, tmp_path):
    body = json.loads(json.dumps(request_body))
    body["stream"] = {"enabled": True, "include_reasoning": False}
    server, thread = running_server(
        manifest,
        tmp_path,
        worker=OverflowEventWorker(),
        event_buffer_max_bytes=1,
    )
    connection = http.client.HTTPConnection(*server.server_address, timeout=5)
    try:
        response = post_request(connection, body)
        assert response.status == 200
        events = read_sse(response)
        assert [name for name, _ in events] == ["queued", "error"]
        assert events[-1][1]["error"]["code"] == "slow_consumer"
        assert events[-1][1]["error"]["retryable"] is True
    finally:
        connection.close()
        close_server(server, thread)


class BlockingWorker(Worker):
    def __init__(self):
        self.started = threading.Event()

    def execute(self, record):
        self.started.set()
        record.cancel_event.wait(2)
        return super().execute(record)

    def cancel(self, record):
        record.cancel_event.set()


class DisconnectAwareWorker(Worker):
    def __init__(self):
        self.started = threading.Event()
        self.cancelled = threading.Event()
        self.execute_returned = threading.Event()
        self.cancel_identity = None

    def execute(self, record):
        record.event_sink.publish(
            InferenceEvent(
                InferenceEventType.STARTED,
                record.request_id,
                record.attempt_id,
                0,
            )
        )
        self.started.set()
        record.cancel_event.wait(3)
        self.execute_returned.set()
        return super().execute(record)

    def cancel(self, record):
        self.cancel_identity = (record.request_id, record.attempt_id)
        record.cancel_event.set()
        self.cancelled.set()


def raw_post_socket(address, body):
    raw = json.dumps(body).encode("utf-8")
    connection = socket.create_connection(address, timeout=2)
    connection.sendall(
        b"POST /v1/model/generate HTTP/1.1\r\n"
        + f"Host: {address[0]}:{address[1]}\r\n".encode("ascii")
        + b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(raw)}\r\n".encode("ascii")
        + b"Connection: close\r\n\r\n"
        + raw
    )
    return connection


@pytest.mark.parametrize("stream_enabled", [False, True])
def test_client_disconnect_cancels_exact_attempt_and_persists_true_terminal(
    manifest,
    request_body,
    tmp_path,
    stream_enabled,
):
    body = json.loads(json.dumps(request_body))
    body["stream"] = {"enabled": stream_enabled, "include_reasoning": False}
    worker = DisconnectAwareWorker()
    root = tmp_path / ("stream" if stream_enabled else "non-stream")
    server, thread = running_server(manifest, root, worker=worker)
    connection = raw_post_socket(server.server_address, body)
    try:
        assert worker.started.wait(1)
        assert wait_until(lambda: bool(server.artifacts.active))
        if stream_enabled:
            headers = b""
            while b"\r\n\r\n" not in headers:
                headers += connection.recv(4096)
            assert headers.startswith(b"HTTP/1.1 200")
        connection.close()

        assert worker.cancelled.wait(1)
        assert worker.execute_returned.wait(1)
        assert wait_until(lambda: not server.artifacts.active)
        record = server.dispatcher.registry.snapshot()[0]
        assert record.lifecycle == Lifecycle.CANCELLED
        assert worker.cancel_identity == (record.request_id, record.attempt_id)

        result_files = list((root / "artifacts").rglob("result.json"))
        assert len(result_files) == 1
        stored = json.loads(result_files[0].read_text(encoding="utf-8"))
        assert stored["request_id"] == record.request_id
        assert stored["attempt_id"] == record.attempt_id
        assert stored["termination"] == record.lifecycle.value.lower()
        assert stored["error"]["code"] == "cancelled"
        assert stored["protocol_valid"] is False
        assert stored["output_valid"] is False
    finally:
        connection.close()
        close_server(server, thread)


def test_http_queue_timeout_is_terminal_while_an_earlier_request_is_active(
    manifest,
    request_body,
    tmp_path,
):
    worker = BlockingWorker()
    server, server_thread = running_server(manifest, tmp_path, worker=worker)
    first_body = json.loads(json.dumps(request_body))
    first_body["limits"]["execution_timeout_ms"] = 1000
    first_done = threading.Event()

    def post_first():
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        try:
            response = post_request(connection, first_body)
            response.read()
        finally:
            connection.close()
            first_done.set()

    first_thread = threading.Thread(target=post_first, daemon=True)
    first_thread.start()
    assert worker.started.wait(1)

    queued_body = json.loads(json.dumps(request_body))
    queued_body["limits"]["queue_timeout_ms"] = 40
    connection = http.client.HTTPConnection(*server.server_address, timeout=1)
    started = time.monotonic()
    try:
        response = post_request(connection, queued_body)
        payload = read_json(response)
        assert response.status == 408
        assert payload["termination"] == Lifecycle.TIMED_OUT.value.lower()
        assert payload["error"]["code"] == "queue_timeout"
        assert time.monotonic() - started < .25
    finally:
        connection.close()
        for record in server.dispatcher.registry.snapshot():
            server.dispatcher.cancel(record.request_id)
        assert first_done.wait(1)
        close_server(server, server_thread)


class ReadinessWorker(Worker):
    def __init__(self):
        self.supervisor_state = SupervisorState.STARTING
        self.process_generation = 0

    def shutdown(self):
        self.supervisor_state = SupervisorState.STOPPED


def test_ready_endpoint_tracks_supervisor_state_and_generation(manifest, tmp_path):
    worker = ReadinessWorker()
    server, thread = running_server(manifest, tmp_path, worker=worker)
    connection = http.client.HTTPConnection(*server.server_address, timeout=2)
    try:
        for state, expected_status in [
            (SupervisorState.STARTING, 503),
            (SupervisorState.READY, 200),
            (SupervisorState.RESTARTING, 503),
            (SupervisorState.DEGRADED, 503),
        ]:
            worker.supervisor_state = state
            if state == SupervisorState.READY:
                worker.process_generation = 3
            connection.request("GET", "/ready")
            response = connection.getresponse()
            payload = read_json(response)
            assert response.status == expected_status
            assert payload["supervisor_state"] == state.value
            assert payload["process_generation"] == worker.process_generation
    finally:
        connection.close()
        close_server(server, thread)


def test_server_shutdown_finishes_active_artifact(manifest, request_body, tmp_path):
    worker = BlockingWorker()
    server, server_thread = running_server(manifest, tmp_path, worker=worker)
    completed = threading.Event()

    def post_active():
        connection = http.client.HTTPConnection(*server.server_address, timeout=2)
        try:
            response = post_request(connection, request_body)
            response.read()
        finally:
            connection.close()
            completed.set()

    request_thread = threading.Thread(target=post_active, daemon=True)
    request_thread.start()
    assert worker.started.wait(1)
    close_server(server, server_thread)
    assert completed.wait(1)
    assert server.artifacts.active == set()


def test_http_rejects_escaped_unpaired_surrogate_without_losing_liveness(
    manifest,
    request_body,
    tmp_path,
):
    server, thread = running_server(manifest, tmp_path)
    connection = http.client.HTTPConnection(*server.server_address, timeout=2)
    try:
        raw = json.dumps(request_body).replace(
            "Return a structured answer.",
            "\\ud800",
        ).encode("ascii")
        connection.request(
            "POST",
            "/v1/model/generate",
            body=raw,
            headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
        )
        response = connection.getresponse()
        payload = read_json(response)
        assert response.status == 400
        assert payload["error"]["code"] == "invalid_request"
        connection.request("GET", "/live")
        live = connection.getresponse()
        assert live.status == 200
        live.read()
    finally:
        connection.close()
        close_server(server, thread)
