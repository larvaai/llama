from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque

import pytest

from model_worker.errors import WorkerError
from model_worker.events import BoundedRequestEventBuffer, InferenceEventType
from model_worker.preflight import preflight
from model_worker.request_registry import RequestRegistry
from model_worker.worker_process import NativeWorkerProcess, SupervisorState


class FakeInput:
    def __init__(self, *, fail_write: bool = False):
        self.fail_write = fail_write
        self.writes: list[str] = []
        self.flushes = 0

    def write(self, value: str):
        if self.fail_write:
            raise BrokenPipeError("closed")
        self.writes.append(value)

    def flush(self):
        self.flushes += 1


class FakeOutput:
    def __init__(self, lines, *, eof_when_empty=False):
        self.lines = deque(lines)
        self.eof_when_empty = eof_when_empty
        self.closed = threading.Event()

    def readline(self):
        if self.lines:
            return self.lines.popleft()
        if self.eof_when_empty:
            return ""
        self.closed.wait(5)
        return ""


class FakeProcess:
    def __init__(self, lines, *, fail_write: bool = False, eof_when_empty: bool = False):
        self.stdin = FakeInput(fail_write=fail_write)
        self.stdout = FakeOutput(lines, eof_when_empty=eof_when_empty)
        self.returncode = None
        self.killed = 0
        self.terminated = 0

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed += 1
        self.returncode = -9
        getattr(self.stdout, "closed", getattr(self.stdout, "release", threading.Event())).set()

    def terminate(self):
        self.terminated += 1
        self.returncode = -15
        getattr(self.stdout, "closed", getattr(self.stdout, "release", threading.Event())).set()


def ready_frame(model_id, **updates):
    frame = {
        "protocol_version": "model-worker-ipc.v1",
        "type": "ready",
        "sequence": 0,
        "model_id": model_id,
    }
    frame.update(updates)
    return json.dumps(frame) + "\n"


def response_frame(record, frame_type="completed", *, sequence=None, **payload):
    if sequence is None:
        sequence = 0 if frame_type == "started" else 1
    if frame_type == "completed":
        payload.setdefault("timing", {})
    return json.dumps(
        {
            "protocol_version": "model-worker-ipc.v1",
            "request_id": record.request_id,
            "attempt_id": record.attempt_id,
            "sequence": sequence,
            "type": frame_type,
            **payload,
        }
    ) + "\n"


def successful_response_frames(record, final_text='{"result":"ok"}', usage=None):
    return [
        response_frame(record, "started", sequence=0),
        response_frame(record, "phase", sequence=1, phase="final"),
        response_frame(record, "final_delta", sequence=2, delta=final_text),
        response_frame(
            record,
            "completed",
            sequence=3,
            final_text=final_text,
            usage={} if usage is None else usage,
        ),
    ]


def request_record(manifest, request_body):
    prepared = preflight(request_body, manifest)
    return RequestRegistry().create(prepared, prepared.limits.queue_timeout_ms, prepared.limits.execution_timeout_ms)


def install_process(monkeypatch, process):
    calls = []

    def popen(*args, **kwargs):
        calls.append((args, kwargs))
        return process

    monkeypatch.setattr("model_worker.worker_process.subprocess.Popen", popen)
    return calls


def install_processes(monkeypatch, *processes):
    remaining = iter(processes)
    monkeypatch.setattr(
        "model_worker.worker_process.subprocess.Popen",
        lambda *args, **kwargs: next(remaining),
    )


def test_start_is_idempotent_and_rejects_invalid_ready(monkeypatch, manifest, tmp_path):
    process = FakeProcess([ready_frame(manifest.id)])
    calls = install_process(monkeypatch, process)
    worker = NativeWorkerProcess(tmp_path / "native.exe", manifest, shutdown_grace=.01)
    worker.start()
    worker.start()
    assert len(calls) == 1
    assert worker.process_generation == 1
    assert worker.model_loads_total == 1
    assert worker.supervisor_state == SupervisorState.READY

    bad = FakeProcess([ready_frame(manifest.id, protocol_version="wrong")])
    install_process(monkeypatch, bad)
    rejected = NativeWorkerProcess(tmp_path / "native.exe", manifest)
    with pytest.raises(WorkerError) as captured:
        rejected.start()
    assert captured.value.code == "worker_not_ready"
    assert bad.killed == 1
    assert rejected.supervisor_state == SupervisorState.DEGRADED

    for missing in ("model_id", "sequence"):
        raw = json.loads(ready_frame(manifest.id))
        raw.pop(missing)
        incomplete = FakeProcess([json.dumps(raw) + "\n"])
        install_process(monkeypatch, incomplete)
        rejected = NativeWorkerProcess(tmp_path / "native.exe", manifest)
        with pytest.raises(WorkerError, match="valid ready frame"):
            rejected.start()
        assert incomplete.killed == 1


def test_execute_validates_identity_json_contract_and_result(monkeypatch, manifest, request_body, tmp_path):
    record = request_record(manifest, request_body)
    process = FakeProcess(
        [
            ready_frame(manifest.id),
            *successful_response_frames(
                record,
                usage={"prompt_tokens": 3, "final_tokens": 1},
            ),
        ]
    )
    install_process(monkeypatch, process)
    worker = NativeWorkerProcess(tmp_path / "native.exe", manifest, shutdown_grace=.01)
    result = worker.execute(record)
    assert result.output == {"result": "ok"}
    assert result.protocol_valid is True and result.output_valid is True
    assert result.model["process_generation"] == 1
    command = json.loads(process.stdin.writes[0])
    assert command["type"] == "generate"
    assert command["request_id"] == record.request_id
    assert command["request"]["model_messages"][0]["role"] == "system"
    assert "model-worker-output-contract-v1" in command["request"]["model_messages"][0]["content"]


def test_execute_tracks_process_heartbeat_and_request_progress(
    monkeypatch,
    manifest,
    request_body,
    tmp_path,
):
    record = request_record(manifest, request_body)
    record.event_sink = BoundedRequestEventBuffer(
        record.request_id,
        record.attempt_id,
        max_events=8,
        max_bytes=4096,
    )
    process = FakeProcess(
        [
            ready_frame(manifest.id),
            response_frame(record, "started"),
            response_frame(record, "heartbeat", sequence=1, sampled_tokens=1),
            response_frame(record, "progress", sequence=2, phase="reasoning", tokens=1),
            response_frame(record, "phase", sequence=3, phase="final"),
            response_frame(record, "final_delta", sequence=4, delta='{"result":"ok"}'),
            response_frame(
                record,
                sequence=5,
                final_text='{"result":"ok"}',
                usage={},
            ),
        ]
    )
    install_process(monkeypatch, process)
    worker = NativeWorkerProcess(tmp_path / "native.exe", manifest)
    started = time.monotonic()
    assert worker.execute(record).output == {"result": "ok"}
    assert worker.last_process_heartbeat is not None
    assert worker.last_process_heartbeat >= started
    assert worker.last_request_progress is not None
    assert worker.last_request_progress >= started
    assert [event.event_type for event in record.event_sink.drain()] == [
        InferenceEventType.STARTED,
        InferenceEventType.HEARTBEAT,
        InferenceEventType.PROGRESS,
        InferenceEventType.PHASE,
        InferenceEventType.FINAL_DELTA,
    ]


def test_execute_forwards_one_thousand_deltas_in_source_order(
    monkeypatch,
    manifest,
    request_body,
    tmp_path,
):
    record = request_record(manifest, request_body)
    record.event_sink = BoundedRequestEventBuffer(
        record.request_id,
        record.attempt_id,
        max_events=1002,
        max_bytes=1024 * 1024,
    )
    deltas = ['{"result":"'] + ["x"] * 998 + ['"}']
    final_text = "".join(deltas)
    frames = [
        ready_frame(manifest.id),
        response_frame(record, "started", sequence=0),
        response_frame(record, "phase", sequence=1, phase="final"),
    ]
    frames.extend(
        response_frame(record, "final_delta", sequence=sequence, delta=delta)
        for sequence, delta in enumerate(deltas, start=2)
    )
    frames.append(
        response_frame(
            record,
            "completed",
            sequence=2 + len(deltas),
            final_text=final_text,
            usage={},
        )
    )
    process = FakeProcess(frames)
    install_process(monkeypatch, process)
    worker = NativeWorkerProcess(tmp_path / "native.exe", manifest, shutdown_grace=.01)

    assert worker.execute(record).output == {"result": "x" * 998}
    retained = record.event_sink.drain()
    retained_deltas = [
        event.delta
        for event in retained
        if event.event_type is InferenceEventType.FINAL_DELTA
    ]
    assert retained_deltas == deltas
    assert record.event_sink.queued_events == 0
    assert record.event_sink.queued_bytes == 0
    worker.shutdown()


def test_completed_final_text_must_match_joined_deltas_before_next_generation(
    monkeypatch,
    manifest,
    request_body,
    tmp_path,
):
    first = request_record(manifest, request_body)
    second = request_record(manifest, request_body)
    corrupted = FakeProcess(
        [
            ready_frame(manifest.id),
            response_frame(first, "started", sequence=0),
            response_frame(first, "phase", sequence=1, phase="final"),
            response_frame(first, "final_delta", sequence=2, delta='{"result":"wrong"}'),
            response_frame(
                first,
                "completed",
                sequence=3,
                final_text='{"result":"ok"}',
                usage={},
            ),
        ]
    )
    replacement = FakeProcess(
        [ready_frame(manifest.id), *successful_response_frames(second)]
    )
    install_processes(monkeypatch, corrupted, replacement)
    worker = NativeWorkerProcess(
        tmp_path / "native.exe",
        manifest,
        shutdown_grace=.01,
    )

    with pytest.raises(WorkerError, match="final deltas") as captured:
        worker.execute(first)
    assert captured.value.code == "worker_crashed"
    assert corrupted.killed == 1
    assert worker.process_generation == 2
    assert worker.execute(second).output == {"result": "ok"}
    worker.shutdown()


@pytest.mark.parametrize(
    ("final_text", "expected_detail"),
    [
        ("not-json", "final output is not strict JSON"),
        ('{"result":1}', "final output violates normalized contract"),
        ('{"result":"ok","extra":true}', "final output violates normalized contract"),
    ],
)
def test_execute_rejects_invalid_final_output(monkeypatch, manifest, request_body, tmp_path, final_text, expected_detail):
    record = request_record(manifest, request_body)
    process = FakeProcess(
        [
            ready_frame(manifest.id),
            *successful_response_frames(record, final_text=final_text),
        ]
    )
    install_process(monkeypatch, process)
    with pytest.raises(WorkerError) as captured:
        NativeWorkerProcess(tmp_path / "native.exe", manifest).execute(record)
    assert captured.value.code == "output_invalid"
    assert captured.value.message == expected_detail


@pytest.mark.parametrize(
    ("native_code", "expected_code"),
    [("cancelled", "cancelled"), ("context_overflow", "context_overflow"), ("unknown", "decode_failed")],
)
def test_execute_maps_native_failures(monkeypatch, manifest, request_body, tmp_path, native_code, expected_code):
    record = request_record(manifest, request_body)
    process = FakeProcess(
        [
            ready_frame(manifest.id),
            response_frame(record, "started"),
            response_frame(record, "failed", error=native_code, detail="native detail"),
        ]
    )
    install_process(monkeypatch, process)
    with pytest.raises(WorkerError) as captured:
        NativeWorkerProcess(tmp_path / "native.exe", manifest).execute(record)
    assert captured.value.code == expected_code
    assert captured.value.message == "native detail"


def test_execute_rejects_pipe_close_and_identity_desync(monkeypatch, manifest, request_body, tmp_path):
    record = request_record(manifest, request_body)
    closed = FakeProcess([ready_frame(manifest.id)], eof_when_empty=True)
    install_process(monkeypatch, closed)
    with pytest.raises(WorkerError) as captured:
        NativeWorkerProcess(tmp_path / "native.exe", manifest).execute(record)
    assert captured.value.code == "worker_crashed"

    wrong = json.loads(response_frame(record, final_text='{"result":"ok"}', usage={}))
    wrong["attempt_id"] = "wrong"
    desync = FakeProcess(
        [ready_frame(manifest.id), response_frame(record, "started"), json.dumps(wrong) + "\n"]
    )
    install_process(monkeypatch, desync)
    with pytest.raises(WorkerError) as captured:
        NativeWorkerProcess(tmp_path / "native.exe", manifest).execute(record)
    assert captured.value.code == "worker_crashed"


def test_cancel_restart_and_shutdown_process_lifecycle(monkeypatch, manifest, request_body, tmp_path):
    record = request_record(manifest, request_body)
    current = FakeProcess([], fail_write=True)
    replacement = FakeProcess([ready_frame(manifest.id)])
    processes = iter([replacement])
    monkeypatch.setattr("model_worker.worker_process.subprocess.Popen", lambda *args, **kwargs: next(processes))
    worker = NativeWorkerProcess(tmp_path / "native.exe", manifest, shutdown_grace=.01)
    worker.process = current

    worker.cancel(record)
    assert record.cancel_event.is_set()
    assert worker.kill_and_restart() is True
    assert current.killed == 1
    assert worker.process is replacement
    assert worker.supervisor_state == SupervisorState.READY

    record.cancel_event.clear()
    worker.cancel(record)
    cancel = json.loads(replacement.stdin.writes[-1])
    assert cancel["type"] == "cancel"
    assert cancel["request_id"] == record.request_id

    worker.shutdown()
    assert replacement.terminated == 1
    assert worker.process is None
    assert worker.supervisor_state == SupervisorState.STOPPED
    worker.shutdown()


def test_restart_reports_failed_readiness(monkeypatch, manifest, tmp_path):
    current = FakeProcess([])
    invalid = FakeProcess([ready_frame(manifest.id, type="not-ready")])
    install_process(monkeypatch, invalid)
    worker = NativeWorkerProcess(tmp_path / "native.exe", manifest, shutdown_grace=.01)
    worker.process = current
    assert worker.kill_and_restart() is False
    assert current.killed == 1
    assert invalid.killed == 1
    assert worker.supervisor_state == SupervisorState.DEGRADED


class BlockingOutput:
    def __init__(self):
        self.release = threading.Event()

    def readline(self):
        self.release.wait(5)
        return ""


def test_startup_timeout_is_bounded_and_leaves_supervisor_degraded(
    monkeypatch,
    manifest,
    tmp_path,
):
    process = FakeProcess([], eof_when_empty=True)
    process.stdout = BlockingOutput()
    install_process(monkeypatch, process)
    worker = NativeWorkerProcess(
        tmp_path / "native.exe",
        manifest,
        startup_timeout=.03,
        terminate_grace=.01,
    )
    started = time.monotonic()
    with pytest.raises(WorkerError, match="startup timed out"):
        worker.start()
    assert time.monotonic() - started < .20
    assert process.terminated == 1
    assert worker.supervisor_state == SupervisorState.DEGRADED
    assert worker.process is None
    process.stdout.release.set()


def test_crash_before_ready_is_rejected_without_false_ready(monkeypatch, manifest, tmp_path):
    process = FakeProcess([], eof_when_empty=True)
    process.returncode = 7
    install_process(monkeypatch, process)
    worker = NativeWorkerProcess(tmp_path / "native.exe", manifest, startup_timeout=.1)
    with pytest.raises(WorkerError, match="exited before ready"):
        worker.start()
    assert worker.supervisor_state == SupervisorState.DEGRADED
    assert worker.process_generation == 0


def test_live_process_exit_changes_ready_state_to_degraded(monkeypatch, manifest, tmp_path):
    process = FakeProcess([ready_frame(manifest.id)])
    install_process(monkeypatch, process)
    worker = NativeWorkerProcess(tmp_path / "native.exe", manifest)
    worker.start()
    assert worker.supervisor_state == SupervisorState.READY
    process.returncode = 9
    assert worker.supervisor_state == SupervisorState.DEGRADED


class ControlledOutput:
    def __init__(self, *lines):
        self.lines: queue.Queue[str] = queue.Queue()
        self.closed = threading.Event()
        for line in lines:
            self.lines.put(line)

    def readline(self):
        while not self.closed.is_set():
            try:
                return self.lines.get(timeout=.05)
            except queue.Empty:
                continue
        return ""

    def push(self, line):
        self.lines.put(line)


def test_cancel_before_native_send_never_starts_process(monkeypatch, manifest, request_body, tmp_path):
    record = request_record(manifest, request_body)
    record.cancel_event.set()
    calls = []
    monkeypatch.setattr(
        "model_worker.worker_process.subprocess.Popen",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    with pytest.raises(WorkerError) as captured:
        NativeWorkerProcess(tmp_path / "native.exe", manifest).execute(record)
    assert captured.value.code == "cancelled"
    assert calls == []


def test_cancel_after_send_before_started_is_forwarded_with_attempt_identity(
    monkeypatch,
    manifest,
    request_body,
    tmp_path,
):
    record = request_record(manifest, request_body)
    output = ControlledOutput(ready_frame(manifest.id))
    process = FakeProcess([])
    process.stdout = output
    install_process(monkeypatch, process)
    worker = NativeWorkerProcess(tmp_path / "native.exe", manifest, shutdown_grace=.01)
    outcome = {}

    def execute():
        try:
            worker.execute(record)
        except WorkerError as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=execute)
    thread.start()
    deadline = time.monotonic() + 1
    while not process.stdin.writes and time.monotonic() < deadline:
        time.sleep(.001)
    assert json.loads(process.stdin.writes[0])["type"] == "generate"
    worker.cancel(record)
    output.push(response_frame(record, "started"))
    failed = json.loads(response_frame(record, "failed", error="cancelled", detail="cancelled"))
    failed["sequence"] = 1
    output.push(json.dumps(failed) + "\n")
    thread.join(1)
    assert not thread.is_alive()
    assert outcome["error"].code == "cancelled"
    cancel = json.loads(process.stdin.writes[-1])
    assert (cancel["request_id"], cancel["attempt_id"]) == (
        record.request_id,
        record.attempt_id,
    )
    worker.shutdown()


def test_slow_consumer_cancels_native_and_drains_terminal_before_typed_error(
    monkeypatch,
    manifest,
    request_body,
    tmp_path,
):
    streaming_body = {**request_body, "stream": {"enabled": True, "include_reasoning": False}}
    record = request_record(manifest, streaming_body)
    record.event_sink = BoundedRequestEventBuffer(
        record.request_id,
        record.attempt_id,
        max_events=1,
        max_bytes=4096,
    )
    output = ControlledOutput(ready_frame(manifest.id))
    process = FakeProcess([])
    process.stdout = output
    install_process(monkeypatch, process)
    worker = NativeWorkerProcess(
        tmp_path / "native.exe",
        manifest,
        terminal_frame_grace=.01,
        shutdown_grace=.01,
    )
    outcome = {}

    def execute():
        try:
            worker.execute(record)
        except WorkerError as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=execute)
    thread.start()
    deadline = time.monotonic() + 1
    while not process.stdin.writes and time.monotonic() < deadline:
        time.sleep(.001)
    assert process.stdin.writes
    output.push(response_frame(record, "started", sequence=0))
    output.push(response_frame(record, "phase", sequence=1, phase="final"))

    deadline = time.monotonic() + 1
    sent_types = []
    while time.monotonic() < deadline:
        sent_types = [json.loads(frame)["type"] for frame in process.stdin.writes]
        if "cancel" in sent_types:
            break
        time.sleep(.001)
    assert "cancel" in sent_types
    assert record.cancel_event.is_set()
    assert record.event_sink.slow_consumer
    assert record.event_sink.queued_events <= record.event_sink.max_events
    assert record.event_sink.queued_bytes <= record.event_sink.max_bytes
    assert thread.is_alive(), "execute must drain the native terminal frame before returning"

    output.push(
        response_frame(
            record,
            "failed",
            sequence=2,
            error="cancelled",
            detail="cancelled",
        )
    )
    thread.join(1)
    assert not thread.is_alive()
    assert outcome["error"].code == "slow_consumer"
    assert outcome["error"].http_status == 429

    second = request_record(manifest, request_body)
    for frame in successful_response_frames(second):
        output.push(frame)
    assert worker.execute(second).output == {"result": "ok"}
    assert worker.supervisor_state == SupervisorState.READY
    cancel = next(
        json.loads(frame)
        for frame in process.stdin.writes
        if json.loads(frame)["type"] == "cancel"
    )
    assert (cancel["request_id"], cancel["attempt_id"]) == (
        record.request_id,
        record.attempt_id,
    )
    worker.shutdown()


@pytest.mark.parametrize("corruption", ["malformed", "wrong_id", "sequence_gap", "duplicate_terminal", "eof"])
def test_protocol_corruption_restarts_before_next_request(
    monkeypatch,
    manifest,
    request_body,
    tmp_path,
    corruption,
):
    first = request_record(manifest, request_body)
    second = request_record(manifest, request_body)
    if corruption == "malformed":
        lines = [ready_frame(manifest.id), "{not-json}\n"]
    elif corruption == "wrong_id":
        frame = json.loads(response_frame(first, final_text='{"result":"ok"}', usage={}))
        frame["request_id"] = "wrong"
        lines = [
            ready_frame(manifest.id),
            response_frame(first, "started"),
            json.dumps(frame) + "\n",
        ]
    elif corruption == "sequence_gap":
        frame = json.loads(response_frame(first, final_text='{"result":"ok"}', usage={}))
        frame["sequence"] = 2
        lines = [
            ready_frame(manifest.id),
            response_frame(first, "started"),
            json.dumps(frame) + "\n",
        ]
    elif corruption == "duplicate_terminal":
        duplicate = json.loads(
            response_frame(
                first,
                "completed",
                sequence=4,
                final_text='{"result":"ok"}',
                usage={},
            )
        )
        lines = [
            ready_frame(manifest.id),
            *successful_response_frames(first),
            json.dumps(duplicate) + "\n",
        ]
    else:
        lines = [ready_frame(manifest.id), response_frame(first, "started")]

    corrupted = FakeProcess(lines, eof_when_empty=corruption == "eof")
    replacement = FakeProcess(
        [
            ready_frame(manifest.id),
            *successful_response_frames(second),
        ]
    )
    install_processes(monkeypatch, corrupted, replacement)
    worker = NativeWorkerProcess(
        tmp_path / "native.exe",
        manifest,
        terminal_frame_grace=.02,
        shutdown_grace=.01,
    )
    with pytest.raises(WorkerError) as captured:
        worker.execute(first)
    assert captured.value.code == "worker_crashed"
    assert worker.supervisor_state == SupervisorState.READY
    assert worker.process_generation == 2
    assert worker.execute(second).output == {"result": "ok"}
    assert corrupted.killed == 1
    worker.shutdown()


def test_shutdown_sends_control_frame_and_allows_responsive_native_exit(
    monkeypatch,
    manifest,
    tmp_path,
):
    process = FakeProcess([ready_frame(manifest.id)])
    install_process(monkeypatch, process)
    worker = NativeWorkerProcess(
        tmp_path / "native.exe",
        manifest,
        shutdown_grace=.2,
        terminate_grace=.02,
    )
    worker.start()

    def acknowledge_shutdown():
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if process.stdin.writes and json.loads(process.stdin.writes[-1])["type"] == "shutdown":
                process.returncode = 0
                process.stdout.closed.set()
                return
            time.sleep(.001)

    acknowledgement = threading.Thread(target=acknowledge_shutdown)
    acknowledgement.start()
    worker.shutdown()
    acknowledgement.join(1)
    assert json.loads(process.stdin.writes[-1])["type"] == "shutdown"
    assert process.terminated == 0 and process.killed == 0
    assert worker.supervisor_state == SupervisorState.STOPPED


def test_shutdown_terminates_then_kills_unresponsive_native(monkeypatch, manifest, tmp_path):
    class StubbornProcess(FakeProcess):
        def terminate(self):
            self.terminated += 1

    process = StubbornProcess([ready_frame(manifest.id)])
    install_process(monkeypatch, process)
    worker = NativeWorkerProcess(
        tmp_path / "native.exe",
        manifest,
        shutdown_grace=.01,
        terminate_grace=.01,
    )
    worker.start()
    started = time.monotonic()
    worker.shutdown()
    assert time.monotonic() - started < .20
    assert process.terminated == 1 and process.killed == 1
    assert worker.supervisor_state == SupervisorState.STOPPED
