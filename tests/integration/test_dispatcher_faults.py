from __future__ import annotations

import time
import threading
from dataclasses import replace

import pytest

from model_worker.dispatcher import Dispatcher
from model_worker.errors import WorkerError
from model_worker.events import (
    BoundedRequestEventBuffer,
    InferenceEvent,
    InferenceEventType,
    NullEventSink,
    PublishResult,
)
from model_worker.preflight import preflight
from model_worker.request_registry import Lifecycle


class FakeWorker:
    def __init__(self): self.mode="ok"; self.received=[]; self.kills=0; self.cancelled=0; self.shutdowns=0; self.started=threading.Event()
    def execute(self, record):
        self.received.append(record.request_id)
        self.started.set()
        if self.mode in {"hang", "unresponsive"}:
            while not record.cancel_event.wait(.01): pass
            if self.mode == "unresponsive": time.sleep(.2)
        if self.mode == "crash": raise RuntimeError("boom")
        if self.mode == "typed_error": raise WorkerError("output_invalid", "bad output")
        return {"termination":"completed","protocol_valid":True,"output_valid":True,"output":{"result":"x"}}
    def cancel(self, record): self.cancelled += 1; record.cancel_event.set()
    def kill_and_restart(self): self.kills += 1; return True
    def shutdown(self): self.shutdowns += 1


class EventPublishingWorker(FakeWorker):
    def execute(self, record):
        self.received.append(record.request_id)
        events = (
            InferenceEvent(
                InferenceEventType.STARTED,
                record.request_id,
                record.attempt_id,
                0,
            ),
            InferenceEvent(
                InferenceEventType.PROGRESS,
                record.request_id,
                record.attempt_id,
                1,
                phase="reasoning",
                tokens=1,
            ),
            InferenceEvent(
                InferenceEventType.PHASE,
                record.request_id,
                record.attempt_id,
                2,
                phase="final",
            ),
            InferenceEvent(
                InferenceEventType.FINAL_DELTA,
                record.request_id,
                record.attempt_id,
                3,
                delta='{"result":"x"}',
            ),
        )
        self.publish_results = [record.event_sink.publish(event) for event in events]
        return {
            "termination": "completed",
            "protocol_valid": True,
            "output_valid": True,
            "output": {"result": "x"},
        }


class SlowConsumerErrorWorker(FakeWorker):
    def execute(self, record):
        record.cancel_event.set()
        raise WorkerError("slow_consumer", "client did not drain stream")


def wait_until(predicate, timeout=.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.005)
    return predicate()


def test_queued_cancel_never_reaches_model(manifest, request_body):
    worker = FakeWorker(); dispatcher = Dispatcher(worker, capacity=2)
    first = preflight(request_body, manifest)
    worker.mode = "hang"
    r1 = dispatcher.submit(first)
    assert worker.started.wait(1)
    r2 = dispatcher.submit(first)
    assert dispatcher.cancel(r2.request_id)
    assert dispatcher.wait(r2, 1).lifecycle == Lifecycle.CANCELLED
    dispatcher.cancel(r1.request_id); dispatcher.wait(r1, 1)
    assert worker.cancelled == 1
    assert r2.request_id not in worker.received
    dispatcher.shutdown()


def test_watchdog_and_crash_do_not_block_next_request(manifest, request_body):
    worker = FakeWorker(); dispatcher = Dispatcher(worker, capacity=3, watchdog_grace_ms=10)
    prepared = preflight(request_body, manifest)
    worker.mode="unresponsive"; timed = dispatcher.submit(prepared)
    assert dispatcher.wait(timed, 1).lifecycle == Lifecycle.TIMED_OUT
    terminal_elapsed = (
        timed.timestamps[Lifecycle.TIMED_OUT.value]
        - timed.timestamps[Lifecycle.RUNNING.value]
    )
    assert terminal_elapsed <= timed.execution_timeout + dispatcher.watchdog_grace + .05
    assert wait_until(lambda: worker.kills == 1)
    worker.mode="crash"; crashed=dispatcher.submit(prepared)
    assert dispatcher.wait(crashed, 1).lifecycle == Lifecycle.FAILED
    worker.mode="ok"; good=dispatcher.submit(prepared)
    assert dispatcher.wait(good, 1).lifecycle == Lifecycle.COMPLETED
    dispatcher.shutdown()


def test_dispatcher_allocates_bounded_events_only_for_stream_requests(
    manifest,
    request_body,
):
    worker = EventPublishingWorker()
    dispatcher = Dispatcher(
        worker,
        capacity=2,
        event_buffer_max_events=4,
        event_buffer_max_bytes=4096,
    )
    streaming_body = {**request_body, "stream": {"enabled": True, "include_reasoning": False}}
    streamed = dispatcher.submit(preflight(streaming_body, manifest))
    assert dispatcher.wait(streamed, 1).lifecycle == Lifecycle.COMPLETED
    assert isinstance(streamed.event_sink, BoundedRequestEventBuffer)
    assert streamed.event_sink.max_events == 4
    assert streamed.event_sink.max_bytes == 4096
    assert worker.publish_results == [PublishResult.ENQUEUED] * 4
    assert [event.event_type for event in streamed.event_sink.drain()] == [
        InferenceEventType.STARTED,
        InferenceEventType.PROGRESS,
        InferenceEventType.PHASE,
        InferenceEventType.FINAL_DELTA,
    ]
    assert streamed.event_sink.get(timeout=0) is None

    non_streamed = dispatcher.submit(preflight(request_body, manifest))
    assert dispatcher.wait(non_streamed, 1).lifecycle == Lifecycle.COMPLETED
    assert isinstance(non_streamed.event_sink, NullEventSink)
    dispatcher.shutdown()


def test_dispatcher_preserves_typed_slow_consumer_failure(manifest, request_body):
    dispatcher = Dispatcher(SlowConsumerErrorWorker(), capacity=1)
    streaming_body = {**request_body, "stream": {"enabled": True, "include_reasoning": False}}
    record = dispatcher.submit(preflight(streaming_body, manifest))

    assert dispatcher.wait(record, 1).lifecycle == Lifecycle.FAILED
    assert isinstance(record.error, WorkerError)
    assert record.error.code == "slow_consumer"
    assert isinstance(record.event_sink, BoundedRequestEventBuffer)
    assert record.event_sink.get(timeout=0) is None
    dispatcher.shutdown()


def with_timeouts(prepared, *, queue_ms=None, execution_ms=None):
    limits = replace(
        prepared.limits,
        queue_timeout_ms=queue_ms or prepared.limits.queue_timeout_ms,
        execution_timeout_ms=execution_ms or prepared.limits.execution_timeout_ms,
    )
    return replace(prepared, request=replace(prepared.request, limits=limits))


def test_queue_full_timeout_typed_error_and_shutdown_rejection(manifest, request_body):
    worker = FakeWorker()
    worker.mode = "hang"
    dispatcher = Dispatcher(worker, capacity=1)
    prepared = preflight(request_body, manifest)
    running = dispatcher.submit(with_timeouts(prepared, execution_ms=1000))
    assert worker.started.wait(1)
    queued = dispatcher.submit(with_timeouts(prepared, queue_ms=30))
    with pytest.raises(WorkerError) as captured:
        dispatcher.submit(prepared)
    assert captured.value.code == "queue_full"

    assert dispatcher.wait(queued, .25).lifecycle == Lifecycle.TIMED_OUT
    dispatcher.cancel(running.request_id)
    assert dispatcher.wait(running, 1).lifecycle == Lifecycle.CANCELLED
    assert queued.error == "queue_timeout"
    dispatcher.shutdown()
    with pytest.raises(WorkerError) as captured:
        dispatcher.submit(prepared)
    assert captured.value.code == "shutdown"


def test_wait_timeout_and_typed_worker_error_are_preserved(manifest, request_body):
    worker = FakeWorker()
    dispatcher = Dispatcher(worker, capacity=2)
    prepared = preflight(request_body, manifest)
    worker.mode = "hang"
    running = dispatcher.submit(with_timeouts(prepared, execution_ms=500))
    assert worker.started.wait(1)
    assert dispatcher.wait(running, .001).lifecycle == Lifecycle.RUNNING
    dispatcher.cancel(running.request_id)
    dispatcher.wait(running, 1)

    worker.started.clear()
    worker.mode = "typed_error"
    failed = dispatcher.submit(prepared)
    assert dispatcher.wait(failed, 1).lifecycle == Lifecycle.FAILED
    assert isinstance(failed.error, WorkerError)
    assert failed.error.code == "output_invalid"
    dispatcher.shutdown()


def test_shutdown_drains_a_full_queue_and_cancels_active_records(manifest, request_body):
    worker = FakeWorker()
    worker.mode = "hang"
    dispatcher = Dispatcher(worker, capacity=1)
    prepared = preflight(request_body, manifest)
    running = dispatcher.submit(prepared)
    assert worker.started.wait(1)
    queued = dispatcher.submit(prepared)
    dispatcher.shutdown(hard_timeout=1)
    assert running.lifecycle == Lifecycle.CANCELLED
    assert queued.lifecycle == Lifecycle.CANCELLED
    assert worker.shutdowns == 1


def test_queued_deadline_expires_while_head_is_still_running_and_reclaims_capacity(
    manifest,
    request_body,
):
    worker = FakeWorker()
    worker.mode = "hang"
    dispatcher = Dispatcher(worker, capacity=1)
    prepared = preflight(request_body, manifest)
    running = dispatcher.submit(with_timeouts(prepared, execution_ms=1000))
    assert worker.started.wait(1)

    started = time.monotonic()
    queued = dispatcher.submit(with_timeouts(prepared, queue_ms=40))
    assert dispatcher.wait(queued, .25).lifecycle == Lifecycle.TIMED_OUT
    assert time.monotonic() - started < .20
    assert running.lifecycle == Lifecycle.RUNNING
    assert queued.error == "queue_timeout"

    # Expiry physically removes the queued item, so it cannot accumulate as a
    # tombstone while the active request remains blocked.
    replacement = dispatcher.submit(with_timeouts(prepared, queue_ms=500))
    assert replacement.lifecycle == Lifecycle.QUEUED
    assert dispatcher.cancel(replacement.request_id)
    assert dispatcher.cancel(running.request_id)
    assert dispatcher.wait(running, 1).lifecycle == Lifecycle.CANCELLED
    dispatcher.shutdown()


def test_cancel_timeout_race_has_one_terminal_transition_and_one_reclaimed_slot(
    manifest,
    request_body,
):
    worker = FakeWorker()
    worker.mode = "hang"
    dispatcher = Dispatcher(worker, capacity=1)
    prepared = preflight(request_body, manifest)
    running = dispatcher.submit(with_timeouts(prepared, execution_ms=1000))
    assert worker.started.wait(1)
    queued = dispatcher.submit(with_timeouts(prepared, queue_ms=30))

    racers = [threading.Thread(target=dispatcher.cancel, args=(queued.request_id,)) for _ in range(8)]
    time.sleep(.025)
    for racer in racers:
        racer.start()
    for racer in racers:
        racer.join(1)

    assert dispatcher.wait(queued, .25).lifecycle in {Lifecycle.CANCELLED, Lifecycle.TIMED_OUT}
    terminal_timestamps = set(queued.timestamps) & {
        Lifecycle.CANCELLED.value,
        Lifecycle.TIMED_OUT.value,
    }
    assert len(terminal_timestamps) == 1
    replacement = dispatcher.submit(with_timeouts(prepared, queue_ms=500))
    assert replacement.lifecycle == Lifecycle.QUEUED
    dispatcher.cancel(replacement.request_id)
    dispatcher.cancel(running.request_id)
    dispatcher.wait(running, 1)
    dispatcher.shutdown()


def test_shutdown_is_idempotent_and_hard_bounded_with_unresponsive_active_work(
    manifest,
    request_body,
):
    worker = FakeWorker()
    worker.mode = "unresponsive"
    dispatcher = Dispatcher(worker, capacity=1)
    prepared = preflight(request_body, manifest)
    running = dispatcher.submit(with_timeouts(prepared, execution_ms=1000))
    assert worker.started.wait(1)
    queued = dispatcher.submit(with_timeouts(prepared, queue_ms=500))

    started = time.monotonic()
    dispatcher.shutdown(hard_timeout=.03)
    assert time.monotonic() - started < .15
    assert running.lifecycle == Lifecycle.CANCELLED
    assert queued.lifecycle == Lifecycle.CANCELLED
    assert all(record.lifecycle in {Lifecycle.CANCELLED, Lifecycle.TIMED_OUT} for record in dispatcher.registry.snapshot())
    assert worker.shutdowns == 1

    dispatcher.shutdown(hard_timeout=.03)
    assert worker.shutdowns == 1
    with pytest.raises(WorkerError, match="draining"):
        dispatcher.submit(prepared)


def test_shutdown_idle_is_idempotent(manifest, request_body):
    worker = FakeWorker()
    dispatcher = Dispatcher(worker, capacity=1)
    dispatcher.shutdown(hard_timeout=.2)
    dispatcher.shutdown(hard_timeout=.2)
    assert worker.shutdowns == 1


def test_cancel_and_expiry_storm_keeps_physical_queue_bounded(manifest, request_body):
    worker = FakeWorker()
    worker.mode = "hang"
    capacity = 2
    dispatcher = Dispatcher(worker, capacity=capacity)
    prepared = preflight(request_body, manifest)
    running = dispatcher.submit(with_timeouts(prepared, execution_ms=1000))
    assert worker.started.wait(1)

    for _ in range(100):
        queued = dispatcher.submit(with_timeouts(prepared, queue_ms=500))
        assert dispatcher.queued_count <= capacity
        assert dispatcher.cancel(queued.request_id)
        assert dispatcher.wait(queued, .1).lifecycle == Lifecycle.CANCELLED
        assert dispatcher.queued_count == 0

    for _ in range(20):
        queued = dispatcher.submit(with_timeouts(prepared, queue_ms=2))
        assert dispatcher.queued_count <= capacity
        assert dispatcher.wait(queued, .2).lifecycle == Lifecycle.TIMED_OUT
        assert dispatcher.queued_count == 0

    assert dispatcher.cancel(running.request_id)
    assert dispatcher.wait(running, 1).lifecycle == Lifecycle.CANCELLED
    dispatcher.shutdown()


class FirstRequestGateWorker(FakeWorker):
    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def execute(self, record):
        self.received.append(record.request_id)
        if len(self.received) == 1:
            self.started.set()
            self.release.wait()
        return {
            "termination": "completed",
            "protocol_valid": True,
            "output_valid": True,
            "output": {"result": "x"},
        }


def test_removing_middle_item_preserves_fifo_for_live_requests(manifest, request_body):
    worker = FirstRequestGateWorker()
    dispatcher = Dispatcher(worker, capacity=3)
    prepared = preflight(request_body, manifest)
    active = dispatcher.submit(prepared)
    assert worker.started.wait(1)
    first = dispatcher.submit(prepared)
    removed = dispatcher.submit(prepared)
    last = dispatcher.submit(prepared)

    try:
        assert dispatcher.cancel(removed.request_id)
        assert dispatcher.queued_count == 2
        worker.release.set()
        assert dispatcher.wait(active, 1).lifecycle == Lifecycle.COMPLETED
        assert dispatcher.wait(first, 1).lifecycle == Lifecycle.COMPLETED
        assert dispatcher.wait(last, 1).lifecycle == Lifecycle.COMPLETED
        assert worker.received == [active.request_id, first.request_id, last.request_id]
    finally:
        worker.release.set()
        dispatcher.shutdown()


class SlowRestartWorker(FakeWorker):
    def __init__(self):
        super().__init__()
        self.execute_release = threading.Event()
        self.restart_started = threading.Event()
        self.restart_release = threading.Event()
        self.restart_finished = threading.Event()

    def execute(self, record):
        self.received.append(record.request_id)
        self.started.set()
        record.cancel_event.wait()
        self.execute_release.wait()
        return {"output": {"result": "late"}}

    def kill_and_restart(self):
        self.kills += 1
        self.restart_started.set()
        self.restart_release.wait()
        self.restart_finished.set()
        return True


def test_execution_timeout_terminalizes_before_slow_restart_finishes(
    manifest,
    request_body,
):
    worker = SlowRestartWorker()
    dispatcher = Dispatcher(worker, capacity=1, watchdog_grace_ms=10)
    prepared = preflight(request_body, manifest)
    record = dispatcher.submit(with_timeouts(prepared, execution_ms=30))

    try:
        assert worker.started.wait(1)
        assert dispatcher.wait(record, .2).lifecycle == Lifecycle.TIMED_OUT
        terminal_elapsed = (
            record.timestamps[Lifecycle.TIMED_OUT.value]
            - record.timestamps[Lifecycle.RUNNING.value]
        )
        assert terminal_elapsed <= record.execution_timeout + dispatcher.watchdog_grace + .05
        assert worker.restart_started.wait(.5)
        assert not worker.restart_finished.is_set()
    finally:
        worker.execute_release.set()
        worker.restart_release.set()
        dispatcher.shutdown(hard_timeout=1)

    assert dispatcher.shutdown_complete


class SlowShutdownWorker(FakeWorker):
    def __init__(self):
        super().__init__()
        self.shutdown_started = threading.Event()
        self.shutdown_release = threading.Event()
        self.shutdown_finished = threading.Event()

    def shutdown(self):
        self.shutdowns += 1
        self.shutdown_started.set()
        self.shutdown_release.wait()
        self.shutdown_finished.set()


def test_repeated_shutdown_reaps_backend_before_marking_complete(manifest, request_body):
    worker = SlowShutdownWorker()
    dispatcher = Dispatcher(worker, capacity=1)

    started = time.monotonic()
    dispatcher.shutdown(hard_timeout=.02)
    assert time.monotonic() - started < .15
    assert worker.shutdown_started.wait(.2)
    assert not dispatcher.shutdown_complete
    assert not worker.shutdown_finished.is_set()

    worker.shutdown_release.set()
    dispatcher.shutdown(hard_timeout=.5)
    assert dispatcher.shutdown_complete
    assert worker.shutdown_finished.is_set()
    assert worker.shutdowns == 1

    dispatcher.shutdown(hard_timeout=.1)
    assert worker.shutdowns == 1
