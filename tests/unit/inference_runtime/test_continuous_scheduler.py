from __future__ import annotations

import json
import threading
import time
from collections import Counter
from dataclasses import dataclass

import pytest

from inference_runtime import (
    BackendCapabilities,
    ContinuousBatchScheduler,
    DecodeOutcome,
    DecodeStatus,
    FinishReason,
    InferenceRuntimeError,
    PrefillOutcome,
    PrefillStatus,
    ReleaseOutcome,
    ReleaseStatus,
    SchedulerEvent,
    SchedulerEventKind,
    SchedulingMetadata,
    SequenceCompletion,
    SequenceHandle,
)
from model_worker.preflight import preflight


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[SchedulerEvent] = []
        self.lock = threading.Lock()

    def publish(self, event: SchedulerEvent) -> None:
        with self.lock:
            self.events.append(event)


@dataclass
class FakeState:
    request_id: str
    decode_remaining: int
    decoded: int = 0


class FakeBatchBackend:
    capabilities = BackendCapabilities(
        backend="fake-batch",
        models=("qwen35-9b-local",),
        supports_full_request=False,
        supports_sequence_steps=True,
        supports_streaming=True,
        supports_cancellation=True,
        supports_chunked_prefill=True,
        supports_decode_batching=True,
        supports_continuous_batching=True,
        supports_prefix_cache=False,
        supports_session_cache=False,
        supports_explicit_release=True,
        max_context_tokens=1024,
        max_output_tokens=128,
        max_concurrent_requests=4,
        max_concurrent_sequences=4,
        max_prefill_tokens_per_step=64,
        max_decode_tokens_per_step=1,
        max_sequences_per_step=4,
    )

    def __init__(self, *, decode_steps=1, block_first_open=False, block_decode=False):
        self.decode_steps = decode_steps
        self.block_first_open = block_first_open
        self.block_decode = block_decode
        self.open_entered = threading.Event()
        self.allow_open = threading.Event()
        self.decode_entered = threading.Event()
        self.allow_decode = threading.Event()
        if not block_first_open:
            self.allow_open.set()
        if not block_decode:
            self.allow_decode.set()
        self.states = {}
        self.max_live_states = 0
        self.next_sequence = 0
        self.prefill_batch_sizes = []
        self.decode_batch_sizes = []
        self.operations = []
        self.release_counts = Counter()
        self.shutdown_called = False

    @property
    def runtime_identity(self):
        return {"process_generation": 1}

    def open_sequence(self, request, *, scheduling, events):
        if self.block_first_open and self.next_sequence == 0:
            self.open_entered.set()
            self.allow_open.wait(2)
        handle = SequenceHandle(
            "fake-batch",
            request.request.model_id,
            f"slot-{self.next_sequence}",
            1,
        )
        self.next_sequence += 1
        self.states[handle] = FakeState(scheduling.request_id, self.decode_steps)
        self.max_live_states = max(self.max_live_states, len(self.states))
        self.operations.append(("open", scheduling.request_id))
        return handle

    def prefill(self, handle, *, token_budget, events):
        return self.prefill_batch((type("Step", (), {"handle": handle, "token_budget": token_budget})(),), events=events)[0]

    def prefill_batch(self, steps, *, events):
        self.prefill_batch_sizes.append(len(steps))
        self.operations.append(("prefill", tuple(self.states[s.handle].request_id for s in steps)))
        return tuple(
            PrefillOutcome(step.handle, PrefillStatus.READY, 10, 0)
            for step in steps
        )

    def decode(self, handle, *, token_budget, events):
        return self.decode_batch((type("Step", (), {"handle": handle, "token_budget": token_budget})(),), events=events)[0]

    def decode_batch(self, steps, *, events):
        self.decode_entered.set()
        self.allow_decode.wait(2)
        self.decode_batch_sizes.append(len(steps))
        self.operations.append(("decode", tuple(self.states[s.handle].request_id for s in steps)))
        outcomes = []
        for step in steps:
            state = self.states[step.handle]
            state.decode_remaining -= 1
            state.decoded += 1
            if state.decode_remaining:
                outcomes.append(
                    DecodeOutcome(
                        step.handle,
                        DecodeStatus.PROGRESSED,
                        (state.decoded,),
                        "",
                    )
                )
            else:
                text = '{"result":"ok"}'
                outcomes.append(
                    DecodeOutcome(
                        step.handle,
                        DecodeStatus.FINISHED,
                        (state.decoded,),
                        text,
                        FinishReason.STOP,
                        SequenceCompletion(
                            text,
                            10,
                            0,
                            1,
                            state.decoded,
                            1.0,
                            float(state.decoded),
                            first_sample_ms=1.5,
                            first_final_ms=float(state.decoded),
                            sample_itl_ms=tuple(
                                2.0 for _ in range(max(0, state.decoded - 1))
                            ),
                        ),
                    )
                )
        return tuple(outcomes)

    def release(self, handle, *, events):
        self.release_counts[handle] += 1
        self.states.pop(handle, None)
        self.operations.append(("release", handle.sequence))
        return ReleaseOutcome(handle, ReleaseStatus.RELEASED, 1024)

    def shutdown(self):
        self.shutdown_called = True


def metadata(request_id, deadline=None):
    return SchedulingMetadata(
        request_id=request_id,
        workflow_id="workflow",
        agent_id=f"agent-{request_id}",
        service_class="throughput",
        weight=1,
        deadline_monotonic=deadline,
    )


def prepared_request(manifest, request_body, **limit_overrides):
    body = json.loads(json.dumps(request_body))
    body["limits"].update(limit_overrides)
    return preflight(body, manifest)


def test_scheduler_forms_real_four_sequence_prefill_and_decode_batches(
    manifest,
    request_body,
):
    backend = FakeBatchBackend(block_first_open=True)
    scheduler = ContinuousBatchScheduler(backend, tick_token_budget=64)
    request = prepared_request(manifest, request_body)
    sinks = [RecordingSink() for _ in range(4)]
    results = {}

    def invoke(index):
        results[index] = scheduler.infer(
            request,
            scheduling=metadata(f"request-{index}"),
            events=sinks[index],
        )

    threads = [threading.Thread(target=invoke, args=(index,)) for index in range(4)]
    threads[0].start()
    assert backend.open_entered.wait(1)
    for thread in threads[1:]:
        thread.start()
    deadline = time.monotonic() + 1
    while scheduler.active_requests < 4 and time.monotonic() < deadline:
        time.sleep(.005)
    backend.allow_open.set()
    for thread in threads:
        thread.join(2)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 4
    assert all(result.output == {"result": "ok"} for result in results.values())
    assert all(result.timing["first_sample_ms"] == 1.5 for result in results.values())
    assert all(result.timing["sample_itl_ms"] == [] for result in results.values())
    assert 4 in backend.prefill_batch_sizes
    assert 4 in backend.decode_batch_sizes
    assert sum(backend.release_counts.values()) == 4
    for sink in sinks:
        terminal = [
            event
            for event in sink.events
            if event.kind in {
                SchedulerEventKind.REQUEST_COMPLETED,
                SchedulerEventKind.REQUEST_FAILED,
            }
        ]
        assert len(terminal) == 1
    assert scheduler.shutdown()
    assert backend.shutdown_called


def test_terminal_event_is_published_before_infer_returns(manifest, request_body):
    backend = FakeBatchBackend()
    scheduler = ContinuousBatchScheduler(backend, tick_token_budget=64)
    entered = threading.Event()
    allow = threading.Event()
    returned = threading.Event()

    class BlockingTerminalSink(RecordingSink):
        def publish(self, event):
            super().publish(event)
            if event.kind is SchedulerEventKind.REQUEST_COMPLETED:
                entered.set()
                allow.wait(2)

    sink = BlockingTerminalSink()

    def invoke():
        scheduler.infer(
            prepared_request(manifest, request_body),
            scheduling=metadata("terminal-order"),
            events=sink,
        )
        returned.set()

    thread = threading.Thread(target=invoke)
    thread.start()
    assert entered.wait(1)
    assert not returned.is_set()
    allow.set()
    thread.join(2)
    assert returned.is_set()
    assert scheduler.shutdown()
    assert backend.shutdown_called


def test_autostart_false_queues_work_for_deterministic_priority_start(
    manifest,
    request_body,
):
    backend = FakeBatchBackend()
    scheduler = ContinuousBatchScheduler(
        backend,
        tick_token_budget=64,
        autostart=False,
    )
    request = prepared_request(manifest, request_body)
    results = {}

    def invoke(request_id, service_class):
        results[request_id] = scheduler.infer(
            request,
            scheduling=SchedulingMetadata(
                request_id,
                request_id,
                request_id,
                service_class,
                1,
                None,
            ),
            events=RecordingSink(),
        )

    low = threading.Thread(target=invoke, args=("low", "background"))
    high = threading.Thread(target=invoke, args=("high", "interactive-critical"))
    low.start()
    high.start()
    deadline = time.monotonic() + 1
    while scheduler.active_requests < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert scheduler.start()
    assert not scheduler.start()
    low.join(2)
    high.join(2)
    assert list(request_id for operation, request_id in backend.operations if operation == "open")[:2] == [
        "high",
        "low",
    ]
    assert len(results) == 2
    assert scheduler.shutdown()


def test_enqueue_between_action_check_and_idle_wait_cannot_lose_wakeup(
    manifest,
    request_body,
):
    backend = FakeBatchBackend()
    scheduler = ContinuousBatchScheduler(
        backend,
        tick_token_budget=64,
        autostart=False,
    )
    action_checked = threading.Event()
    allow_idle_transition = threading.Event()
    original_choose = scheduler._choose_batch_action
    first_check = True

    def controlled_choose():
        nonlocal first_check
        result = original_choose()
        if first_check:
            first_check = False
            action_checked.set()
            allow_idle_transition.wait(2)
        return result

    scheduler._choose_batch_action = controlled_choose
    assert scheduler.start()
    assert action_checked.wait(1)
    captured = {}

    def invoke():
        captured["result"] = scheduler.infer(
            prepared_request(
                manifest,
                request_body,
                queue_timeout_ms=500,
            ),
            scheduling=metadata("lost-wakeup"),
            events=RecordingSink(),
        )

    thread = threading.Thread(target=invoke)
    thread.start()
    deadline = time.monotonic() + 1
    while scheduler.active_requests < 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert scheduler.active_requests == 1
    allow_idle_transition.set()
    thread.join(2)

    assert not thread.is_alive()
    assert captured["result"].output == {"result": "ok"}
    assert backend.operations[0] == ("open", "lost-wakeup")
    assert scheduler.shutdown()


def test_shutdown_before_autostart_terminalizes_registered_waiter(
    manifest,
    request_body,
):
    backend = FakeBatchBackend()
    scheduler = ContinuousBatchScheduler(
        backend,
        tick_token_budget=64,
        autostart=False,
    )
    errors = []

    def invoke():
        try:
            scheduler.infer(
                prepared_request(manifest, request_body),
                scheduling=metadata("never-started"),
                events=RecordingSink(),
            )
        except InferenceRuntimeError as exc:
            errors.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    deadline = time.monotonic() + 1
    while scheduler.active_requests < 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert scheduler.active_requests == 1
    assert scheduler.shutdown()
    thread.join(1)

    assert not thread.is_alive()
    assert len(errors) == 1 and errors[0].code == "shutdown"
    assert scheduler.admission_snapshot.pending_requests == 0


def test_idle_worker_crash_rejects_future_request_without_waiting_for_deadline(
    manifest,
    request_body,
):
    backend = FakeBatchBackend()
    scheduler = ContinuousBatchScheduler(
        backend,
        tick_token_budget=64,
        autostart=False,
    )

    def crash_before_work():
        raise RuntimeError("injected idle failure")

    scheduler._process_terminal_boundaries = crash_before_work
    assert scheduler.start()
    deadline = time.monotonic() + 1
    while scheduler._thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert not scheduler._thread.is_alive()

    started = time.monotonic()
    with pytest.raises(InferenceRuntimeError) as raised:
        scheduler.infer(
            prepared_request(
                manifest,
                request_body,
                queue_timeout_ms=5000,
            ),
            scheduling=metadata("after-idle-crash"),
            events=RecordingSink(),
        )
    assert time.monotonic() - started < 0.5
    assert raised.value.code == "scheduler_crashed"
    assert "injected idle failure" in raised.value.detail
    assert scheduler.shutdown()


def test_scheduler_enforces_per_agent_sequence_quota_before_backend_open(
    manifest,
    request_body,
):
    backend = FakeBatchBackend(block_first_open=True, block_decode=True)
    scheduler = ContinuousBatchScheduler(backend, tick_token_budget=64)
    request = prepared_request(manifest, request_body)
    results = {}

    def invoke(index):
        results[index] = scheduler.infer(
            request,
            scheduling=SchedulingMetadata(
                f"quota-{index}",
                "one-workflow",
                "one-agent",
                "throughput",
                1,
                None,
            ),
            events=RecordingSink(),
        )

    threads = [threading.Thread(target=invoke, args=(index,)) for index in range(3)]
    threads[0].start()
    assert backend.open_entered.wait(1)
    for thread in threads[1:]:
        thread.start()
    deadline = time.monotonic() + 1
    while scheduler.active_requests < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    backend.allow_open.set()
    assert backend.decode_entered.wait(1)
    snapshot = scheduler.admission_snapshot
    assert snapshot.active_sequences == 2
    assert snapshot.pending_requests == 1
    assert backend.max_live_states == 2

    backend.allow_decode.set()
    for thread in threads:
        thread.join(2)
    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 3
    assert backend.max_live_states == 2
    assert scheduler.shutdown()


def test_cancel_during_decode_releases_before_single_terminal_error(
    manifest,
    request_body,
):
    backend = FakeBatchBackend(decode_steps=5, block_decode=True)
    scheduler = ContinuousBatchScheduler(backend, tick_token_budget=8)
    request = prepared_request(manifest, request_body)
    sink = RecordingSink()
    captured = []

    def invoke():
        try:
            scheduler.infer(
                request,
                scheduling=metadata("cancel-me"),
                events=sink,
            )
        except InferenceRuntimeError as exc:
            captured.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    assert backend.decode_entered.wait(1)
    assert scheduler.cancel("cancel-me")
    backend.allow_decode.set()
    thread.join(2)

    assert not thread.is_alive()
    assert captured[0].code == "cancelled"
    assert sum(backend.release_counts.values()) == 1
    terminal = [
        event for event in sink.events if event.kind is SchedulerEventKind.REQUEST_FAILED
    ]
    assert len(terminal) == 1 and terminal[0].error_code == "cancelled"
    assert scheduler.shutdown()


def test_queued_deadline_expires_while_an_open_call_is_blocked(
    manifest,
    request_body,
):
    backend = FakeBatchBackend(decode_steps=2, block_first_open=True)
    scheduler = ContinuousBatchScheduler(backend, tick_token_budget=8)
    first = prepared_request(
        manifest,
        request_body,
        queue_timeout_ms=1000,
    )
    expiring = prepared_request(
        manifest,
        request_body,
        queue_timeout_ms=20,
    )
    errors = {}

    def invoke(name, request):
        try:
            scheduler.infer(
                request,
                scheduling=metadata(name),
                events=RecordingSink(),
            )
        except InferenceRuntimeError as exc:
            errors[name] = exc.code

    first_thread = threading.Thread(target=invoke, args=("first", first))
    second_thread = threading.Thread(target=invoke, args=("expiring", expiring))
    first_thread.start()
    assert backend.open_entered.wait(1)
    second_thread.start()
    time.sleep(.05)
    backend.allow_open.set()
    first_thread.join(2)
    second_thread.join(2)

    assert errors["expiring"] == "queue_timeout"
    assert scheduler.shutdown()


def test_scheduler_rejects_invalid_output_after_releasing_sequence(
    manifest,
    request_body,
):
    class InvalidOutputBackend(FakeBatchBackend):
        def decode_batch(self, steps, *, events):
            outcomes = super().decode_batch(steps, events=events)
            return tuple(
                DecodeOutcome(
                    outcome.handle,
                    DecodeStatus.FINISHED,
                    outcome.token_ids,
                    "{}",
                    FinishReason.STOP,
                    SequenceCompletion("{}", 10, 0, 1, 1, 1, 1),
                )
                for outcome in outcomes
            )

    backend = InvalidOutputBackend()
    scheduler = ContinuousBatchScheduler(backend, tick_token_budget=8)
    with pytest.raises(InferenceRuntimeError) as raised:
        scheduler.infer(
            prepared_request(
                manifest,
                request_body,
                queue_timeout_ms=5000,
                execution_timeout_ms=5000,
            ),
            scheduling=metadata("invalid-output"),
            events=RecordingSink(),
        )
    assert raised.value.code == "output_invalid"
    assert sum(backend.release_counts.values()) == 1
    assert scheduler.shutdown()
