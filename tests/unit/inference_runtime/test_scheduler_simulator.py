from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest
from hypothesis import given, settings, strategies as st

from inference_runtime import (
    AdmissionError,
    BackendCapabilities,
    DecodeOutcome,
    DecodeStatus,
    DeterministicSchedulerSimulator,
    FakeMonotonicClock,
    FinishReason,
    PrefillOutcome,
    PrefillStatus,
    ReleaseOutcome,
    ReleaseStatus,
    SchedulerEvent,
    SchedulerEventKind,
    SchedulerPolicy,
    SchedulingMetadata,
    SequenceCompletion,
    SequenceHandle,
    SimulatorLifecycle,
    SimulatorTermination,
    TickAction,
)


@dataclass(frozen=True, slots=True)
class FakeWork:
    prompt_tokens: int
    decode_tokens: int


@dataclass(slots=True)
class FakeSequenceState:
    request_id: str
    remaining_prompt: int
    remaining_decode: int
    next_token: int = 1


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[SchedulerEvent] = []

    def publish(self, event: SchedulerEvent) -> None:
        self.events.append(event)


def backend_capabilities(
    *,
    max_requests: int = 128,
    max_sequences: int = 128,
) -> BackendCapabilities:
    return BackendCapabilities(
        backend="fake-step-backend",
        models=("fake-model",),
        supports_full_request=False,
        supports_sequence_steps=True,
        supports_streaming=True,
        supports_cancellation=True,
        supports_chunked_prefill=True,
        supports_decode_batching=False,
        supports_continuous_batching=False,
        supports_prefix_cache=False,
        supports_session_cache=False,
        supports_explicit_release=True,
        max_context_tokens=64,
        max_output_tokens=32,
        max_concurrent_requests=max_requests,
        max_concurrent_sequences=max_sequences,
        max_prefill_tokens_per_step=2,
        max_decode_tokens_per_step=1,
        max_sequences_per_step=1,
    )


class FakeSteppableBackend:
    def __init__(self, capabilities: BackendCapabilities | None = None) -> None:
        self.capabilities = capabilities or backend_capabilities()
        self.operations: list[tuple[str, str, int]] = []
        self.release_counts: Counter[SequenceHandle] = Counter()
        self._states: dict[SequenceHandle, FakeSequenceState] = {}
        self._next_sequence = 0

    def open_sequence(self, request, *, scheduling, events):
        assert type(request) is FakeWork
        handle = SequenceHandle(
            self.capabilities.backend,
            self.capabilities.models[0],
            f"sequence-{self._next_sequence}",
            1,
        )
        self._next_sequence += 1
        self._states[handle] = FakeSequenceState(
            scheduling.request_id,
            request.prompt_tokens,
            request.decode_tokens,
        )
        self.operations.append(("open", scheduling.request_id, 0))
        return handle

    def prefill(self, handle, *, token_budget, events):
        state = self._states[handle]
        processed = min(token_budget, state.remaining_prompt)
        state.remaining_prompt -= processed
        self.operations.append(("prefill", state.request_id, processed))
        status = (
            PrefillStatus.READY
            if state.remaining_prompt == 0
            else PrefillStatus.PARTIAL
        )
        return PrefillOutcome(
            handle,
            status,
            processed,
            state.remaining_prompt,
        )

    def decode(self, handle, *, token_budget, events):
        state = self._states[handle]
        assert state.remaining_prompt == 0
        produced = min(token_budget, state.remaining_decode)
        token_ids = tuple(range(state.next_token, state.next_token + produced))
        state.next_token += produced
        state.remaining_decode -= produced
        self.operations.append(("decode", state.request_id, produced))
        if state.remaining_decode == 0:
            return DecodeOutcome(
                handle,
                DecodeStatus.FINISHED,
                token_ids,
                "x" * produced,
                FinishReason.STOP,
                SequenceCompletion(
                    "x" * state.next_token,
                    1,
                    0,
                    state.next_token - 1,
                    state.next_token - 1,
                    1.0,
                    1.0,
                ),
            )
        return DecodeOutcome(
            handle,
            DecodeStatus.PROGRESSED,
            token_ids,
            "x" * produced,
        )

    def release(self, handle, *, events):
        self.release_counts[handle] += 1
        state = self._states.pop(handle, None)
        request_id = state.request_id if state is not None else "already-released"
        self.operations.append(("release", request_id, 0))
        if state is None:
            return ReleaseOutcome(handle, ReleaseStatus.ALREADY_RELEASED, 0)
        return ReleaseOutcome(handle, ReleaseStatus.RELEASED, 4096)


def policy(*, aging_interval: float = 1.0) -> SchedulerPolicy:
    return SchedulerPolicy(
        (("high", 4), ("normal", 1), ("low", 0)),
        aging_interval,
    )


def metadata(
    request_id: str,
    *,
    service_class: str = "normal",
    weight: int = 1,
    deadline: float | None = None,
) -> SchedulingMetadata:
    return SchedulingMetadata(
        request_id=request_id,
        workflow_id=f"workflow-{request_id}",
        agent_id=f"agent-{request_id}",
        service_class=service_class,
        weight=weight,
        deadline_monotonic=deadline,
    )


def simulator(
    *,
    scheduler_policy: SchedulerPolicy | None = None,
    capabilities: BackendCapabilities | None = None,
):
    clock = FakeMonotonicClock()
    sink = RecordingSink()
    backend = FakeSteppableBackend(capabilities)
    scheduler = DeterministicSchedulerSimulator(
        backend,
        clock=clock,
        events=sink,
        policy=scheduler_policy or policy(),
    )
    return scheduler, backend, clock, sink


def terminal_events(sink: RecordingSink, request_id: str) -> list[SchedulerEvent]:
    return [
        event
        for event in sink.events
        if event.request_id == request_id
        and event.kind
        in {SchedulerEventKind.REQUEST_COMPLETED, SchedulerEventKind.REQUEST_FAILED}
    ]


def test_serial_lifecycle_is_admit_prefill_decode_release_terminal():
    scheduler, backend, _, sink = simulator()
    admitted = scheduler.admit(FakeWork(5, 2), metadata("request-a"))
    assert admitted.lifecycle is SimulatorLifecycle.ADMITTED

    outcomes = [scheduler.tick() for _ in range(6)]
    assert [outcome.action for outcome in outcomes] == [
        TickAction.OPEN_SEQUENCE,
        TickAction.PREFILL,
        TickAction.PREFILL,
        TickAction.PREFILL,
        TickAction.DECODE,
        TickAction.DECODE,
    ]
    snapshot = scheduler.snapshot("request-a")
    assert snapshot.lifecycle is SimulatorLifecycle.TERMINAL
    assert snapshot.termination is SimulatorTermination.COMPLETED
    assert snapshot.release_status is ReleaseStatus.RELEASED
    assert snapshot.service_steps == 6
    assert backend.release_counts[snapshot.handle] == 1
    assert [event.kind for event in sink.events] == [
        SchedulerEventKind.ADMITTED,
        SchedulerEventKind.SEQUENCE_OPENED,
        SchedulerEventKind.PREFILL_COMPLETED,
        SchedulerEventKind.PREFILL_COMPLETED,
        SchedulerEventKind.PREFILL_COMPLETED,
        SchedulerEventKind.DECODE_COMPLETED,
        SchedulerEventKind.DECODE_COMPLETED,
        SchedulerEventKind.SEQUENCE_RELEASED,
        SchedulerEventKind.REQUEST_COMPLETED,
    ]
    assert len(terminal_events(sink, "request-a")) == 1


def test_priority_is_explicit_service_class_policy_not_agent_role():
    scheduler, backend, _, _ = simulator(
        scheduler_policy=policy(aging_interval=1000),
    )
    scheduler.admit(FakeWork(1, 1), metadata("low", service_class="low"))
    scheduler.admit(FakeWork(1, 1), metadata("high", service_class="high"))

    outcome = scheduler.tick()
    assert outcome.selected_request_id == "high"
    assert backend.operations[0] == ("open", "high", 0)


def test_weighted_fairness_uses_deterministic_service_debt():
    scheduler, _, _, _ = simulator(
        scheduler_policy=policy(aging_interval=1000),
    )
    scheduler.admit(FakeWork(1, 100), metadata("weight-1", weight=1))
    scheduler.admit(FakeWork(1, 100), metadata("weight-3", weight=3))

    for _ in range(40):
        scheduler.tick()

    slow = scheduler.snapshot("weight-1").service_steps
    fast = scheduler.snapshot("weight-3").service_steps
    assert (slow, fast) == (10, 30)


def test_aging_prevents_starvation_under_continuous_higher_priority_arrivals():
    scheduler, backend, clock, _ = simulator()
    scheduler.admit(FakeWork(1, 3), metadata("low", service_class="low"))

    for index in range(80):
        scheduler.admit(
            FakeWork(1, 1),
            metadata(f"high-{index}", service_class="high"),
        )
        scheduler.tick()
        clock.advance(1)
        if scheduler.snapshot("low").lifecycle is SimulatorLifecycle.TERMINAL:
            break

    low = scheduler.snapshot("low")
    assert low.termination is SimulatorTermination.COMPLETED
    assert low.service_steps == 5
    assert backend.release_counts[low.handle] == 1


def test_deadline_is_terminal_exactly_once_and_releases_open_sequence():
    scheduler, backend, clock, sink = simulator()
    scheduler.admit(FakeWork(1, 10), metadata("deadline", deadline=5))
    scheduler.tick()
    scheduler.tick()
    handle = scheduler.snapshot("deadline").handle

    clock.advance(5)
    outcome = scheduler.tick()
    assert outcome.action is TickAction.IDLE
    assert outcome.expired_request_ids == ("deadline",)
    snapshot = scheduler.snapshot("deadline")
    assert snapshot.termination is SimulatorTermination.DEADLINE_EXCEEDED
    assert snapshot.release_status is ReleaseStatus.RELEASED
    assert backend.release_counts[handle] == 1
    assert len(terminal_events(sink, "deadline")) == 1
    assert terminal_events(sink, "deadline")[0].error_code == "deadline_exceeded"

    assert scheduler.cancel("deadline") is False
    scheduler.tick()
    scheduler.tick()
    assert backend.release_counts[handle] == 1
    assert len(terminal_events(sink, "deadline")) == 1


def test_cancel_releases_before_single_terminal_event():
    scheduler, backend, _, sink = simulator()
    scheduler.admit(FakeWork(4, 4), metadata("cancelled"))
    scheduler.tick()
    handle = scheduler.snapshot("cancelled").handle

    assert scheduler.cancel("cancelled") is True
    assert scheduler.cancel("cancelled") is False
    snapshot = scheduler.snapshot("cancelled")
    assert snapshot.termination is SimulatorTermination.CANCELLED
    assert snapshot.release_status is ReleaseStatus.RELEASED
    assert backend.release_counts[handle] == 1
    request_events = [
        event.kind for event in sink.events if event.request_id == "cancelled"
    ]
    assert request_events[-2:] == [
        SchedulerEventKind.SEQUENCE_RELEASED,
        SchedulerEventKind.REQUEST_FAILED,
    ]
    assert len(terminal_events(sink, "cancelled")) == 1


def test_admission_and_clock_validation_fail_closed():
    scheduler, _, clock, _ = simulator(
        capabilities=backend_capabilities(max_requests=1, max_sequences=1),
    )
    scheduler.admit(FakeWork(1, 1), metadata("first"))
    with pytest.raises(AdmissionError) as capacity:
        scheduler.admit(FakeWork(1, 1), metadata("second"))
    assert capacity.value.code == "capacity"
    with pytest.raises(AdmissionError) as duplicate:
        scheduler.admit(FakeWork(1, 1), metadata("first"))
    assert duplicate.value.code == "duplicate_request"
    with pytest.raises(AdmissionError) as unknown_class:
        scheduler.cancel("first")
        scheduler.admit(FakeWork(1, 1), metadata("unknown", service_class="role-name"))
    assert unknown_class.value.code == "unknown_service_class"
    with pytest.raises(AdmissionError) as elapsed:
        scheduler.admit(FakeWork(1, 1), metadata("elapsed", deadline=0))
    assert elapsed.value.code == "deadline_elapsed"
    with pytest.raises(ValueError):
        clock.advance(float("inf"))


WORKLOADS = st.lists(
    st.tuples(
        st.sampled_from(("high", "normal", "low")),
        st.integers(min_value=1, max_value=4),
        st.integers(min_value=1, max_value=5),
        st.integers(min_value=1, max_value=5),
        st.one_of(st.none(), st.integers(min_value=1, max_value=30)),
    ),
    min_size=1,
    max_size=8,
)


def run_workload(workload):
    scheduler, backend, clock, sink = simulator(
        capabilities=backend_capabilities(max_requests=16, max_sequences=4),
    )
    for index, (service_class, weight, prompt, decode, deadline) in enumerate(workload):
        scheduler.admit(
            FakeWork(prompt, decode),
            metadata(
                f"request-{index}",
                service_class=service_class,
                weight=weight,
                deadline=deadline,
            ),
        )
    for _ in range(200):
        if not scheduler.active_request_ids:
            break
        scheduler.tick()
        clock.advance(1)
    assert not scheduler.active_request_ids

    terminal_counts = Counter(
        event.request_id
        for event in sink.events
        if event.kind
        in {SchedulerEventKind.REQUEST_COMPLETED, SchedulerEventKind.REQUEST_FAILED}
    )
    assert terminal_counts == Counter(
        {f"request-{index}": 1 for index in range(len(workload))}
    )
    assert all(count == 1 for count in backend.release_counts.values())
    snapshots = tuple(
        scheduler.snapshot(f"request-{index}") for index in range(len(workload))
    )
    return tuple(sink.events), tuple(backend.operations), snapshots


@settings(max_examples=50, deadline=None)
@given(WORKLOADS)
def test_randomized_workloads_are_deterministic_and_terminal_once(workload):
    assert run_workload(workload) == run_workload(workload)
