from __future__ import annotations

import pytest

from inference_runtime import (
    BackendCapabilities,
    BackendConformanceError,
    BatchSteppableBackend,
    DecodeOutcome,
    DecodeStatus,
    FinishReason,
    InferencePort,
    ManagedBackend,
    PrefillOutcome,
    PrefillStatus,
    ReleaseOutcome,
    ReleaseStatus,
    SchedulerEvent,
    SchedulerEventKind,
    SchedulerEventSink,
    SchedulingMetadata,
    SequenceCompletion,
    SequenceHandle,
    SequenceStep,
    SteppableBackend,
    require_inference_port,
    require_batch_steppable_backend,
    require_managed_backend,
    require_steppable_backend,
)


def managed_capabilities() -> BackendCapabilities:
    return BackendCapabilities(
        backend="managed",
        models=("model-a",),
        supports_full_request=True,
        supports_sequence_steps=False,
        supports_streaming=True,
        supports_cancellation=True,
        supports_chunked_prefill=False,
        supports_decode_batching=False,
        supports_continuous_batching=True,
        supports_prefix_cache=True,
        supports_session_cache=False,
        supports_explicit_release=False,
        max_context_tokens=4096,
        max_output_tokens=512,
        max_concurrent_requests=16,
        max_concurrent_sequences=None,
        max_prefill_tokens_per_step=None,
        max_decode_tokens_per_step=None,
        max_sequences_per_step=None,
    )


def steppable_capabilities(**overrides) -> BackendCapabilities:
    values = dict(
        backend="steppable",
        models=("model-a",),
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
        max_context_tokens=4096,
        max_output_tokens=512,
        max_concurrent_requests=8,
        max_concurrent_sequences=8,
        max_prefill_tokens_per_step=512,
        max_decode_tokens_per_step=1,
        max_sequences_per_step=1,
    )
    values.update(overrides)
    return BackendCapabilities(**values)


def scheduling() -> SchedulingMetadata:
    return SchedulingMetadata(
        request_id="request-a",
        workflow_id="workflow-a",
        agent_id="agent-a",
        service_class="throughput",
        weight=10,
        deadline_monotonic=100,
    )


class RecordingSink:
    def __init__(self) -> None:
        self.events = []

    def publish(self, event: SchedulerEvent) -> None:
        self.events.append(event)


class FakeInferencePort:
    def infer(self, request, *, scheduling, events):
        events.publish(
            SchedulerEvent(
                SchedulerEventKind.REQUEST_COMPLETED,
                scheduling.request_id,
                1,
            )
        )
        return {"request": request, "service_class": scheduling.service_class}

    def cancel(self, request_id: str) -> bool:
        return bool(request_id)


class FakeManagedBackend:
    capabilities = managed_capabilities()

    def generate(self, request, *, scheduling, events):
        events.publish(
            SchedulerEvent(
                SchedulerEventKind.REQUEST_COMPLETED,
                scheduling.request_id,
                1,
            )
        )
        return {"output": request}

    def cancel(self, request_id: str) -> bool:
        return bool(request_id)


class FakeSteppableBackend:
    capabilities = steppable_capabilities()

    def open_sequence(self, request, *, scheduling, events):
        handle = SequenceHandle(
            self.capabilities.backend,
            self.capabilities.models[0],
            f"sequence-{request}",
            1,
        )
        events.publish(
            SchedulerEvent(
                SchedulerEventKind.SEQUENCE_OPENED,
                scheduling.request_id,
                1,
                handle=handle,
            )
        )
        return handle

    def prefill(self, handle, *, token_budget, events):
        events.publish(
            SchedulerEvent(
                SchedulerEventKind.PREFILL_COMPLETED,
                "request-a",
                2,
                handle=handle,
                tokens=token_budget,
            )
        )
        return PrefillOutcome(handle, PrefillStatus.READY, token_budget, 0)

    def decode(self, handle, *, token_budget, events):
        assert token_budget == 1
        events.publish(
            SchedulerEvent(
                SchedulerEventKind.DECODE_COMPLETED,
                "request-a",
                3,
                handle=handle,
                tokens=1,
            )
        )
        return DecodeOutcome(
            handle,
            DecodeStatus.FINISHED,
            (7,),
            "x",
            FinishReason.STOP,
            SequenceCompletion("x", 1, 0, 1, 1, 1.0, 1.0),
        )

    def release(self, handle, *, events):
        events.publish(
            SchedulerEvent(
                SchedulerEventKind.SEQUENCE_RELEASED,
                "request-a",
                4,
                handle=handle,
            )
        )
        return ReleaseOutcome(handle, ReleaseStatus.RELEASED, 4096)


class FakeBatchSteppableBackend(FakeSteppableBackend):
    capabilities = steppable_capabilities(
        supports_decode_batching=True,
        supports_continuous_batching=True,
        max_sequences_per_step=4,
    )

    def prefill_batch(self, steps, *, events):
        return tuple(
            self.prefill(
                step.handle,
                token_budget=step.token_budget,
                events=events,
            )
            for step in steps
        )

    def decode_batch(self, steps, *, events):
        return tuple(
            self.decode(
                step.handle,
                token_budget=step.token_budget,
                events=events,
            )
            for step in steps
        )


def test_high_level_port_is_distinct_from_backend_interfaces():
    port = require_inference_port(FakeInferencePort())
    sink = RecordingSink()
    result = port.infer("prompt", scheduling=scheduling(), events=sink)

    assert isinstance(port, InferencePort)
    assert not isinstance(port, ManagedBackend)
    assert not isinstance(port, SteppableBackend)
    assert result["service_class"] == "throughput"
    assert sink.events[-1].kind is SchedulerEventKind.REQUEST_COMPLETED


def test_managed_backend_conforms_without_fake_step_methods():
    backend = require_managed_backend(FakeManagedBackend())
    sink = RecordingSink()
    result = backend.generate("prompt", scheduling=scheduling(), events=sink)

    assert isinstance(backend, ManagedBackend)
    assert not isinstance(backend, SteppableBackend)
    assert not any(
        hasattr(backend, name)
        for name in ("open_sequence", "prefill", "decode", "release")
    )
    assert result == {"output": "prompt"}


def test_steppable_backend_exposes_explicit_sequence_lifecycle():
    backend = require_steppable_backend(FakeSteppableBackend())
    sink = RecordingSink()
    handle = backend.open_sequence("a", scheduling=scheduling(), events=sink)
    prefilled = backend.prefill(handle, token_budget=8, events=sink)
    decoded = backend.decode(handle, token_budget=1, events=sink)
    released = backend.release(handle, events=sink)

    assert isinstance(backend, SteppableBackend)
    assert isinstance(sink, SchedulerEventSink)
    assert prefilled.status is PrefillStatus.READY
    assert decoded.finish_reason is FinishReason.STOP
    assert released.status is ReleaseStatus.RELEASED
    assert [event.kind for event in sink.events] == [
        SchedulerEventKind.SEQUENCE_OPENED,
        SchedulerEventKind.PREFILL_COMPLETED,
        SchedulerEventKind.DECODE_COMPLETED,
        SchedulerEventKind.SEQUENCE_RELEASED,
    ]


def test_batch_backend_registration_requires_real_batch_methods():
    backend = require_batch_steppable_backend(FakeBatchSteppableBackend())
    sink = RecordingSink()
    handles = (
        backend.open_sequence("a", scheduling=scheduling(), events=sink),
        backend.open_sequence("b", scheduling=scheduling(), events=sink),
    )
    outcomes = backend.prefill_batch(
        tuple(SequenceStep(handle, 2) for handle in handles),
        events=sink,
    )
    assert isinstance(backend, BatchSteppableBackend)
    assert len(outcomes) == 2

    class MissingBatchMethods(FakeSteppableBackend):
        capabilities = FakeBatchSteppableBackend.capabilities

    with pytest.raises(BackendConformanceError):
        require_batch_steppable_backend(MissingBatchMethods())
    with pytest.raises(BackendConformanceError):
        require_batch_steppable_backend(FakeSteppableBackend())


def test_adapter_registration_fails_closed_on_shape_or_capability_mismatch():
    class MissingCancel:
        capabilities = managed_capabilities()

        def generate(self, request, *, scheduling, events):
            return request

    class MisadvertisedManaged(FakeManagedBackend):
        capabilities = steppable_capabilities()

    class MisadvertisedSteppable(FakeSteppableBackend):
        capabilities = managed_capabilities()

    invalid_pairs = (
        (require_inference_port, object()),
        (require_managed_backend, MissingCancel()),
        (require_managed_backend, MisadvertisedManaged()),
        (require_steppable_backend, MisadvertisedSteppable()),
    )
    for require_adapter, adapter in invalid_pairs:
        with pytest.raises(BackendConformanceError):
            require_adapter(adapter)
