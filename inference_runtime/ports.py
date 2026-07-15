from __future__ import annotations

from typing import Any, Protocol, TypeVar, cast, runtime_checkable

from .contracts import (
    BackendCapabilities,
    DecodeOutcome,
    PrefillOutcome,
    ReleaseOutcome,
    SchedulerEvent,
    SchedulingMetadata,
    SequenceHandle,
    SequenceStep,
)


RequestT = TypeVar("RequestT", contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)


@runtime_checkable
class SchedulerEventSink(Protocol):
    def publish(self, event: SchedulerEvent) -> None: ...


@runtime_checkable
class InferencePort(Protocol[RequestT, ResultT]):
    """High-level harness port implemented by the inference control plane."""

    def infer(
        self,
        request: RequestT,
        *,
        scheduling: SchedulingMetadata,
        events: SchedulerEventSink,
    ) -> ResultT: ...

    def cancel(self, request_id: str) -> bool: ...


@runtime_checkable
class ManagedBackend(Protocol[RequestT, ResultT]):
    """Backend that owns scheduling and exposes only full-request generation."""

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def generate(
        self,
        request: RequestT,
        *,
        scheduling: SchedulingMetadata,
        events: SchedulerEventSink,
    ) -> ResultT: ...

    def cancel(self, request_id: str) -> bool: ...


@runtime_checkable
class SteppableBackend(Protocol[RequestT]):
    """Backend whose sequence lifecycle is explicitly owned by the control plane."""

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def open_sequence(
        self,
        request: RequestT,
        *,
        scheduling: SchedulingMetadata,
        events: SchedulerEventSink,
    ) -> SequenceHandle: ...

    def prefill(
        self,
        handle: SequenceHandle,
        *,
        token_budget: int,
        events: SchedulerEventSink,
    ) -> PrefillOutcome: ...

    def decode(
        self,
        handle: SequenceHandle,
        *,
        token_budget: int,
        events: SchedulerEventSink,
    ) -> DecodeOutcome: ...

    def release(
        self,
        handle: SequenceHandle,
        *,
        events: SchedulerEventSink,
    ) -> ReleaseOutcome: ...


@runtime_checkable
class BatchSteppableBackend(SteppableBackend[RequestT], Protocol[RequestT]):
    """Steppable backend that performs one real native call per microbatch."""

    def prefill_batch(
        self,
        steps: tuple[SequenceStep, ...],
        *,
        events: SchedulerEventSink,
    ) -> tuple[PrefillOutcome, ...]: ...

    def decode_batch(
        self,
        steps: tuple[SequenceStep, ...],
        *,
        events: SchedulerEventSink,
    ) -> tuple[DecodeOutcome, ...]: ...


class BackendConformanceError(TypeError):
    pass


def require_inference_port(adapter: object) -> InferencePort[Any, Any]:
    if not isinstance(adapter, InferencePort):
        raise BackendConformanceError("adapter does not implement InferencePort")
    return cast(InferencePort[Any, Any], adapter)


def require_managed_backend(adapter: object) -> ManagedBackend[Any, Any]:
    if not isinstance(adapter, ManagedBackend):
        raise BackendConformanceError("adapter does not implement ManagedBackend")
    capabilities = adapter.capabilities
    if type(capabilities) is not BackendCapabilities:
        raise BackendConformanceError("managed backend capabilities are not validated")
    if not capabilities.supports_full_request:
        raise BackendConformanceError("managed backend does not advertise full-request support")
    return cast(ManagedBackend[Any, Any], adapter)


def require_steppable_backend(adapter: object) -> SteppableBackend[Any]:
    if not isinstance(adapter, SteppableBackend):
        raise BackendConformanceError("adapter does not implement SteppableBackend")
    capabilities = adapter.capabilities
    if type(capabilities) is not BackendCapabilities:
        raise BackendConformanceError("steppable backend capabilities are not validated")
    if not capabilities.supports_sequence_steps:
        raise BackendConformanceError("backend does not advertise sequence-step support")
    return cast(SteppableBackend[Any], adapter)


def require_batch_steppable_backend(adapter: object) -> BatchSteppableBackend[Any]:
    require_steppable_backend(adapter)
    if not isinstance(adapter, BatchSteppableBackend):
        raise BackendConformanceError(
            "backend advertises batching but does not implement batch step methods"
        )
    capabilities = adapter.capabilities
    if not capabilities.supports_decode_batching:
        raise BackendConformanceError("backend does not advertise decode batching")
    if capabilities.max_sequences_per_step is None or capabilities.max_sequences_per_step < 2:
        raise BackendConformanceError("batch backend has no usable sequence batch limit")
    return cast(BatchSteppableBackend[Any], adapter)
