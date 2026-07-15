from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from inference_runtime import (
    BackendCapabilities,
    CacheScope,
    CacheVisibility,
    ContractValidationError,
    DecodeOutcome,
    DecodeStatus,
    FinishReason,
    PrefillOutcome,
    PrefillStatus,
    ReleaseOutcome,
    ReleaseStatus,
    SchedulerEvent,
    SchedulerEventKind,
    SchedulingMetadata,
    SessionCacheControl,
    SequenceCompletion,
    SequenceHandle,
    SequenceStep,
    validate_sequence_batch,
)


def managed_capabilities(**overrides) -> BackendCapabilities:
    values = {
        "backend": "managed-local",
        "models": ("model-a",),
        "supports_full_request": True,
        "supports_sequence_steps": False,
        "supports_streaming": True,
        "supports_cancellation": True,
        "supports_chunked_prefill": False,
        "supports_decode_batching": False,
        "supports_continuous_batching": True,
        "supports_prefix_cache": True,
        "supports_session_cache": False,
        "supports_explicit_release": False,
        "max_context_tokens": 4096,
        "max_output_tokens": 512,
        "max_concurrent_requests": 16,
        "max_concurrent_sequences": None,
        "max_prefill_tokens_per_step": None,
        "max_decode_tokens_per_step": None,
        "max_sequences_per_step": None,
    }
    values.update(overrides)
    return BackendCapabilities(**values)


def steppable_capabilities(**overrides) -> BackendCapabilities:
    values = {
        "backend": "steppable-local",
        "models": ("model-a",),
        "supports_full_request": False,
        "supports_sequence_steps": True,
        "supports_streaming": True,
        "supports_cancellation": True,
        "supports_chunked_prefill": True,
        "supports_decode_batching": False,
        "supports_continuous_batching": False,
        "supports_prefix_cache": False,
        "supports_session_cache": False,
        "supports_explicit_release": True,
        "max_context_tokens": 4096,
        "max_output_tokens": 512,
        "max_concurrent_requests": 8,
        "max_concurrent_sequences": 8,
        "max_prefill_tokens_per_step": 512,
        "max_decode_tokens_per_step": 1,
        "max_sequences_per_step": 1,
    }
    values.update(overrides)
    return BackendCapabilities(**values)


def handle() -> SequenceHandle:
    return SequenceHandle("backend-a", "model-a", "sequence-a", 3)


def test_sequence_handle_and_scheduling_metadata_are_immutable_and_opaque():
    sequence = handle()
    scheduling = SchedulingMetadata(
        request_id="request-a",
        workflow_id="opaque/workflow/Planner",
        agent_id="opaque-agent-value",
        service_class="interactive-critical",
        weight=100,
        deadline_monotonic=123,
    )

    assert sequence.generation == 3
    assert scheduling.deadline_monotonic == 123.0
    assert scheduling.workflow_id == "opaque/workflow/Planner"
    assert "role" not in {field.name for field in fields(SchedulingMetadata)}
    with pytest.raises(FrozenInstanceError):
        scheduling.weight = 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", ""),
        ("model", "model\ncontrol"),
        ("sequence", ""),
        ("generation", True),
        ("generation", -1),
    ],
)
def test_sequence_handle_rejects_invalid_identity(field, value):
    values = {
        "backend": "backend-a",
        "model": "model-a",
        "sequence": "sequence-a",
        "generation": 0,
    }
    values[field] = value
    with pytest.raises(ContractValidationError) as captured:
        SequenceHandle(**values)
    assert captured.value.field == field


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", ""),
        ("workflow_id", "workflow\ncontrol"),
        ("agent_id", ""),
        ("service_class", ""),
        ("weight", True),
        ("weight", 0),
        ("weight", 1_000_001),
        ("deadline_monotonic", float("nan")),
        ("deadline_monotonic", -1),
    ],
)
def test_scheduling_metadata_fails_closed(field, value):
    values = {
        "request_id": "request-a",
        "workflow_id": "workflow-a",
        "agent_id": "agent-a",
        "service_class": "throughput",
        "weight": 1,
        "deadline_monotonic": None,
    }
    values[field] = value
    with pytest.raises(ContractValidationError) as captured:
        SchedulingMetadata(**values)
    assert captured.value.field == field


def test_session_cache_control_requires_private_scope_and_explicit_lineage():
    private_scope = CacheScope(
        "tenant-a",
        "workflow-a",
        "agent-a",
        CacheVisibility.PRIVATE,
    )
    control = SessionCacheControl("session-a", parent_generation=7, commit=True)
    scheduling = SchedulingMetadata(
        "request-a",
        "workflow-a",
        "agent-a",
        "throughput",
        1,
        None,
        private_scope,
        control,
    )

    assert scheduling.session_cache == control
    assert scheduling.session_cache.parent_generation == 7

    invalid = (
        lambda: SessionCacheControl("", None, True),
        lambda: SessionCacheControl("session-a", 0, True),
        lambda: SessionCacheControl("session-a", 1, 1),
        lambda: SchedulingMetadata(
            "request-a",
            "workflow-a",
            "agent-a",
            "throughput",
            1,
            None,
            None,
            control,
        ),
        lambda: SchedulingMetadata(
            "request-a",
            "workflow-a",
            "agent-a",
            "throughput",
            1,
            None,
            CacheScope(
                "tenant-a",
                "workflow-a",
                "agent-a",
                CacheVisibility.WORKFLOW,
            ),
            control,
        ),
        lambda: SchedulingMetadata(
            "request-a",
            "workflow-other",
            "agent-a",
            "throughput",
            1,
            None,
            private_scope,
            None,
        ),
        lambda: SchedulingMetadata(
            "request-a",
            "workflow-a",
            "agent-other",
            "throughput",
            1,
            None,
            private_scope,
            None,
        ),
    )
    for factory in invalid:
        with pytest.raises(ContractValidationError):
            factory()


def test_backend_capabilities_distinguish_managed_and_steppable_limits():
    managed = managed_capabilities()
    steppable = steppable_capabilities()
    batched = steppable_capabilities(
        supports_decode_batching=True,
        supports_continuous_batching=True,
        max_sequences_per_step=4,
    )

    assert managed.supports_full_request and not managed.supports_sequence_steps
    assert managed.max_concurrent_sequences is None
    assert managed.max_prefill_tokens_per_step is None
    assert steppable.supports_sequence_steps and steppable.supports_explicit_release
    assert batched.max_sequences_per_step == 4


def test_backend_capabilities_reject_inconsistent_flags_and_limits():
    invalid_factories = (
        lambda: managed_capabilities(supports_streaming=1),
        lambda: managed_capabilities(
            supports_full_request=False,
            supports_sequence_steps=False,
        ),
        lambda: managed_capabilities(supports_chunked_prefill=True),
        lambda: managed_capabilities(max_prefill_tokens_per_step=1),
        lambda: managed_capabilities(max_concurrent_sequences=1),
        lambda: steppable_capabilities(supports_explicit_release=False),
        lambda: steppable_capabilities(supports_cancellation=False),
        lambda: steppable_capabilities(max_sequences_per_step=2),
        lambda: steppable_capabilities(
            supports_chunked_prefill=False,
            max_prefill_tokens_per_step=512,
        ),
        lambda: steppable_capabilities(max_output_tokens=5000),
        lambda: steppable_capabilities(max_decode_tokens_per_step=513),
        lambda: steppable_capabilities(max_concurrent_sequences=9),
        lambda: steppable_capabilities(
            supports_continuous_batching=True,
            supports_decode_batching=False,
        ),
    )
    for factory in invalid_factories:
        with pytest.raises(ContractValidationError):
            factory()


def test_sequence_batch_contract_is_bounded_and_has_unique_handles():
    first = SequenceStep(handle(), 2)
    second = SequenceStep(
        SequenceHandle("backend-a", "model-a", "sequence-b", 3),
        1,
    )
    assert validate_sequence_batch(
        (first, second),
        max_sequences=2,
        max_tokens_per_step=2,
    ) == (first, second)

    invalid = (
        lambda: SequenceStep(handle(), 0),
        lambda: validate_sequence_batch(
            (), max_sequences=2, max_tokens_per_step=2
        ),
        lambda: validate_sequence_batch(
            (first, first), max_sequences=2, max_tokens_per_step=2
        ),
        lambda: validate_sequence_batch(
            (first, second), max_sequences=1, max_tokens_per_step=2
        ),
        lambda: validate_sequence_batch(
            (first,), max_sequences=2, max_tokens_per_step=1
        ),
    )
    for operation in invalid:
        with pytest.raises(ContractValidationError):
            operation()


def test_prefill_decode_and_release_outcomes_are_discriminated():
    sequence = handle()
    partial = PrefillOutcome(sequence, PrefillStatus.PARTIAL, 128, 128)
    ready = PrefillOutcome(sequence, PrefillStatus.READY, 128, 0)
    progressed = DecodeOutcome(
        sequence,
        DecodeStatus.PROGRESSED,
        (7,),
        "x",
    )
    finished = DecodeOutcome(
        sequence,
        DecodeStatus.FINISHED,
        (),
        "",
        FinishReason.STOP,
        SequenceCompletion("{}", 2, 0, 1, 1, 1, 2),
    )
    released = ReleaseOutcome(sequence, ReleaseStatus.RELEASED, 4096)

    assert partial.remaining_tokens == 128
    assert ready.status is PrefillStatus.READY
    assert progressed.finish_reason is None
    assert finished.finish_reason is FinishReason.STOP
    assert released.released_bytes == 4096


def test_sequence_completion_validates_native_token_timing_samples():
    completion = SequenceCompletion(
        "{}",
        2,
        2,
        2,
        5,
        1,
        10,
        first_sample_ms=2.5,
        first_final_ms=7.5,
        sample_itl_ms=(1, 2, 3, 4),
        final_itl_ms=(2,),
    )
    assert completion.sample_itl_ms == (1.0, 2.0, 3.0, 4.0)
    assert completion.final_itl_ms == (2.0,)

    invalid = (
        lambda: SequenceCompletion(
            "{}", 2, 0, 1, 1, 1, 1, sample_itl_ms=(1,)
        ),
        lambda: SequenceCompletion(
            "{}", 2, 0, 2, 2, 1, 1, final_itl_ms=(1, 2)
        ),
        lambda: SequenceCompletion(
            "{}", 2, 0, 1, 2, 1, 1, sample_itl_ms=(float("nan"),)
        ),
        lambda: SequenceCompletion(
            "{}", 2, 0, 1, 2, 1, 1, first_sample_ms=0, sample_itl_ms=(1,)
        ),
    )
    for factory in invalid:
        with pytest.raises(ContractValidationError):
            factory()


def test_sequence_completion_validates_session_cache_metadata():
    completion = SequenceCompletion(
        "{}",
        20,
        0,
        1,
        1,
        1,
        1,
        cached_prompt_tokens=16,
        cache_hit=True,
        cache_match="session",
        session_id="session-a",
        session_parent_generation=1,
        session_generation=2,
        session_copy_on_write=True,
    )
    assert completion.session_generation == 2

    invalid = (
        lambda: replace(completion, session_copy_on_write=1),
        lambda: replace(completion, session_id=None),
        lambda: replace(completion, session_parent_generation=None),
        lambda: replace(completion, cache_hit=False),
        lambda: replace(completion, cached_prompt_tokens=0),
        lambda: replace(completion, cache_match="unsafe"),
    )
    for factory in invalid:
        with pytest.raises(ContractValidationError):
            factory()


def test_step_outcomes_reject_impossible_states():
    sequence = handle()
    invalid_factories = (
        lambda: PrefillOutcome(sequence, PrefillStatus.PARTIAL, 0, 10),
        lambda: PrefillOutcome(sequence, PrefillStatus.PARTIAL, 10, 0),
        lambda: PrefillOutcome(sequence, PrefillStatus.READY, 10, 1),
        lambda: PrefillOutcome(sequence, "ready", 10, 0),
        lambda: DecodeOutcome(sequence, DecodeStatus.PROGRESSED, (), ""),
        lambda: DecodeOutcome(
            sequence,
            DecodeStatus.PROGRESSED,
            (1,),
            "x",
            FinishReason.STOP,
        ),
        lambda: DecodeOutcome(sequence, DecodeStatus.FINISHED, (), ""),
        lambda: DecodeOutcome(
            sequence,
            DecodeStatus.FAILED,
            (),
            "",
            error_code=None,
        ),
        lambda: DecodeOutcome(sequence, DecodeStatus.PROGRESSED, (True,), "x"),
        lambda: ReleaseOutcome(sequence, ReleaseStatus.NOT_FOUND, 1),
        lambda: ReleaseOutcome(sequence, "released", 0),
    )
    for factory in invalid_factories:
        with pytest.raises(ContractValidationError):
            factory()


def test_scheduler_events_validate_kind_specific_payloads():
    sequence = handle()
    event = SchedulerEvent(
        SchedulerEventKind.PREFILL_COMPLETED,
        "request-a",
        1,
        handle=sequence,
        tokens=128,
    )
    failure = SchedulerEvent(
        SchedulerEventKind.REQUEST_FAILED,
        "request-a",
        2.5,
        error_code="deadline_exceeded",
    )
    assert event.at_monotonic == 1.0
    assert failure.error_code == "deadline_exceeded"

    invalid_events = (
        lambda: replace(event, handle=None),
        lambda: replace(event, tokens=None),
        lambda: replace(event, error_code="not-allowed"),
        lambda: replace(failure, error_code=None),
        lambda: replace(failure, at_monotonic=float("inf")),
    )
    for factory in invalid_events:
        with pytest.raises(ContractValidationError):
            factory()
