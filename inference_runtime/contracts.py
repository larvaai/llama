from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .cache import CacheScope, CacheVisibility


MAX_IDENTIFIER_BYTES = 256
MAX_SERVICE_CLASS_BYTES = 64
MAX_ERROR_CODE_BYTES = 128
MAX_SCHEDULING_WEIGHT = 1_000_000


class ContractValidationError(ValueError):
    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


def _fail(field: str, message: str) -> None:
    raise ContractValidationError(field, message)


def _bounded_text(value: Any, field: str, max_bytes: int) -> str:
    if type(value) is not str or not value:
        _fail(field, "must be a non-empty string")
    if len(value.encode("utf-8")) > max_bytes:
        _fail(field, f"must be at most {max_bytes} UTF-8 bytes")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        _fail(field, "must not contain control characters")
    return value


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        _fail(field, "must be a positive integer")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        _fail(field, "must be a non-negative integer")
    return value


def _optional_positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field)


def _monotonic_time(value: Any, field: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        _fail(field, "must be a finite non-negative monotonic timestamp")
    return float(value)


@dataclass(frozen=True, slots=True)
class SequenceHandle:
    """Opaque sequence identity scoped to one backend process generation."""

    backend: str
    model: str
    sequence: str
    generation: int

    def __post_init__(self) -> None:
        _bounded_text(self.backend, "backend", MAX_IDENTIFIER_BYTES)
        _bounded_text(self.model, "model", MAX_IDENTIFIER_BYTES)
        _bounded_text(self.sequence, "sequence", MAX_IDENTIFIER_BYTES)
        _non_negative_int(self.generation, "generation")


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Explicit adapter promises; unsupported step limits must remain ``None``."""

    backend: str
    models: tuple[str, ...]
    supports_full_request: bool
    supports_sequence_steps: bool
    supports_streaming: bool
    supports_cancellation: bool
    supports_chunked_prefill: bool
    supports_decode_batching: bool
    supports_continuous_batching: bool
    supports_prefix_cache: bool
    supports_session_cache: bool
    supports_explicit_release: bool
    max_context_tokens: int
    max_output_tokens: int
    max_concurrent_requests: int
    max_concurrent_sequences: int | None
    max_prefill_tokens_per_step: int | None
    max_decode_tokens_per_step: int | None
    max_sequences_per_step: int | None

    def __post_init__(self) -> None:
        _bounded_text(self.backend, "backend", MAX_IDENTIFIER_BYTES)
        if type(self.models) is not tuple or not self.models:
            _fail("models", "must be a non-empty tuple")
        for index, model in enumerate(self.models):
            _bounded_text(model, f"models[{index}]", MAX_IDENTIFIER_BYTES)
        if len(set(self.models)) != len(self.models):
            _fail("models", "must not contain duplicates")

        flag_names = (
            "supports_full_request",
            "supports_sequence_steps",
            "supports_streaming",
            "supports_cancellation",
            "supports_chunked_prefill",
            "supports_decode_batching",
            "supports_continuous_batching",
            "supports_prefix_cache",
            "supports_session_cache",
            "supports_explicit_release",
        )
        for name in flag_names:
            if type(getattr(self, name)) is not bool:
                _fail(name, "must be a boolean")
        if not (self.supports_full_request or self.supports_sequence_steps):
            _fail(
                "supports_full_request",
                "at least one full-request or sequence-step interface is required",
            )

        max_context = _positive_int(self.max_context_tokens, "max_context_tokens")
        max_output = _positive_int(self.max_output_tokens, "max_output_tokens")
        max_concurrent = _positive_int(
            self.max_concurrent_requests,
            "max_concurrent_requests",
        )
        max_concurrent_sequences = _optional_positive_int(
            self.max_concurrent_sequences,
            "max_concurrent_sequences",
        )
        if max_output > max_context:
            _fail("max_output_tokens", "must not exceed max_context_tokens")

        prefill_limit = _optional_positive_int(
            self.max_prefill_tokens_per_step,
            "max_prefill_tokens_per_step",
        )
        decode_limit = _optional_positive_int(
            self.max_decode_tokens_per_step,
            "max_decode_tokens_per_step",
        )
        sequence_limit = _optional_positive_int(
            self.max_sequences_per_step,
            "max_sequences_per_step",
        )
        if not self.supports_sequence_steps:
            step_flags = (
                "supports_chunked_prefill",
                "supports_decode_batching",
                "supports_explicit_release",
            )
            if any(getattr(self, name) for name in step_flags):
                _fail(
                    "supports_sequence_steps",
                    "step-only flags require sequence-step support",
                )
            if any(
                value is not None
                for value in (
                    max_concurrent_sequences,
                    prefill_limit,
                    decode_limit,
                    sequence_limit,
                )
            ):
                _fail(
                    "supports_sequence_steps",
                    "step limits must be None without sequence-step support",
                )
            return

        if not self.supports_cancellation:
            _fail(
                "supports_cancellation",
                "sequence-step backends must support bounded cancellation",
            )

        if not self.supports_explicit_release:
            _fail(
                "supports_explicit_release",
                "sequence-step backends must release sequence resources explicitly",
            )
        if (
            max_concurrent_sequences is None
            or prefill_limit is None
            or decode_limit is None
            or sequence_limit is None
        ):
            _fail(
                "supports_sequence_steps",
                "sequence-step backends must declare every step limit",
            )
        if prefill_limit > max_context:
            _fail(
                "max_prefill_tokens_per_step",
                "must not exceed max_context_tokens",
            )
        if decode_limit > max_output:
            _fail(
                "max_decode_tokens_per_step",
                "must not exceed max_output_tokens",
            )
        if max_concurrent_sequences > max_concurrent:
            _fail(
                "max_concurrent_sequences",
                "must not exceed max_concurrent_requests",
            )
        if sequence_limit > max_concurrent_sequences:
            _fail(
                "max_sequences_per_step",
                "must not exceed max_concurrent_sequences",
            )
        if self.supports_decode_batching and sequence_limit < 2:
            _fail(
                "max_sequences_per_step",
                "decode batching requires a limit of at least two sequences",
            )
        if not self.supports_decode_batching and sequence_limit != 1:
            _fail(
                "max_sequences_per_step",
                "a non-batching backend must step exactly one sequence",
            )
        if self.supports_continuous_batching and not (
            self.supports_chunked_prefill and self.supports_decode_batching
        ):
            _fail(
                "supports_continuous_batching",
                "continuous batching requires chunked prefill and decode batching",
            )
        if not self.supports_chunked_prefill and prefill_limit < max_context:
            _fail(
                "max_prefill_tokens_per_step",
                "must cover max_context_tokens when chunked prefill is unsupported",
            )


@dataclass(frozen=True, slots=True)
class SessionCacheControl:
    """Immutable native session snapshot request with explicit parent lineage."""

    session_id: str
    parent_generation: int | None = None
    commit: bool = True

    def __post_init__(self) -> None:
        _bounded_text(self.session_id, "session_id", MAX_IDENTIFIER_BYTES)
        if self.parent_generation is not None:
            _positive_int(self.parent_generation, "parent_generation")
        if type(self.commit) is not bool:
            _fail("commit", "must be a bool")


@dataclass(frozen=True, slots=True)
class SchedulingMetadata:
    """Policy-only metadata. Workflow and agent identifiers remain opaque."""

    request_id: str
    workflow_id: str
    agent_id: str
    service_class: str
    weight: int
    deadline_monotonic: float | None
    cache_scope: CacheScope | None = None
    session_cache: SessionCacheControl | None = None

    def __post_init__(self) -> None:
        _bounded_text(self.request_id, "request_id", MAX_IDENTIFIER_BYTES)
        _bounded_text(self.workflow_id, "workflow_id", MAX_IDENTIFIER_BYTES)
        _bounded_text(self.agent_id, "agent_id", MAX_IDENTIFIER_BYTES)
        _bounded_text(
            self.service_class,
            "service_class",
            MAX_SERVICE_CLASS_BYTES,
        )
        weight = _positive_int(self.weight, "weight")
        if weight > MAX_SCHEDULING_WEIGHT:
            _fail("weight", f"must not exceed {MAX_SCHEDULING_WEIGHT}")
        if self.deadline_monotonic is not None:
            object.__setattr__(
                self,
                "deadline_monotonic",
                _monotonic_time(self.deadline_monotonic, "deadline_monotonic"),
            )
        if self.cache_scope is not None:
            if type(self.cache_scope) is not CacheScope:
                _fail("cache_scope", "must be CacheScope or None")
            if self.cache_scope.workflow_id != self.workflow_id:
                _fail("cache_scope", "workflow_id must match scheduling metadata")
            if self.cache_scope.agent_id != self.agent_id:
                _fail("cache_scope", "agent_id must match scheduling metadata")
        if self.session_cache is not None:
            if type(self.session_cache) is not SessionCacheControl:
                _fail("session_cache", "must be SessionCacheControl or None")
            if (
                self.cache_scope is None
                or self.cache_scope.visibility is not CacheVisibility.PRIVATE
            ):
                _fail("session_cache", "requires a private cache_scope")


@dataclass(frozen=True, slots=True)
class SequenceStep:
    """One sequence's bounded contribution to a native batch operation."""

    handle: SequenceHandle
    token_budget: int

    def __post_init__(self) -> None:
        if type(self.handle) is not SequenceHandle:
            _fail("handle", "must be a SequenceHandle")
        _positive_int(self.token_budget, "token_budget")


def validate_sequence_batch(
    steps: tuple[SequenceStep, ...],
    *,
    max_sequences: int,
    max_tokens_per_step: int,
) -> tuple[SequenceStep, ...]:
    if type(steps) is not tuple or not steps:
        _fail("steps", "must be a non-empty tuple")
    _positive_int(max_sequences, "max_sequences")
    _positive_int(max_tokens_per_step, "max_tokens_per_step")
    if len(steps) > max_sequences:
        _fail("steps", "exceeds the backend sequence batch limit")
    handles: set[SequenceHandle] = set()
    for index, step in enumerate(steps):
        if type(step) is not SequenceStep:
            _fail(f"steps[{index}]", "must be a SequenceStep")
        if step.handle in handles:
            _fail("steps", "must not contain duplicate sequence handles")
        if step.token_budget > max_tokens_per_step:
            _fail(f"steps[{index}].token_budget", "exceeds the backend step limit")
        handles.add(step.handle)
    return steps


class PrefillStatus(str, Enum):
    PARTIAL = "partial"
    READY = "ready"


@dataclass(frozen=True, slots=True)
class PrefillOutcome:
    handle: SequenceHandle
    status: PrefillStatus
    processed_tokens: int
    remaining_tokens: int

    def __post_init__(self) -> None:
        if type(self.handle) is not SequenceHandle:
            _fail("handle", "must be a SequenceHandle")
        if type(self.status) is not PrefillStatus:
            _fail("status", "must be a PrefillStatus")
        processed = _non_negative_int(self.processed_tokens, "processed_tokens")
        remaining = _non_negative_int(self.remaining_tokens, "remaining_tokens")
        if self.status is PrefillStatus.PARTIAL:
            if processed == 0 or remaining == 0:
                _fail(
                    "status",
                    "partial prefill must make progress and retain input tokens",
                )
        elif remaining != 0:
            _fail("remaining_tokens", "ready prefill must have no remaining tokens")


class DecodeStatus(str, Enum):
    PROGRESSED = "progressed"
    FINISHED = "finished"
    FAILED = "failed"


class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SequenceCompletion:
    final_text: str
    prompt_tokens: int
    reasoning_tokens: int
    final_tokens: int
    sampled_tokens: int
    prompt_decode_ms: float
    generation_ms: float
    cached_prompt_tokens: int = 0
    cache_hit: bool = False
    first_sample_ms: float = 0.0
    first_final_ms: float = 0.0
    sample_itl_ms: tuple[float, ...] = ()
    final_itl_ms: tuple[float, ...] = ()
    grammar_engine: str = "unknown"
    cache_match: str = "unknown"
    session_id: str | None = None
    session_parent_generation: int | None = None
    session_generation: int | None = None
    session_copy_on_write: bool = False

    def __post_init__(self) -> None:
        if type(self.final_text) is not str:
            _fail("final_text", "must be a string")
        prompt = _non_negative_int(self.prompt_tokens, "prompt_tokens")
        reasoning = _non_negative_int(self.reasoning_tokens, "reasoning_tokens")
        final = _non_negative_int(self.final_tokens, "final_tokens")
        sampled = _non_negative_int(self.sampled_tokens, "sampled_tokens")
        cached = _non_negative_int(
            self.cached_prompt_tokens,
            "cached_prompt_tokens",
        )
        if reasoning + final > sampled:
            _fail("sampled_tokens", "must cover reasoning and final token counts")
        if prompt + sampled == 0:
            _fail("sampled_tokens", "completion must account at least one token")
        if cached > prompt:
            _fail("cached_prompt_tokens", "must not exceed prompt_tokens")
        if type(self.cache_hit) is not bool:
            _fail("cache_hit", "must be a bool")
        if type(self.session_copy_on_write) is not bool:
            _fail("session_copy_on_write", "must be a bool")
        if (
            type(self.grammar_engine) is not str
            or not self.grammar_engine
            or len(self.grammar_engine) > 64
        ):
            _fail("grammar_engine", "must be a non-empty bounded string")
        if self.cache_match not in {"unknown", "none", "exact", "prefix", "session"}:
            _fail("cache_match", "must identify a supported cache match kind")
        if self.cache_match in {"exact", "prefix", "session"} and not self.cache_hit:
            _fail("cache_match", "a cache match requires cache_hit")
        if self.cache_match == "session" and (
            self.session_id is None or self.session_parent_generation is None
        ):
            _fail(
                "cache_match",
                "a session match requires a session id and parent generation",
            )
        if self.session_id is None:
            if (
                self.session_parent_generation is not None
                or self.session_generation is not None
                or self.session_copy_on_write
            ):
                _fail("session_id", "is required for session cache metadata")
        else:
            _bounded_text(self.session_id, "session_id", MAX_IDENTIFIER_BYTES)
            if self.session_parent_generation is not None:
                _positive_int(
                    self.session_parent_generation,
                    "session_parent_generation",
                )
            if self.session_generation is not None:
                _positive_int(self.session_generation, "session_generation")
            if self.session_copy_on_write and (
                self.session_parent_generation is None
                or self.session_generation is None
            ):
                _fail(
                    "session_copy_on_write",
                    "requires parent and committed generations",
                )
        if self.cache_hit != (cached > 0):
            _fail("cache_hit", "must match whether cached prompt tokens were reused")
        for name in (
            "prompt_decode_ms",
            "generation_ms",
            "first_sample_ms",
            "first_final_ms",
        ):
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                _fail(name, "must be a finite non-negative number")
            object.__setattr__(self, name, float(value))
        for name, maximum in (
            ("sample_itl_ms", max(0, sampled - 1)),
            ("final_itl_ms", max(0, final - 1)),
        ):
            values = getattr(self, name)
            if type(values) is not tuple or len(values) > maximum:
                _fail(name, "must be a bounded tuple of token intervals")
            normalized = []
            for index, value in enumerate(values):
                if (
                    type(value) not in {int, float}
                    or not math.isfinite(value)
                    or value < 0
                ):
                    _fail(f"{name}[{index}]", "must be finite and non-negative")
                normalized.append(float(value))
            object.__setattr__(self, name, tuple(normalized))
        if sampled > 0 and self.first_sample_ms == 0 and self.sample_itl_ms:
            _fail("first_sample_ms", "must be positive when token timings are present")
        if final > 0 and self.first_final_ms == 0 and self.final_itl_ms:
            _fail("first_final_ms", "must be positive when final timings are present")


@dataclass(frozen=True, slots=True)
class DecodeOutcome:
    handle: SequenceHandle
    status: DecodeStatus
    token_ids: tuple[int, ...]
    text_delta: str
    finish_reason: FinishReason | None = None
    completion: SequenceCompletion | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.handle) is not SequenceHandle:
            _fail("handle", "must be a SequenceHandle")
        if type(self.status) is not DecodeStatus:
            _fail("status", "must be a DecodeStatus")
        if type(self.token_ids) is not tuple:
            _fail("token_ids", "must be a tuple")
        for index, token_id in enumerate(self.token_ids):
            _non_negative_int(token_id, f"token_ids[{index}]")
        if type(self.text_delta) is not str:
            _fail("text_delta", "must be a string")
        if self.finish_reason is not None and type(self.finish_reason) is not FinishReason:
            _fail("finish_reason", "must be a FinishReason or None")
        if self.completion is not None and type(self.completion) is not SequenceCompletion:
            _fail("completion", "must be a SequenceCompletion or None")
        if self.status is DecodeStatus.PROGRESSED:
            if not self.token_ids:
                _fail("token_ids", "progressed decode must produce at least one token")
            if self.finish_reason is not None:
                _fail("finish_reason", "progressed decode must not have a finish reason")
            if self.completion is not None or self.error_code is not None:
                _fail("status", "progressed decode cannot carry completion or error")
        elif self.status is DecodeStatus.FINISHED:
            if self.finish_reason is None:
                _fail("finish_reason", "finished decode requires a finish reason")
            if self.completion is None:
                _fail("completion", "finished decode requires completion statistics")
            if self.error_code is not None:
                _fail("error_code", "finished decode cannot carry an error")
        else:
            if self.finish_reason is not None or self.completion is not None:
                _fail("status", "failed decode cannot carry a successful completion")
            _bounded_text(self.error_code, "error_code", MAX_ERROR_CODE_BYTES)


class ReleaseStatus(str, Enum):
    RELEASED = "released"
    ALREADY_RELEASED = "already_released"
    STALE_HANDLE = "stale_handle"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class ReleaseOutcome:
    handle: SequenceHandle
    status: ReleaseStatus
    released_bytes: int

    def __post_init__(self) -> None:
        if type(self.handle) is not SequenceHandle:
            _fail("handle", "must be a SequenceHandle")
        if type(self.status) is not ReleaseStatus:
            _fail("status", "must be a ReleaseStatus")
        released_bytes = _non_negative_int(self.released_bytes, "released_bytes")
        if self.status is not ReleaseStatus.RELEASED and released_bytes != 0:
            _fail(
                "released_bytes",
                "non-released outcomes must report zero released bytes",
            )


class SchedulerEventKind(str, Enum):
    ADMITTED = "admitted"
    SEQUENCE_OPENED = "sequence_opened"
    PREFILL_COMPLETED = "prefill_completed"
    DECODE_COMPLETED = "decode_completed"
    SEQUENCE_RELEASED = "sequence_released"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_FAILED = "request_failed"


_SEQUENCE_EVENT_KINDS = frozenset({
    SchedulerEventKind.SEQUENCE_OPENED,
    SchedulerEventKind.PREFILL_COMPLETED,
    SchedulerEventKind.DECODE_COMPLETED,
    SchedulerEventKind.SEQUENCE_RELEASED,
})
_TOKEN_EVENT_KINDS = frozenset({
    SchedulerEventKind.PREFILL_COMPLETED,
    SchedulerEventKind.DECODE_COMPLETED,
})


@dataclass(frozen=True, slots=True)
class SchedulerEvent:
    kind: SchedulerEventKind
    request_id: str
    at_monotonic: float
    handle: SequenceHandle | None = None
    tokens: int | None = None
    error_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not SchedulerEventKind:
            _fail("kind", "must be a SchedulerEventKind")
        _bounded_text(self.request_id, "request_id", MAX_IDENTIFIER_BYTES)
        object.__setattr__(
            self,
            "at_monotonic",
            _monotonic_time(self.at_monotonic, "at_monotonic"),
        )
        if self.handle is not None and type(self.handle) is not SequenceHandle:
            _fail("handle", "must be a SequenceHandle or None")
        if self.kind in _SEQUENCE_EVENT_KINDS and self.handle is None:
            _fail("handle", "sequence events require a SequenceHandle")
        if self.kind in _TOKEN_EVENT_KINDS:
            if self.tokens is None:
                _fail("tokens", "prefill/decode events require a token count")
            _non_negative_int(self.tokens, "tokens")
        elif self.tokens is not None:
            _fail("tokens", "this event kind must not carry a token count")
        if self.kind is SchedulerEventKind.REQUEST_FAILED:
            _bounded_text(self.error_code, "error_code", MAX_ERROR_CODE_BYTES)
        elif self.error_code is not None:
            _fail("error_code", "only request_failed events may carry an error code")
