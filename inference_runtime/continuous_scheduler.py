from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from model_worker.contracts import GenerateResult
from model_worker.output_contract import validate_output
from model_worker.preflight import PreflightedRequest
from model_worker.strict_json import loads

from .contracts import (
    DecodeOutcome,
    DecodeStatus,
    PrefillOutcome,
    PrefillStatus,
    SchedulerEvent,
    SchedulerEventKind,
    SchedulingMetadata,
    SequenceCompletion,
    SequenceHandle,
    SequenceStep,
)
from .governance import (
    AdmissionLimits,
    AdmissionRejection,
    HierarchicalFairSelector,
    ResourceAdmissionController,
    RuntimeSchedulingPolicy,
)
from .ports import (
    BatchSteppableBackend,
    SchedulerEventSink,
    require_batch_steppable_backend,
)


class RuntimeRequestState(str, Enum):
    QUEUED = "queued"
    OPENING = "opening"
    PREFILL = "prefill"
    PREFILL_INFLIGHT = "prefill_inflight"
    DECODE = "decode"
    DECODE_INFLIGHT = "decode_inflight"
    RELEASING = "releasing"
    TERMINAL = "terminal"


class InferenceRuntimeError(RuntimeError):
    def __init__(self, code: str, detail: str, *, retryable: bool) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.retryable = retryable


@dataclass(slots=True)
class _RequestEntry:
    request: PreflightedRequest
    scheduling: SchedulingMetadata
    events: SchedulerEventSink
    attempt_id: str
    admitted_at: float
    queue_deadline: float
    order: int
    estimated_tokens: int
    state: RuntimeRequestState = RuntimeRequestState.QUEUED
    handle: SequenceHandle | None = None
    execution_deadline: float | None = None
    opened_at: float | None = None
    cancel_requested: bool = False
    result: GenerateResult | None = None
    error: InferenceRuntimeError | None = None
    terminal_at: float | None = None
    admitted_event_sent: bool = False
    terminalizing: bool = False
    resource_release_confirmed: bool = False
    final_deltas: list[str] = field(default_factory=list)


class ContinuousBatchScheduler:
    """Threaded control plane with chunked prefill and decode microbatches.

    The backend owns llama.cpp compute only. This scheduler owns request
    deadlines, sequence admission, queue state, decode-first policy and release.
    """

    def __init__(
        self,
        backend: object,
        *,
        tick_token_budget: int,
        max_pending_requests: int = 1024,
        max_decode_burst: int = 8,
        scheduling_policy: RuntimeSchedulingPolicy | None = None,
        admission_limits: AdmissionLimits | None = None,
        autostart: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.backend: BatchSteppableBackend[PreflightedRequest] = (
            require_batch_steppable_backend(backend)
        )
        if type(tick_token_budget) is not int or tick_token_budget <= 0:
            raise ValueError("tick_token_budget must be a positive integer")
        if type(max_pending_requests) is not int or max_pending_requests <= 0:
            raise ValueError("max_pending_requests must be a positive integer")
        if type(max_decode_burst) is not int or max_decode_burst <= 0:
            raise ValueError("max_decode_burst must be a positive integer")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if type(autostart) is not bool:
            raise TypeError("autostart must be a bool")
        capabilities = self.backend.capabilities
        if capabilities.max_sequences_per_step is None:
            raise ValueError("batch backend omitted max_sequences_per_step")
        if capabilities.max_prefill_tokens_per_step is None:
            raise ValueError("batch backend omitted max_prefill_tokens_per_step")
        if capabilities.max_decode_tokens_per_step is None:
            raise ValueError("batch backend omitted max_decode_tokens_per_step")
        if tick_token_budget < capabilities.max_sequences_per_step:
            raise ValueError("tick budget must cover progress for the maximum decode batch")
        self.tick_token_budget = tick_token_budget
        self.max_pending_requests = max_pending_requests
        self.max_decode_burst = max_decode_burst
        self._clock = clock
        self.scheduling_policy = scheduling_policy or RuntimeSchedulingPolicy.defaults()
        self._selector = HierarchicalFairSelector(self.scheduling_policy)
        if admission_limits is None:
            max_sequences = capabilities.max_concurrent_sequences or 1
            runtime_manifest = getattr(self.backend, "runtime_manifest", None)
            scheduler_config = getattr(runtime_manifest, "scheduler", None)
            kv_tokens = int(
                getattr(
                    scheduler_config,
                    "kv_tokens",
                    capabilities.max_context_tokens * max_sequences,
                )
            )
            agent_sequences = max(1, max_sequences // 2)
            admission_limits = AdmissionLimits(
                max_pending=max_pending_requests,
                max_sequences=max_sequences,
                kv_token_budget=kv_tokens,
                max_pending_per_workflow=max_pending_requests,
                max_pending_per_agent=max(1, max_pending_requests // 2),
                max_sequences_per_workflow=max_sequences,
                max_sequences_per_agent=agent_sequences,
                max_kv_tokens_per_workflow=kv_tokens,
                max_kv_tokens_per_agent=min(
                    kv_tokens,
                    capabilities.max_context_tokens * agent_sequences,
                ),
                load_shed_pending_threshold=max(1, (max_pending_requests * 3) // 4),
                load_shed_min_priority=0,
            )
        self._admission = ResourceAdmissionController(
            admission_limits,
            self.scheduling_policy,
        )
        self._condition = threading.Condition()
        self._entries: dict[str, _RequestEntry] = {}
        self._queued: deque[str] = deque()
        self._prefill: deque[str] = deque()
        self._decode: deque[str] = deque()
        self._decode_burst = 0
        self._stopping = False
        self._stopped = False
        self._fatal_error: InferenceRuntimeError | None = None
        self._event_failures = 0
        self._next_order = 0
        self._started = False
        self._thread = threading.Thread(
            target=self._run,
            name="inference-continuous-batch-scheduler",
            daemon=True,
        )
        if autostart:
            self.start()

    def start(self) -> bool:
        with self._condition:
            if self._started:
                return False
            if self._stopping:
                raise RuntimeError("cannot start a draining scheduler")
            self._started = True
            self._thread.start()
            return True

    @property
    def capabilities(self):
        """Expose the routed backend contract without leaking step operations."""

        return self.backend.capabilities

    @property
    def event_failures(self) -> int:
        with self._condition:
            return self._event_failures

    @property
    def active_requests(self) -> int:
        with self._condition:
            return sum(
                entry.state is not RuntimeRequestState.TERMINAL
                for entry in self._entries.values()
            )

    @property
    def admission_snapshot(self):
        return self._admission.snapshot()

    @property
    def fairness_snapshot(self):
        return self._selector.snapshot()

    def infer(
        self,
        request: PreflightedRequest,
        *,
        scheduling: SchedulingMetadata,
        events: SchedulerEventSink,
    ) -> GenerateResult:
        if type(request) is not PreflightedRequest:
            raise TypeError("continuous scheduler requires PreflightedRequest")
        if type(scheduling) is not SchedulingMetadata:
            raise TypeError("scheduling must be SchedulingMetadata")
        if not isinstance(events, SchedulerEventSink):
            raise TypeError("events must implement SchedulerEventSink")
        now = self._now()
        queue_deadline = now + request.limits.queue_timeout_ms / 1000
        if scheduling.deadline_monotonic is not None:
            queue_deadline = min(queue_deadline, scheduling.deadline_monotonic)
        if queue_deadline <= now:
            raise InferenceRuntimeError(
                "deadline_exceeded",
                "request deadline elapsed before admission",
                retryable=True,
            )
        entry = _RequestEntry(
            request=request,
            scheduling=scheduling,
            events=events,
            attempt_id=uuid.uuid4().hex,
            admitted_at=now,
            queue_deadline=queue_deadline,
            order=-1,
            estimated_tokens=self._estimate_reservation(request),
        )
        with self._condition:
            if self._stopped:
                raise self._fatal_error or InferenceRuntimeError(
                    "shutdown",
                    "inference scheduler is stopped",
                    retryable=True,
                )
            if self._stopping:
                raise InferenceRuntimeError(
                    "shutdown",
                    "inference scheduler is draining",
                    retryable=True,
                )
            if scheduling.request_id in self._entries:
                raise InferenceRuntimeError(
                    "duplicate_request",
                    "request_id is already registered",
                    retryable=False,
                )
            nonterminal = sum(
                current.state is not RuntimeRequestState.TERMINAL
                for current in self._entries.values()
            )
            if nonterminal >= self.max_pending_requests:
                raise InferenceRuntimeError(
                    "queue_full",
                    "inference scheduler queue is full",
                    retryable=True,
                )
            try:
                self._admission.admit(scheduling, entry.estimated_tokens)
            except AdmissionRejection as exc:
                raise InferenceRuntimeError(
                    exc.code,
                    exc.detail,
                    retryable=exc.retryable,
                ) from exc
            entry.order = self._next_order
            self._next_order += 1
            self._entries[scheduling.request_id] = entry
            self._queued.append(scheduling.request_id)
            self._condition.notify_all()
            while entry.state is not RuntimeRequestState.TERMINAL:
                self._condition.wait()
            result = entry.result
            error = entry.error
            self._entries.pop(scheduling.request_id, None)
        if error is not None:
            raise error
        if result is None:
            raise InferenceRuntimeError(
                "scheduler_invariant",
                "terminal request has no result or error",
                retryable=False,
            )
        return result

    def _estimate_reservation(self, request: PreflightedRequest) -> int:
        prompt_bytes = sum(
            len(message.content.encode("utf-8")) + len(message.role.encode("utf-8"))
            for message in request.model_messages
        )
        estimate = prompt_bytes + request.limits.total_tokens + 64
        return min(self.backend.capabilities.max_context_tokens, max(1, estimate))

    def cancel(self, request_id: str) -> bool:
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or entry.state is RuntimeRequestState.TERMINAL:
                return False
            entry.cancel_requested = True
            self._condition.notify_all()
            return True

    def _now(self) -> float:
        value = float(self._clock())
        if value < 0 or value == float("inf") or value != value:
            raise RuntimeError("scheduler clock must be finite and non-negative")
        return value

    def _run(self) -> None:
        try:
            while True:
                if self._process_terminal_boundaries():
                    continue
                if self._open_one():
                    continue
                action = self._choose_batch_action()
                if action == "decode":
                    self._run_decode_batch()
                    continue
                if action == "prefill":
                    self._run_prefill_batch()
                    continue
                with self._condition:
                    if self._stopping and not self._has_nonterminal_locked():
                        self._stopped = True
                        self._condition.notify_all()
                        return
                    # A producer may enqueue/cancel after the action checks above
                    # but before this block acquires the condition. Re-check the
                    # predicate while holding the same lock used by producers so
                    # their notification cannot be lost immediately before wait().
                    if self._has_immediate_work_locked():
                        continue
                    timeout = self._next_deadline_wait_locked()
                    self._condition.wait(timeout)
        except BaseException as exc:
            error = InferenceRuntimeError(
                "scheduler_crashed",
                (
                    f"scheduler loop failed: {type(exc).__name__}: "
                    f"{str(exc)[:384]}"
                ),
                retryable=True,
            )
            with self._condition:
                self._fatal_error = error
            self._fail_all(error)
            with self._condition:
                self._stopped = True
                self._condition.notify_all()

    def _has_nonterminal_locked(self) -> bool:
        return any(
            entry.state is not RuntimeRequestState.TERMINAL
            for entry in self._entries.values()
        )

    def _has_immediate_work_locked(self) -> bool:
        """Return whether the worker must loop instead of entering wait().

        Callers hold ``self._condition``. This is deliberately a predicate
        re-check, not a second scheduling decision, so a producer notification
        cannot be lost in the transition to the idle wait.
        """

        now = self._now()
        for entry in self._entries.values():
            if entry.state is RuntimeRequestState.TERMINAL:
                continue
            if self._stopping or entry.cancel_requested:
                return True
            deadline = (
                entry.queue_deadline
                if entry.handle is None
                else entry.execution_deadline
            )
            if deadline is not None and deadline <= now:
                return True
        if self._queue_has_state_locked(
            self._prefill,
            RuntimeRequestState.PREFILL,
        ) or self._queue_has_state_locked(
            self._decode,
            RuntimeRequestState.DECODE,
        ):
            return True
        return any(
            entry is not None
            and entry.state is RuntimeRequestState.QUEUED
            and self._admission.can_activate(request_id)
            for request_id in self._queued
            for entry in (self._entries.get(request_id),)
        )

    def _process_terminal_boundaries(self) -> bool:
        target: _RequestEntry | None = None
        error: InferenceRuntimeError | None = None
        now = self._now()
        with self._condition:
            for entry in self._entries.values():
                if entry.state is RuntimeRequestState.TERMINAL:
                    continue
                if self._stopping or entry.cancel_requested:
                    target = entry
                    error = InferenceRuntimeError(
                        "cancelled" if not self._stopping else "shutdown",
                        "request cancelled" if not self._stopping else "scheduler shutdown",
                        retryable=self._stopping,
                    )
                    break
                deadline = (
                    entry.queue_deadline
                    if entry.handle is None
                    else entry.execution_deadline
                )
                if deadline is not None and deadline <= now:
                    target = entry
                    error = InferenceRuntimeError(
                        "queue_timeout" if entry.handle is None else "deadline_exceeded",
                        "queue deadline exceeded"
                        if entry.handle is None
                        else "execution deadline exceeded",
                        retryable=True,
                    )
                    break
            if target is None:
                return False
            target.state = RuntimeRequestState.RELEASING
        self._release_and_terminalize(target, error=error)
        return True

    def _open_one(self) -> bool:
        with self._condition:
            capabilities = self.backend.capabilities
            active_sequences = sum(
                entry.handle is not None
                and entry.state is not RuntimeRequestState.TERMINAL
                for entry in self._entries.values()
            )
            if active_sequences >= (capabilities.max_concurrent_sequences or 0):
                return False
            entry = self._pop_state_locked(
                self._queued,
                RuntimeRequestState.QUEUED,
                predicate=lambda candidate: self._admission.can_activate(
                    candidate.scheduling.request_id
                ),
            )
            if entry is None:
                return False
            entry.state = RuntimeRequestState.OPENING
        if not entry.admitted_event_sent:
            self._publish(entry, SchedulerEventKind.ADMITTED)
            entry.admitted_event_sent = True
        try:
            handle = self.backend.open_sequence(
                entry.request,
                scheduling=entry.scheduling,
                events=entry.events,
            )
        except Exception as exc:
            self._terminalize(
                entry,
                error=self._backend_error("open_sequence_failed", exc),
            )
            return True
        try:
            reservation_reader = getattr(self.backend, "reservation_tokens", None)
            reserved_tokens = (
                reservation_reader(handle) if callable(reservation_reader) else None
            )
            self._admission.activate(
                entry.scheduling.request_id,
                reserved_tokens,
            )
        except Exception as exc:
            with self._condition:
                entry.handle = handle
                entry.state = RuntimeRequestState.RELEASING
                self._condition.notify_all()
            self._publish(entry, SchedulerEventKind.SEQUENCE_OPENED, handle=handle)
            self._release_and_terminalize(
                entry,
                error=self._backend_error("resource_admission_failed", exc),
            )
            return True
        self._selector.charge(entry.scheduling, 1)
        with self._condition:
            entry.handle = handle
            now = self._now()
            entry.opened_at = now
            execution_deadline = now + entry.request.limits.execution_timeout_ms / 1000
            if entry.scheduling.deadline_monotonic is not None:
                execution_deadline = min(
                    execution_deadline,
                    entry.scheduling.deadline_monotonic,
                )
            entry.execution_deadline = execution_deadline
            if entry.cancel_requested or self._stopping or execution_deadline <= now:
                entry.state = RuntimeRequestState.RELEASING
                release_immediately = True
            else:
                entry.state = RuntimeRequestState.PREFILL
                self._prefill.append(entry.scheduling.request_id)
                release_immediately = False
            self._condition.notify_all()
        self._publish(entry, SchedulerEventKind.SEQUENCE_OPENED, handle=handle)
        if release_immediately:
            self._release_and_terminalize(
                entry,
                error=InferenceRuntimeError(
                    "cancelled" if entry.cancel_requested else "deadline_exceeded",
                    "request cancelled during sequence admission"
                    if entry.cancel_requested
                    else "execution deadline exceeded during sequence admission",
                    retryable=not entry.cancel_requested,
                ),
            )
        return True

    def _choose_batch_action(self) -> str | None:
        with self._condition:
            has_decode = self._queue_has_state_locked(
                self._decode,
                RuntimeRequestState.DECODE,
            )
            has_prefill = self._queue_has_state_locked(
                self._prefill,
                RuntimeRequestState.PREFILL,
            )
            if has_decode and (not has_prefill or self._decode_burst < self.max_decode_burst):
                return "decode"
            if has_prefill:
                return "prefill"
            if has_decode:
                return "decode"
            return None

    def _run_prefill_batch(self) -> None:
        capabilities = self.backend.capabilities
        assert capabilities.max_sequences_per_step is not None
        assert capabilities.max_prefill_tokens_per_step is not None
        selected: list[_RequestEntry] = []
        steps: list[SequenceStep] = []
        with self._condition:
            candidates = self._take_state_locked(
                self._prefill,
                RuntimeRequestState.PREFILL,
                capabilities.max_sequences_per_step,
            )
            remaining_budget = self.tick_token_budget
            for index, entry in enumerate(candidates):
                if entry.handle is None:
                    continue
                slots_left = max(1, len(candidates) - index)
                budget = min(
                    capabilities.max_prefill_tokens_per_step,
                    max(1, remaining_budget // slots_left),
                )
                remaining_budget -= budget
                entry.state = RuntimeRequestState.PREFILL_INFLIGHT
                selected.append(entry)
                steps.append(SequenceStep(entry.handle, budget))
        if not selected:
            return
        try:
            outcomes = self.backend.prefill_batch(tuple(steps), events=selected[0].events)
            if len(outcomes) != len(selected):
                raise RuntimeError("prefill outcome cardinality mismatch")
        except Exception as exc:
            error = self._backend_error("prefill_batch_failed", exc)
            for entry in selected:
                self._release_and_terminalize(entry, error=error)
            return
        event_outcomes: list[tuple[_RequestEntry, PrefillOutcome]] = []
        with self._condition:
            for entry, outcome in zip(selected, outcomes, strict=True):
                if outcome.handle != entry.handle:
                    error = InferenceRuntimeError(
                        "protocol_violation",
                        "prefill outcome handle mismatch",
                        retryable=True,
                    )
                    entry.state = RuntimeRequestState.RELEASING
                    self._condition.notify_all()
                    # Release outside this lock below.
                    entry.error = error
                    continue
                event_outcomes.append((entry, outcome))
                if outcome.status is PrefillStatus.READY:
                    entry.state = RuntimeRequestState.DECODE
                    self._decode.append(entry.scheduling.request_id)
                else:
                    entry.state = RuntimeRequestState.PREFILL
                    self._prefill.append(entry.scheduling.request_id)
            self._decode_burst = 0
            self._condition.notify_all()
        for entry, outcome in event_outcomes:
            if outcome.processed_tokens:
                self._selector.charge(entry.scheduling, outcome.processed_tokens)
            self._publish(
                entry,
                SchedulerEventKind.PREFILL_COMPLETED,
                handle=entry.handle,
                tokens=outcome.processed_tokens,
            )
        for entry in selected:
            if entry.state is RuntimeRequestState.RELEASING:
                self._release_and_terminalize(entry, error=entry.error)

    def _run_decode_batch(self) -> None:
        capabilities = self.backend.capabilities
        assert capabilities.max_sequences_per_step is not None
        with self._condition:
            selected = self._take_state_locked(
                self._decode,
                RuntimeRequestState.DECODE,
                min(capabilities.max_sequences_per_step, self.tick_token_budget),
            )
            steps = []
            quantum = min(
                capabilities.max_decode_tokens_per_step,
                max(1, self.tick_token_budget // max(1, len(selected))),
            )
            for entry in selected:
                if entry.handle is None:
                    continue
                entry.state = RuntimeRequestState.DECODE_INFLIGHT
                steps.append(SequenceStep(entry.handle, quantum))
        if not steps:
            return
        try:
            outcomes = self.backend.decode_batch(tuple(steps), events=selected[0].events)
            if len(outcomes) != len(selected):
                raise RuntimeError("decode outcome cardinality mismatch")
        except Exception as exc:
            error = self._backend_error("decode_batch_failed", exc)
            for entry in selected:
                self._release_and_terminalize(entry, error=error)
            return

        terminals: list[tuple[_RequestEntry, DecodeOutcome]] = []
        event_outcomes: list[tuple[_RequestEntry, DecodeOutcome]] = []
        with self._condition:
            for entry, outcome in zip(selected, outcomes, strict=True):
                if outcome.handle != entry.handle:
                    outcome = DecodeOutcome(
                        entry.handle,
                        DecodeStatus.FAILED,
                        (),
                        "",
                        error_code="protocol_violation",
                    )
                event_outcomes.append((entry, outcome))
                if outcome.text_delta:
                    entry.final_deltas.append(outcome.text_delta)
                if outcome.status is DecodeStatus.PROGRESSED:
                    entry.state = RuntimeRequestState.DECODE
                    self._decode.append(entry.scheduling.request_id)
                else:
                    entry.state = RuntimeRequestState.RELEASING
                    terminals.append((entry, outcome))
            self._decode_burst += 1
            self._condition.notify_all()
        for entry, outcome in event_outcomes:
            if outcome.token_ids:
                self._selector.charge(entry.scheduling, len(outcome.token_ids))
            self._publish(
                entry,
                SchedulerEventKind.DECODE_COMPLETED,
                handle=entry.handle,
                tokens=len(outcome.token_ids),
            )
        for entry, outcome in terminals:
            if outcome.status is DecodeStatus.FAILED:
                self._release_and_terminalize(
                    entry,
                    error=InferenceRuntimeError(
                        outcome.error_code or "decode_failed",
                        "native sequence decode failed",
                        retryable=bool(
                            (outcome.error_code or "").startswith("backend_")
                        ),
                    ),
                )
            else:
                self._complete_entry(entry, outcome.completion)

    def _complete_entry(
        self,
        entry: _RequestEntry,
        completion: SequenceCompletion | None,
    ) -> None:
        if completion is None:
            self._release_and_terminalize(
                entry,
                error=InferenceRuntimeError(
                    "protocol_violation",
                    "finished sequence omitted completion data",
                    retryable=True,
                ),
            )
            return
        if "".join(entry.final_deltas) != completion.final_text:
            self._release_and_terminalize(
                entry,
                error=InferenceRuntimeError(
                    "protocol_violation",
                    "decode deltas do not match final completion text",
                    retryable=True,
                ),
            )
            return
        try:
            output = loads(completion.final_text)
            validation_errors = validate_output(output, entry.request.contract)
            if validation_errors:
                raise ValueError("structured output contract violation")
        except Exception as exc:
            self._release_and_terminalize(
                entry,
                error=InferenceRuntimeError(
                    "output_invalid",
                    f"native final output is invalid: {type(exc).__name__}",
                    retryable=False,
                ),
            )
            return
        now = self._now()
        queue_ms = max(
            0.0,
            (entry.opened_at or now) - entry.admitted_at,
        ) * 1000
        timing = {
            "queue_ms": queue_ms,
            "prompt_decode_ms": completion.prompt_decode_ms,
            "generation_ms": completion.generation_ms,
            "first_sample_ms": completion.first_sample_ms,
            "first_final_ms": completion.first_final_ms,
            "sample_itl_ms": list(completion.sample_itl_ms),
            "final_itl_ms": list(completion.final_itl_ms),
            "grammar_engine": completion.grammar_engine,
            "cache_match": completion.cache_match,
            "session_id": completion.session_id,
            "session_parent_generation": completion.session_parent_generation,
            "session_generation": completion.session_generation,
            "session_copy_on_write": completion.session_copy_on_write,
            "total_ms": max(0.0, now - entry.admitted_at) * 1000,
        }
        usage = {
            "prompt_tokens": completion.prompt_tokens,
            "reasoning_tokens": completion.reasoning_tokens,
            "final_tokens": completion.final_tokens,
            "sampled_tokens": completion.sampled_tokens,
            "cached_prompt_tokens": completion.cached_prompt_tokens,
            "cache_hit": completion.cache_hit,
            "cache_match": completion.cache_match,
            "session_generation": completion.session_generation,
            "session_copy_on_write": completion.session_copy_on_write,
            "context_limit": self.backend.capabilities.max_context_tokens,
            "context_headroom": max(
                0,
                self.backend.capabilities.max_context_tokens
                - completion.prompt_tokens
                - entry.request.limits.total_tokens,
            ),
        }
        identity = getattr(self.backend, "runtime_identity", {})
        model = {
            "id": entry.request.request.model_id,
            "backend": self.backend.capabilities.backend,
            **(dict(identity) if isinstance(identity, dict) else {}),
        }
        result = GenerateResult(
            entry.scheduling.request_id,
            entry.attempt_id,
            "completed",
            True,
            True,
            output,
            usage,
            timing,
            model,
        )
        self._release_and_terminalize(entry, result=result)

    def _release_and_terminalize(
        self,
        entry: _RequestEntry,
        *,
        result: GenerateResult | None = None,
        error: InferenceRuntimeError | None = None,
    ) -> None:
        handle = entry.handle
        if handle is not None:
            try:
                released = self.backend.release(handle, events=entry.events)
                if released.status.value not in {"released", "already_released"}:
                    raise RuntimeError(f"release returned {released.status.value}")
                entry.resource_release_confirmed = True
                self._publish(
                    entry,
                    SchedulerEventKind.SEQUENCE_RELEASED,
                    handle=handle,
                )
            except Exception as exc:
                result = None
                error = self._backend_error("backend_release_failed", exc)
        else:
            entry.resource_release_confirmed = True
        self._terminalize(entry, result=result, error=error)

    def _terminalize(
        self,
        entry: _RequestEntry,
        *,
        result: GenerateResult | None = None,
        error: InferenceRuntimeError | None = None,
    ) -> bool:
        if not entry.admitted_event_sent:
            self._publish(entry, SchedulerEventKind.ADMITTED)
            entry.admitted_event_sent = True
        if entry.handle is None:
            entry.resource_release_confirmed = True
        with self._condition:
            if entry.state is RuntimeRequestState.TERMINAL or entry.terminalizing:
                return False
            if (result is None) == (error is None):
                error = InferenceRuntimeError(
                    "scheduler_invariant",
                    "terminalization requires exactly one result or error",
                    retryable=False,
                )
                result = None
            entry.terminalizing = True
        if entry.resource_release_confirmed:
            self._admission.release(entry.scheduling.request_id)
        if result is not None:
            self._publish(entry, SchedulerEventKind.REQUEST_COMPLETED)
        else:
            self._publish(
                entry,
                SchedulerEventKind.REQUEST_FAILED,
                error_code=(error.code if error is not None else "scheduler_invariant")[:128],
            )
        with self._condition:
            entry.state = RuntimeRequestState.TERMINAL
            entry.result = result
            entry.error = error
            entry.terminal_at = self._now()
            other_entries = [
                current
                for current in self._entries.values()
                if current is not entry
                and current.state is not RuntimeRequestState.TERMINAL
            ]
            same_agent = any(
                current.scheduling.workflow_id == entry.scheduling.workflow_id
                and current.scheduling.agent_id == entry.scheduling.agent_id
                for current in other_entries
            )
            same_workflow = any(
                current.scheduling.workflow_id == entry.scheduling.workflow_id
                for current in other_entries
            )
            self._selector.forget(
                entry.scheduling,
                drop_agent=not same_agent,
                drop_workflow=not same_workflow,
            )
            self._condition.notify_all()
        return True

    def _publish(
        self,
        entry: _RequestEntry,
        kind: SchedulerEventKind,
        *,
        handle: SequenceHandle | None = None,
        tokens: int | None = None,
        error_code: str | None = None,
    ) -> None:
        try:
            entry.events.publish(
                SchedulerEvent(
                    kind,
                    entry.scheduling.request_id,
                    self._now(),
                    handle=handle,
                    tokens=tokens,
                    error_code=error_code,
                )
            )
        except Exception:
            with self._condition:
                self._event_failures += 1

    @staticmethod
    def _backend_error(code: str, exc: BaseException) -> InferenceRuntimeError:
        detail = getattr(exc, "detail", str(exc) or type(exc).__name__)
        native_code = getattr(exc, "code", None)
        return InferenceRuntimeError(
            str(native_code or code)[:128],
            str(detail),
            retryable=True,
        )

    def _pop_state_locked(
        self,
        queue: deque[str],
        state: RuntimeRequestState,
        *,
        predicate: Callable[[_RequestEntry], bool] | None = None,
    ) -> _RequestEntry | None:
        valid_ids = []
        candidates = []
        for request_id in queue:
            entry = self._entries.get(request_id)
            if entry is None or entry.state is not state:
                continue
            valid_ids.append(request_id)
            if predicate is None or predicate(entry):
                candidates.append(entry)
        queue.clear()
        queue.extend(valid_ids)
        selected = self._selector.select(candidates, self._now())
        if selected is None:
            return None
        queue.remove(selected.scheduling.request_id)
        return selected

    def _take_state_locked(
        self,
        queue: deque[str],
        state: RuntimeRequestState,
        limit: int,
    ) -> list[_RequestEntry]:
        result = []
        while len(result) < limit:
            entry = self._pop_state_locked(queue, state)
            if entry is None:
                break
            result.append(entry)
        return result

    def _queue_has_state_locked(
        self,
        queue: deque[str],
        state: RuntimeRequestState,
    ) -> bool:
        while queue:
            entry = self._entries.get(queue[0])
            if entry is not None and entry.state is state:
                return True
            queue.popleft()
        return False

    def _next_deadline_wait_locked(self) -> float | None:
        now = self._now()
        deadlines = []
        for entry in self._entries.values():
            if entry.state is RuntimeRequestState.TERMINAL:
                continue
            deadline = entry.queue_deadline if entry.handle is None else entry.execution_deadline
            if deadline is not None:
                deadlines.append(deadline)
        if not deadlines:
            return None
        return max(0.0, min(deadlines) - now)

    def _fail_all(self, error: InferenceRuntimeError) -> None:
        with self._condition:
            entries = [
                entry
                for entry in self._entries.values()
                if entry.state is not RuntimeRequestState.TERMINAL
            ]
        for entry in entries:
            self._release_and_terminalize(entry, error=error)

    def shutdown(self, timeout: float = 10.0) -> bool:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        with self._condition:
            self._stopping = True
            for entry in self._entries.values():
                if entry.state is not RuntimeRequestState.TERMINAL:
                    entry.cancel_requested = True
            self._condition.notify_all()
        if self._started:
            self._thread.join(timeout)
            stopped = not self._thread.is_alive()
        else:
            self._fail_all(
                InferenceRuntimeError(
                    "shutdown",
                    "scheduler shutdown before worker start",
                    retryable=True,
                )
            )
            with self._condition:
                self._stopped = True
                self._condition.notify_all()
            stopped = True
        shutdown = getattr(self.backend, "shutdown", None)
        if callable(shutdown):
            shutdown()
        return stopped
