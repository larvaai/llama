"""Deterministic, single-threaded reference scheduler for policy conformance tests.

This module deliberately does not implement production concurrency, batching, KV
allocation, or cache policy. One tick performs at most one backend lifecycle action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Protocol, runtime_checkable

from .contracts import (
    BackendCapabilities,
    DecodeOutcome,
    DecodeStatus,
    PrefillOutcome,
    PrefillStatus,
    ReleaseOutcome,
    ReleaseStatus,
    SchedulerEvent,
    SchedulerEventKind,
    SchedulingMetadata,
    SequenceHandle,
)
from .ports import SchedulerEventSink, SteppableBackend, require_steppable_backend


@runtime_checkable
class MonotonicClock(Protocol):
    def now(self) -> float: ...


def _finite_non_negative(value: Any, field: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return float(value)


class FakeMonotonicClock:
    def __init__(self, initial: float = 0.0) -> None:
        self._now = _finite_non_negative(initial, "initial")

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        delta = _finite_non_negative(seconds, "seconds")
        advanced = self._now + delta
        if not math.isfinite(advanced):
            raise ValueError("advanced clock value must remain finite")
        self._now = advanced
        return self._now


@dataclass(frozen=True, slots=True)
class SchedulerPolicy:
    """Reference scoring policy, independent of agent role semantics."""

    service_class_priorities: tuple[tuple[str, int], ...]
    aging_interval: float

    def __post_init__(self) -> None:
        if type(self.service_class_priorities) is not tuple or not self.service_class_priorities:
            raise ValueError("service_class_priorities must be a non-empty tuple")
        seen = set()
        for index, item in enumerate(self.service_class_priorities):
            if type(item) is not tuple or len(item) != 2:
                raise ValueError(f"service_class_priorities[{index}] must be a pair")
            name, priority = item
            if type(name) is not str or not name or any(ord(char) < 0x20 for char in name):
                raise ValueError(f"service_class_priorities[{index}] has an invalid class")
            if name in seen:
                raise ValueError("service class priorities must be unique")
            if type(priority) is not int or abs(priority) > 1_000_000:
                raise ValueError(f"service_class_priorities[{index}] has an invalid priority")
            seen.add(name)
        interval = _finite_non_negative(self.aging_interval, "aging_interval")
        if interval == 0:
            raise ValueError("aging_interval must be positive")
        object.__setattr__(self, "aging_interval", interval)

    def priority_for(self, service_class: str) -> int:
        for name, priority in self.service_class_priorities:
            if name == service_class:
                return priority
        raise AdmissionError("unknown_service_class", service_class)


class SimulatorLifecycle(str, Enum):
    ADMITTED = "admitted"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    TERMINAL = "terminal"


class SimulatorTermination(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    BACKEND_ERROR = "backend_error"


class TickAction(str, Enum):
    IDLE = "idle"
    OPEN_SEQUENCE = "open_sequence"
    PREFILL = "prefill"
    DECODE = "decode"
    BACKEND_FAILED = "backend_failed"


@dataclass(frozen=True, slots=True)
class TickOutcome:
    at_monotonic: float
    action: TickAction
    selected_request_id: str | None
    expired_request_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    request_id: str
    lifecycle: SimulatorLifecycle
    termination: SimulatorTermination | None
    handle: SequenceHandle | None
    service_steps: int
    release_status: ReleaseStatus | None
    admitted_at: float
    terminal_at: float | None


class AdmissionError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class SchedulerInvariantError(RuntimeError):
    pass


@dataclass(slots=True)
class _Entry:
    request: Any
    scheduling: SchedulingMetadata
    order: int
    admitted_at: float
    lifecycle: SimulatorLifecycle = SimulatorLifecycle.ADMITTED
    termination: SimulatorTermination | None = None
    handle: SequenceHandle | None = None
    service_steps: int = 0
    release_attempted: bool = False
    release_status: ReleaseStatus | None = None
    terminal_at: float | None = None


class DeterministicSchedulerSimulator:
    """Serial reference policy used to test scheduling invariants deterministically."""

    def __init__(
        self,
        backend: object,
        *,
        clock: MonotonicClock,
        events: SchedulerEventSink,
        policy: SchedulerPolicy,
    ) -> None:
        self._backend: SteppableBackend[Any] = require_steppable_backend(backend)
        if not isinstance(clock, MonotonicClock):
            raise TypeError("clock does not implement MonotonicClock")
        if not isinstance(events, SchedulerEventSink):
            raise TypeError("events does not implement SchedulerEventSink")
        if type(policy) is not SchedulerPolicy:
            raise TypeError("policy must be a validated SchedulerPolicy")
        _finite_non_negative(clock.now(), "clock.now()")
        self._clock = clock
        self._events = events
        self._policy = policy
        self._capabilities: BackendCapabilities = self._backend.capabilities
        self._entries: dict[str, _Entry] = {}
        self._next_order = 0

    def admit(self, request: Any, scheduling: SchedulingMetadata) -> RequestSnapshot:
        if type(scheduling) is not SchedulingMetadata:
            raise AdmissionError("invalid_metadata", "scheduling metadata was not validated")
        request_id = scheduling.request_id
        if request_id in self._entries:
            raise AdmissionError("duplicate_request", request_id)
        self._policy.priority_for(scheduling.service_class)
        now = self._now()
        if scheduling.deadline_monotonic is not None and scheduling.deadline_monotonic <= now:
            raise AdmissionError("deadline_elapsed", request_id)
        if len(self.active_request_ids) >= self._capabilities.max_concurrent_requests:
            raise AdmissionError("capacity", request_id)
        entry = _Entry(request, scheduling, self._next_order, now)
        self._next_order += 1
        self._entries[request_id] = entry
        self._publish(SchedulerEventKind.ADMITTED, entry)
        return self._snapshot(entry)

    def cancel(self, request_id: str) -> bool:
        entry = self._entries.get(request_id)
        if entry is None or entry.lifecycle is SimulatorLifecycle.TERMINAL:
            return False
        self._terminalize(entry, SimulatorTermination.CANCELLED, "cancelled")
        return True

    def tick(self) -> TickOutcome:
        now = self._now()
        expired = []
        for entry in sorted(self._active_entries(), key=lambda item: item.order):
            deadline = entry.scheduling.deadline_monotonic
            if deadline is not None and deadline <= now:
                self._terminalize(
                    entry,
                    SimulatorTermination.DEADLINE_EXCEEDED,
                    "deadline_exceeded",
                )
                expired.append(entry.scheduling.request_id)

        eligible = self._eligible_entries()
        if not eligible:
            return TickOutcome(now, TickAction.IDLE, None, tuple(expired))
        entry = min(eligible, key=lambda item: self._selection_key(item, now))
        action = self._step(entry)
        return TickOutcome(
            now,
            action,
            entry.scheduling.request_id,
            tuple(expired),
        )

    def snapshot(self, request_id: str) -> RequestSnapshot:
        try:
            entry = self._entries[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown request: {request_id}") from exc
        return self._snapshot(entry)

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.scheduling.request_id
            for entry in sorted(self._active_entries(), key=lambda item: item.order)
        )

    @property
    def terminal_request_ids(self) -> tuple[str, ...]:
        return tuple(
            entry.scheduling.request_id
            for entry in sorted(self._entries.values(), key=lambda item: item.order)
            if entry.lifecycle is SimulatorLifecycle.TERMINAL
        )

    def _now(self) -> float:
        return _finite_non_negative(self._clock.now(), "clock.now()")

    def _active_entries(self) -> list[_Entry]:
        return [
            entry
            for entry in self._entries.values()
            if entry.lifecycle is not SimulatorLifecycle.TERMINAL
        ]

    def _eligible_entries(self) -> list[_Entry]:
        active_handles = sum(
            entry.handle is not None and not entry.release_attempted
            for entry in self._active_entries()
        )
        sequence_capacity = self._capabilities.max_concurrent_sequences
        if sequence_capacity is None:
            raise SchedulerInvariantError("steppable backend omitted sequence capacity")
        return [
            entry
            for entry in self._active_entries()
            if entry.lifecycle is not SimulatorLifecycle.ADMITTED
            or active_handles < sequence_capacity
        ]

    def _selection_key(self, entry: _Entry, now: float) -> tuple[Any, ...]:
        base_priority = self._policy.priority_for(entry.scheduling.service_class)
        age_quanta = int((now - entry.admitted_at) / self._policy.aging_interval)
        effective_score = Fraction(base_priority + age_quanta, 1) - Fraction(
            entry.service_steps,
            entry.scheduling.weight,
        )
        deadline = entry.scheduling.deadline_monotonic
        return (
            -effective_score,
            math.inf if deadline is None else deadline,
            entry.order,
        )

    def _step(self, entry: _Entry) -> TickAction:
        try:
            if entry.lifecycle is SimulatorLifecycle.ADMITTED:
                action = self._open_sequence(entry)
            elif entry.lifecycle is SimulatorLifecycle.PREFILLING:
                action = self._prefill(entry)
            elif entry.lifecycle is SimulatorLifecycle.DECODING:
                action = self._decode(entry)
            else:
                raise SchedulerInvariantError("terminal entry was selected")
        except Exception:
            self._terminalize(
                entry,
                SimulatorTermination.BACKEND_ERROR,
                "backend_operation_failed",
            )
            return TickAction.BACKEND_FAILED
        entry.service_steps += 1
        return action

    def _open_sequence(self, entry: _Entry) -> TickAction:
        handle = self._backend.open_sequence(
            entry.request,
            scheduling=entry.scheduling,
            events=self._events,
        )
        if type(handle) is not SequenceHandle:
            raise SchedulerInvariantError("open_sequence returned an invalid handle")
        if handle.backend != self._capabilities.backend:
            raise SchedulerInvariantError("sequence handle backend does not match capabilities")
        if handle.model not in self._capabilities.models:
            raise SchedulerInvariantError("sequence handle model does not match capabilities")
        if any(
            other.handle == handle
            for other in self._active_entries()
            if other is not entry
        ):
            raise SchedulerInvariantError("backend reused an active sequence handle")
        entry.handle = handle
        entry.lifecycle = SimulatorLifecycle.PREFILLING
        self._publish(SchedulerEventKind.SEQUENCE_OPENED, entry)
        return TickAction.OPEN_SEQUENCE

    def _prefill(self, entry: _Entry) -> TickAction:
        handle = self._require_handle(entry)
        budget = self._capabilities.max_prefill_tokens_per_step
        if budget is None:
            raise SchedulerInvariantError("backend omitted prefill step limit")
        outcome = self._backend.prefill(
            handle,
            token_budget=budget,
            events=self._events,
        )
        if type(outcome) is not PrefillOutcome or outcome.handle != handle:
            raise SchedulerInvariantError("prefill returned a mismatched outcome")
        if outcome.processed_tokens > budget:
            raise SchedulerInvariantError("prefill exceeded its token budget")
        self._publish(
            SchedulerEventKind.PREFILL_COMPLETED,
            entry,
            tokens=outcome.processed_tokens,
        )
        if outcome.status is PrefillStatus.READY:
            entry.lifecycle = SimulatorLifecycle.DECODING
        return TickAction.PREFILL

    def _decode(self, entry: _Entry) -> TickAction:
        handle = self._require_handle(entry)
        budget = self._capabilities.max_decode_tokens_per_step
        if budget is None:
            raise SchedulerInvariantError("backend omitted decode step limit")
        outcome = self._backend.decode(
            handle,
            token_budget=budget,
            events=self._events,
        )
        if type(outcome) is not DecodeOutcome or outcome.handle != handle:
            raise SchedulerInvariantError("decode returned a mismatched outcome")
        if len(outcome.token_ids) > budget:
            raise SchedulerInvariantError("decode exceeded its token budget")
        self._publish(
            SchedulerEventKind.DECODE_COMPLETED,
            entry,
            tokens=len(outcome.token_ids),
        )
        if outcome.status is DecodeStatus.FINISHED:
            self._terminalize(entry, SimulatorTermination.COMPLETED, None)
        elif outcome.status is DecodeStatus.FAILED:
            self._terminalize(
                entry,
                SimulatorTermination.BACKEND_ERROR,
                outcome.error_code or "backend_decode_failed",
            )
        return TickAction.DECODE

    def _terminalize(
        self,
        entry: _Entry,
        termination: SimulatorTermination,
        error_code: str | None,
    ) -> bool:
        if entry.lifecycle is SimulatorLifecycle.TERMINAL:
            return False
        release_confirmed = self._release(entry)
        if entry.handle is not None and not release_confirmed:
            termination = SimulatorTermination.BACKEND_ERROR
            error_code = "backend_release_failed"
        entry.lifecycle = SimulatorLifecycle.TERMINAL
        entry.termination = termination
        entry.terminal_at = self._now()
        if termination is SimulatorTermination.COMPLETED:
            self._publish(SchedulerEventKind.REQUEST_COMPLETED, entry)
        else:
            self._publish(
                SchedulerEventKind.REQUEST_FAILED,
                entry,
                error_code=error_code or termination.value,
            )
        return True

    def _release(self, entry: _Entry) -> bool:
        if entry.handle is None:
            return True
        if entry.release_attempted:
            return entry.release_status in {
                ReleaseStatus.RELEASED,
                ReleaseStatus.ALREADY_RELEASED,
            }
        entry.release_attempted = True
        try:
            outcome = self._backend.release(entry.handle, events=self._events)
        except Exception:
            return False
        if type(outcome) is not ReleaseOutcome or outcome.handle != entry.handle:
            return False
        entry.release_status = outcome.status
        confirmed = outcome.status in {
            ReleaseStatus.RELEASED,
            ReleaseStatus.ALREADY_RELEASED,
        }
        if confirmed:
            self._publish(SchedulerEventKind.SEQUENCE_RELEASED, entry)
        return confirmed

    @staticmethod
    def _require_handle(entry: _Entry) -> SequenceHandle:
        if entry.handle is None:
            raise SchedulerInvariantError("sequence action requires a handle")
        return entry.handle

    def _publish(
        self,
        kind: SchedulerEventKind,
        entry: _Entry,
        *,
        tokens: int | None = None,
        error_code: str | None = None,
    ) -> None:
        self._events.publish(
            SchedulerEvent(
                kind,
                entry.scheduling.request_id,
                self._now(),
                handle=entry.handle,
                tokens=tokens,
                error_code=error_code,
            )
        )

    @staticmethod
    def _snapshot(entry: _Entry) -> RequestSnapshot:
        return RequestSnapshot(
            request_id=entry.scheduling.request_id,
            lifecycle=entry.lifecycle,
            termination=entry.termination,
            handle=entry.handle,
            service_steps=entry.service_steps,
            release_status=entry.release_status,
            admitted_at=entry.admitted_at,
            terminal_at=entry.terminal_at,
        )
