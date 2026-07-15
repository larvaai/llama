from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .events import EventSink, NullEventSink


class Lifecycle(str, Enum):
    RECEIVED = "RECEIVED"
    PREFLIGHTED = "PREFLIGHTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


TERMINAL = {Lifecycle.COMPLETED, Lifecycle.FAILED, Lifecycle.CANCELLED, Lifecycle.TIMED_OUT}
ALLOWED = {
    Lifecycle.RECEIVED: {Lifecycle.PREFLIGHTED, Lifecycle.FAILED},
    Lifecycle.PREFLIGHTED: {Lifecycle.QUEUED, Lifecycle.FAILED},
    Lifecycle.QUEUED: {Lifecycle.RUNNING, Lifecycle.CANCELLED, Lifecycle.TIMED_OUT, Lifecycle.FAILED},
    Lifecycle.RUNNING: {Lifecycle.COMPLETED, Lifecycle.FAILED, Lifecycle.CANCELLED, Lifecycle.TIMED_OUT},
}


@dataclass(slots=True)
class RequestRecord:
    request_id: str
    attempt_id: str
    request: Any
    queue_deadline: float
    execution_timeout: float
    lifecycle: Lifecycle = Lifecycle.RECEIVED
    timestamps: dict[str, float] = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    event_sink: EventSink = field(default_factory=NullEventSink)
    result: Any = None
    error: Any = None
    condition: threading.Condition = field(default_factory=threading.Condition)


TerminalObserver = Callable[[RequestRecord], None]


@dataclass(frozen=True, slots=True)
class RegistryPruneStats:
    scanned_records: int
    terminal_records: int
    eligible_records: int
    removed_records: int
    remaining_records: int
    eligible_remaining: int
    max_records: int
    limit_reached: bool


class RequestRegistry:
    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        terminal_observers: Iterable[TerminalObserver] = (),
    ) -> None:
        clock = time.monotonic if clock is None else clock
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._lock = threading.RLock()
        self._records: dict[str, RequestRecord] = {}
        self._clock = clock
        self._terminal_observers: list[TerminalObserver] = []
        self._terminal_observer_failures = 0
        for observer in terminal_observers:
            self.add_terminal_observer(observer)

    def _now(self) -> float:
        value = self._clock()
        if type(value) not in {int, float}:
            raise RuntimeError("registry clock must return a finite number")
        try:
            converted = float(value)
        except (OverflowError, ValueError):
            raise RuntimeError("registry clock must return a finite number") from None
        if not math.isfinite(converted):
            raise RuntimeError("registry clock must return a finite number")
        return converted

    @staticmethod
    def _same_observer(left: TerminalObserver, right: TerminalObserver) -> bool:
        if left is right:
            return True
        left_function = getattr(left, "__func__", None)
        right_function = getattr(right, "__func__", None)
        return (
            left_function is not None
            and left_function is right_function
            and getattr(left, "__self__", None) is getattr(right, "__self__", None)
        )

    def add_terminal_observer(self, observer: TerminalObserver) -> bool:
        """Register for future terminal transitions, at most once by identity."""
        if not callable(observer):
            raise TypeError("terminal observer must be callable")
        with self._lock:
            if any(
                self._same_observer(candidate, observer)
                for candidate in self._terminal_observers
            ):
                return False
            self._terminal_observers.append(observer)
            return True

    def remove_terminal_observer(self, observer: TerminalObserver) -> bool:
        """Remove an observer; an already-won transition may still deliver its snapshot."""
        with self._lock:
            for index, candidate in enumerate(self._terminal_observers):
                if self._same_observer(candidate, observer):
                    del self._terminal_observers[index]
                    return True
            return False

    @property
    def terminal_observer_failures(self) -> int:
        with self._lock:
            return self._terminal_observer_failures

    def create(self, request: Any, queue_timeout_ms: int, execution_timeout_ms: int) -> RequestRecord:
        now = self._now()
        record = RequestRecord(uuid.uuid4().hex, uuid.uuid4().hex, request, now + queue_timeout_ms / 1000, execution_timeout_ms / 1000)
        record.timestamps[Lifecycle.RECEIVED.value] = now
        with self._lock:
            self._records[record.request_id] = record
        return record

    def transition(self, record: RequestRecord, target: Lifecycle, *, result: Any = None, error: Any = None) -> bool:
        with self._lock, record.condition:
            changed, observers = self._transition_locked(
                record,
                target,
                result=result,
                error=error,
            )
        self._notify_terminal(record, observers)
        return changed

    def compare_and_transition(
        self,
        record: RequestRecord,
        expected: Lifecycle,
        target: Lifecycle,
        *,
        predicate: Callable[[], bool] | None = None,
        result: Any = None,
        error: Any = None,
    ) -> bool:
        """Atomically transition only while lifecycle and optional guard still match."""
        with self._lock, record.condition:
            if record.lifecycle != expected:
                return False
            if predicate is not None and not predicate():
                return False
            changed, observers = self._transition_locked(
                record,
                target,
                result=result,
                error=error,
            )
        self._notify_terminal(record, observers)
        return changed

    def _transition_locked(
        self,
        record: RequestRecord,
        target: Lifecycle,
        *,
        result: Any = None,
        error: Any = None,
    ) -> tuple[bool, tuple[TerminalObserver, ...]]:
        if record.lifecycle in TERMINAL:
            return False, ()
        if target not in ALLOWED.get(record.lifecycle, set()):
            raise RuntimeError(f"invalid lifecycle transition {record.lifecycle} -> {target}")
        record.lifecycle = target
        record.timestamps[target.value] = self._now()
        record.result, record.error = result, error
        record.condition.notify_all()
        observers = tuple(self._terminal_observers) if target in TERMINAL else ()
        return True, observers

    def _notify_terminal(
        self,
        record: RequestRecord,
        observers: tuple[TerminalObserver, ...],
    ) -> None:
        # Observers run after registry and condition locks are released. The
        # single winning transition owns this immutable callback snapshot.
        for observer in observers:
            try:
                observer(record)
            except Exception:
                # Telemetry must never roll back or disrupt terminalization.
                with self._lock:
                    self._terminal_observer_failures += 1

    def cancel(self, request_id: str) -> bool:
        observers: tuple[TerminalObserver, ...] = ()
        with self._lock:
            record = self._records.get(request_id)
            if record is None or record.lifecycle in TERMINAL:
                return False
            record.cancel_event.set()
            if record.lifecycle == Lifecycle.QUEUED:
                with record.condition:
                    _, observers = self._transition_locked(
                        record,
                        Lifecycle.CANCELLED,
                        error="cancelled",
                    )
        self._notify_terminal(record, observers)
        return True

    def get(self, request_id: str) -> RequestRecord | None:
        with self._lock:
            return self._records.get(request_id)

    def snapshot(self) -> tuple[RequestRecord, ...]:
        with self._lock:
            return tuple(self._records.values())

    def prune_terminal(
        self,
        *,
        ttl_seconds: float,
        max_records: int,
        now: float | None = None,
    ) -> RegistryPruneStats:
        """Remove at most ``max_records`` expired terminal records oldest-first."""
        try:
            bounded_ttl = float(ttl_seconds)
        except (OverflowError, TypeError, ValueError):
            bounded_ttl = math.nan
        if type(ttl_seconds) not in {int, float} or not math.isfinite(bounded_ttl) or bounded_ttl < 0:
            raise ValueError("ttl_seconds must be a finite non-negative number")
        if type(max_records) is not int or max_records < 0:
            raise ValueError("max_records must be a non-negative integer")
        if now is None:
            current = self._now()
        else:
            try:
                current = float(now)
            except (OverflowError, TypeError, ValueError):
                current = math.nan
            if type(now) not in {int, float} or not math.isfinite(current):
                raise ValueError("now must be a finite number")

        with self._lock:
            scanned = len(self._records)
            terminal_records = 0
            candidates: list[tuple[float, str, RequestRecord]] = []
            for request_id, record in self._records.items():
                if record.lifecycle not in TERMINAL:
                    continue
                terminal_records += 1
                terminal_at = record.timestamps.get(record.lifecycle.value)
                if terminal_at is None:
                    continue
                if current - terminal_at >= bounded_ttl:
                    candidates.append((terminal_at, request_id, record))

            candidates.sort(key=lambda item: (item[0], item[1]))
            removed = 0
            for _, request_id, record in candidates[:max_records]:
                if self._records.get(request_id) is record:
                    del self._records[request_id]
                    removed += 1
            eligible_remaining = len(candidates) - removed
            remaining = len(self._records)

        return RegistryPruneStats(
            scanned_records=scanned,
            terminal_records=terminal_records,
            eligible_records=len(candidates),
            removed_records=removed,
            remaining_records=remaining,
            eligible_remaining=eligible_remaining,
            max_records=max_records,
            limit_reached=eligible_remaining > 0 and removed >= max_records,
        )
