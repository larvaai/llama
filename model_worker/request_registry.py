from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
    result: Any = None
    error: Any = None
    condition: threading.Condition = field(default_factory=threading.Condition)


class RequestRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, RequestRecord] = {}

    def create(self, request: Any, queue_timeout_ms: int, execution_timeout_ms: int) -> RequestRecord:
        now = time.monotonic()
        record = RequestRecord(uuid.uuid4().hex, uuid.uuid4().hex, request, now + queue_timeout_ms / 1000, execution_timeout_ms / 1000)
        record.timestamps[Lifecycle.RECEIVED.value] = now
        with self._lock:
            self._records[record.request_id] = record
        return record

    def transition(self, record: RequestRecord, target: Lifecycle, *, result: Any = None, error: Any = None) -> bool:
        with self._lock, record.condition:
            if record.lifecycle in TERMINAL:
                return False
            if target not in ALLOWED.get(record.lifecycle, set()):
                raise RuntimeError(f"invalid lifecycle transition {record.lifecycle} -> {target}")
            record.lifecycle = target
            record.timestamps[target.value] = time.monotonic()
            record.result, record.error = result, error
            record.condition.notify_all()
            return True

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            record = self._records.get(request_id)
            if record is None or record.lifecycle in TERMINAL:
                return False
            record.cancel_event.set()
            if record.lifecycle == Lifecycle.QUEUED:
                self.transition(record, Lifecycle.CANCELLED, error="cancelled")
            return True

    def get(self, request_id: str) -> RequestRecord | None:
        with self._lock:
            return self._records.get(request_id)

    def snapshot(self) -> tuple[RequestRecord, ...]:
        with self._lock:
            return tuple(self._records.values())
