from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


EVENT_PROTOCOL_VERSION = "inference-event.v1"
IPC_PROTOCOL_VERSION = "model-worker-ipc.v1"
_MAX_CORRELATION_ID_BYTES = 128
_MAX_PHASE_BYTES = 64


class InferenceEventType(str, Enum):
    STARTED = "started"
    PHASE = "phase"
    FINAL_DELTA = "final_delta"
    PROGRESS = "progress"
    HEARTBEAT = "heartbeat"


class PublishResult(str, Enum):
    ENQUEUED = "enqueued"
    COALESCED = "coalesced"
    DROPPED_TELEMETRY = "dropped_telemetry"
    SLOW_CONSUMER = "slow_consumer"
    CLOSED = "closed"


class EventValidationCode(str, Enum):
    INVALID_EVENT = "invalid_event"
    CORRELATION_MISMATCH = "correlation_mismatch"
    SEQUENCE_MISMATCH = "sequence_mismatch"


class EventValidationError(ValueError):
    def __init__(self, code: EventValidationCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _invalid(message: str) -> EventValidationError:
    return EventValidationError(EventValidationCode.INVALID_EVENT, message)


def _validate_bounded_text(value: Any, name: str, max_bytes: int) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > max_bytes:
        raise _invalid(f"{name} must be a non-empty string of at most {max_bytes} bytes")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise _invalid(f"{name} must not contain control characters")
    return value


def _validate_non_negative_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise _invalid(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class InferenceEvent:
    event_type: InferenceEventType
    request_id: str
    attempt_id: str
    sequence: int
    phase: str | None = None
    tokens: int | None = None
    delta: str | None = None
    protocol_version: str = EVENT_PROTOCOL_VERSION
    _encoded_size: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.protocol_version != EVENT_PROTOCOL_VERSION:
            raise _invalid(f"unsupported event protocol version: {self.protocol_version!r}")
        if type(self.event_type) is not InferenceEventType:
            raise _invalid("event_type must be an InferenceEventType")
        _validate_bounded_text(self.request_id, "request_id", _MAX_CORRELATION_ID_BYTES)
        _validate_bounded_text(self.attempt_id, "attempt_id", _MAX_CORRELATION_ID_BYTES)
        _validate_non_negative_integer(self.sequence, "sequence")
        self._validate_payload()
        object.__setattr__(self, "_encoded_size", len(self.to_json_bytes()))

    def _validate_payload(self) -> None:
        if self.event_type is InferenceEventType.STARTED:
            if self.phase is not None or self.tokens is not None or self.delta is not None:
                raise _invalid("started event must not carry a payload")
            return
        if self.event_type is InferenceEventType.PHASE:
            _validate_bounded_text(self.phase, "phase", _MAX_PHASE_BYTES)
            if self.tokens is not None or self.delta is not None:
                raise _invalid("phase event may only carry phase")
            return
        if self.event_type is InferenceEventType.FINAL_DELTA:
            if type(self.delta) is not str or not self.delta:
                raise _invalid("final_delta event requires a non-empty delta string")
            if self.phase is not None or self.tokens is not None:
                raise _invalid("final_delta event may only carry delta")
            return
        if self.event_type is InferenceEventType.PROGRESS:
            _validate_bounded_text(self.phase, "phase", _MAX_PHASE_BYTES)
            _validate_non_negative_integer(self.tokens, "tokens")
            if self.delta is not None:
                raise _invalid("progress event must not carry delta")
            return
        if self.event_type is InferenceEventType.HEARTBEAT:
            _validate_non_negative_integer(self.tokens, "tokens")
            if self.phase is not None or self.delta is not None:
                raise _invalid("heartbeat event may only carry tokens")
            return
        raise _invalid(f"unsupported event type: {self.event_type!r}")

    @property
    def encoded_size(self) -> int:
        return self._encoded_size

    @property
    def is_coalescible(self) -> bool:
        return self.event_type in {
            InferenceEventType.PROGRESS,
            InferenceEventType.HEARTBEAT,
        }

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "type": self.event_type.value,
            "request_id": self.request_id,
            "attempt_id": self.attempt_id,
            "sequence": self.sequence,
        }
        if self.event_type is InferenceEventType.PHASE:
            payload["phase"] = self.phase
        elif self.event_type is InferenceEventType.FINAL_DELTA:
            payload["delta"] = self.delta
        elif self.event_type is InferenceEventType.PROGRESS:
            payload.update({"phase": self.phase, "tokens": self.tokens})
        elif self.event_type is InferenceEventType.HEARTBEAT:
            payload["sampled_tokens"] = self.tokens
        return payload

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            self.as_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    @classmethod
    def from_ipc_frame(cls, frame: Mapping[str, Any]) -> InferenceEvent:
        if not isinstance(frame, Mapping) or frame.get("protocol_version") != IPC_PROTOCOL_VERSION:
            raise _invalid("invalid IPC event frame or protocol version")
        try:
            event_type = InferenceEventType(frame.get("type"))
        except (TypeError, ValueError) as exc:
            raise _invalid("IPC frame is not a supported inference event") from exc

        common = {"protocol_version", "type", "request_id", "attempt_id", "sequence"}
        payload_keys = {
            InferenceEventType.STARTED: set(),
            InferenceEventType.PHASE: {"phase"},
            InferenceEventType.FINAL_DELTA: {"delta"},
            InferenceEventType.PROGRESS: {"phase", "tokens"},
            InferenceEventType.HEARTBEAT: {"sampled_tokens"},
        }[event_type]
        if set(frame) != common | payload_keys:
            raise _invalid("IPC event frame has missing or unknown fields")

        return cls(
            event_type=event_type,
            request_id=frame["request_id"],
            attempt_id=frame["attempt_id"],
            sequence=frame["sequence"],
            phase=frame.get("phase"),
            tokens=frame.get("tokens", frame.get("sampled_tokens")),
            delta=frame.get("delta"),
        )


@runtime_checkable
class EventSink(Protocol):
    def publish(self, event: InferenceEvent) -> PublishResult: ...


class NullEventSink:
    def publish(self, event: InferenceEvent) -> PublishResult:
        if type(event) is not InferenceEvent:
            raise _invalid("event sink only accepts InferenceEvent instances")
        return PublishResult.ENQUEUED


class BoundedRequestEventBuffer:
    """A single-request, non-blocking producer buffer for inference events."""

    def __init__(
        self,
        request_id: str,
        attempt_id: str,
        *,
        max_events: int,
        max_bytes: int,
        next_sequence: int = 0,
    ) -> None:
        _validate_bounded_text(request_id, "request_id", _MAX_CORRELATION_ID_BYTES)
        _validate_bounded_text(attempt_id, "attempt_id", _MAX_CORRELATION_ID_BYTES)
        if type(max_events) is not int or max_events <= 0:
            raise ValueError("max_events must be a positive integer")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        _validate_non_negative_integer(next_sequence, "next_sequence")

        self.request_id = request_id
        self.attempt_id = attempt_id
        self.max_events = max_events
        self.max_bytes = max_bytes
        self._next_sequence = next_sequence
        self._events: deque[InferenceEvent] = deque()
        self._queued_bytes = 0
        self._closed = False
        self._slow_consumer = False
        self._coalesced_total = 0
        self._dropped_telemetry_total = 0
        self._condition = threading.Condition()

    def publish(self, event: InferenceEvent) -> PublishResult:
        if type(event) is not InferenceEvent:
            raise _invalid("event sink only accepts InferenceEvent instances")
        with self._condition:
            if self._closed:
                return PublishResult.CLOSED
            self._validate_stream_position(event)
            self._next_sequence += 1
            if self._slow_consumer:
                return PublishResult.SLOW_CONSUMER

            coalesced = self._remove_older_telemetry(event) if event.is_coalescible else False
            if self._fits(event):
                self._events.append(event)
                self._queued_bytes += event.encoded_size
                if coalesced:
                    self._coalesced_total += 1
                self._condition.notify()
                return PublishResult.COALESCED if coalesced else PublishResult.ENQUEUED

            if event.is_coalescible:
                self._dropped_telemetry_total += 1
                return PublishResult.DROPPED_TELEMETRY

            self._slow_consumer = True
            self._condition.notify_all()
            return PublishResult.SLOW_CONSUMER

    def _validate_stream_position(self, event: InferenceEvent) -> None:
        if event.request_id != self.request_id or event.attempt_id != self.attempt_id:
            raise EventValidationError(
                EventValidationCode.CORRELATION_MISMATCH,
                "event request/attempt identity does not match the buffer",
            )
        if event.sequence != self._next_sequence:
            raise EventValidationError(
                EventValidationCode.SEQUENCE_MISMATCH,
                f"expected event sequence {self._next_sequence}, got {event.sequence}",
            )

    def _remove_older_telemetry(self, event: InferenceEvent) -> bool:
        for queued in self._events:
            if queued.event_type is event.event_type:
                self._events.remove(queued)
                self._queued_bytes -= queued.encoded_size
                return True
        return False

    def _fits(self, event: InferenceEvent) -> bool:
        return (
            len(self._events) < self.max_events
            and self._queued_bytes + event.encoded_size <= self.max_bytes
        )

    def get(self, timeout: float | None = None) -> InferenceEvent | None:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._events:
                if self._closed or self._slow_consumer:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            event = self._events.popleft()
            self._queued_bytes -= event.encoded_size
            return event

    def drain(self) -> tuple[InferenceEvent, ...]:
        with self._condition:
            events = tuple(self._events)
            self._events.clear()
            self._queued_bytes = 0
            return events

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def queued_events(self) -> int:
        with self._condition:
            return len(self._events)

    @property
    def queued_bytes(self) -> int:
        with self._condition:
            return self._queued_bytes

    @property
    def next_sequence(self) -> int:
        with self._condition:
            return self._next_sequence

    @property
    def slow_consumer(self) -> bool:
        with self._condition:
            return self._slow_consumer

    @property
    def coalesced_total(self) -> int:
        with self._condition:
            return self._coalesced_total

    @property
    def dropped_telemetry_total(self) -> int:
        with self._condition:
            return self._dropped_telemetry_total
