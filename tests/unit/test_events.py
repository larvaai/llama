from __future__ import annotations

import json
import threading

import pytest

from model_worker.events import (
    EVENT_PROTOCOL_VERSION,
    BoundedRequestEventBuffer,
    EventSink,
    EventValidationCode,
    EventValidationError,
    InferenceEvent,
    InferenceEventType,
    NullEventSink,
    PublishResult,
)


REQUEST_ID = "request"
ATTEMPT_ID = "attempt"


def event(sequence: int, event_type: InferenceEventType, **payload) -> InferenceEvent:
    return InferenceEvent(
        event_type,
        REQUEST_ID,
        ATTEMPT_ID,
        sequence,
        **payload,
    )


def test_inference_event_is_typed_versioned_and_serializes_canonically():
    delta = event(0, InferenceEventType.FINAL_DELTA, delta="xin chào")
    assert delta.protocol_version == EVENT_PROTOCOL_VERSION
    assert json.loads(delta.to_json_bytes()) == {
        "protocol_version": "inference-event.v1",
        "type": "final_delta",
        "request_id": REQUEST_ID,
        "attempt_id": ATTEMPT_ID,
        "sequence": 0,
        "delta": "xin chào",
    }
    assert delta.encoded_size == len(delta.to_json_bytes())

    progress = InferenceEvent.from_ipc_frame(
        {
            "protocol_version": "model-worker-ipc.v1",
            "type": "progress",
            "request_id": REQUEST_ID,
            "attempt_id": ATTEMPT_ID,
            "sequence": 1,
            "phase": "prompt_decode",
            "tokens": 32,
        }
    )
    assert progress == event(
        1,
        InferenceEventType.PROGRESS,
        phase="prompt_decode",
        tokens=32,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"event_type": "progress", "phase": "final", "tokens": 1},
        {"event_type": InferenceEventType.PROGRESS, "phase": "final", "tokens": True},
        {"event_type": InferenceEventType.FINAL_DELTA, "delta": ""},
        {"event_type": InferenceEventType.HEARTBEAT, "tokens": 1, "delta": "extra"},
        {"event_type": InferenceEventType.STARTED, "phase": "final"},
    ],
)
def test_inference_event_rejects_invalid_type_and_cross_field_payload(kwargs):
    with pytest.raises(EventValidationError) as captured:
        InferenceEvent(
            request_id=REQUEST_ID,
            attempt_id=ATTEMPT_ID,
            sequence=0,
            **kwargs,
        )
    assert captured.value.code is EventValidationCode.INVALID_EVENT


def test_ipc_conversion_rejects_unknown_or_missing_payload_fields():
    base = {
        "protocol_version": "model-worker-ipc.v1",
        "type": "heartbeat",
        "request_id": REQUEST_ID,
        "attempt_id": ATTEMPT_ID,
        "sequence": 0,
    }
    for invalid in (
        {**base, "unknown": 1},
        base,
        {**base, "type": "completed", "sampled_tokens": 1},
        {**base, "protocol_version": "wrong", "sampled_tokens": 1},
    ):
        with pytest.raises(EventValidationError) as captured:
            InferenceEvent.from_ipc_frame(invalid)
        assert captured.value.code is EventValidationCode.INVALID_EVENT


def test_buffer_validates_correlation_and_strict_source_order():
    buffer = BoundedRequestEventBuffer(
        REQUEST_ID,
        ATTEMPT_ID,
        max_events=4,
        max_bytes=4096,
    )
    wrong_request = InferenceEvent(
        InferenceEventType.STARTED,
        "other",
        ATTEMPT_ID,
        0,
    )
    with pytest.raises(EventValidationError) as captured:
        buffer.publish(wrong_request)
    assert captured.value.code is EventValidationCode.CORRELATION_MISMATCH
    assert buffer.next_sequence == 0

    with pytest.raises(EventValidationError) as captured:
        buffer.publish(event(1, InferenceEventType.STARTED))
    assert captured.value.code is EventValidationCode.SEQUENCE_MISMATCH
    assert buffer.next_sequence == 0

    assert buffer.publish(event(0, InferenceEventType.STARTED)) is PublishResult.ENQUEUED
    assert buffer.publish(event(1, InferenceEventType.PHASE, phase="final")) is PublishResult.ENQUEUED
    assert [item.sequence for item in buffer.drain()] == [0, 1]


def test_progress_and_heartbeat_coalesce_without_reordering_retained_events():
    buffer = BoundedRequestEventBuffer(
        REQUEST_ID,
        ATTEMPT_ID,
        max_events=3,
        max_bytes=4096,
    )
    assert buffer.publish(event(0, InferenceEventType.FINAL_DELTA, delta="a")) is PublishResult.ENQUEUED
    assert buffer.publish(
        event(1, InferenceEventType.PROGRESS, phase="reasoning", tokens=16)
    ) is PublishResult.ENQUEUED
    assert buffer.publish(
        event(2, InferenceEventType.HEARTBEAT, tokens=16)
    ) is PublishResult.ENQUEUED
    assert buffer.publish(
        event(3, InferenceEventType.PROGRESS, phase="final", tokens=32)
    ) is PublishResult.COALESCED
    assert buffer.publish(
        event(4, InferenceEventType.HEARTBEAT, tokens=32)
    ) is PublishResult.COALESCED

    retained = buffer.drain()
    assert [(item.event_type, item.sequence) for item in retained] == [
        (InferenceEventType.FINAL_DELTA, 0),
        (InferenceEventType.PROGRESS, 3),
        (InferenceEventType.HEARTBEAT, 4),
    ]
    assert buffer.coalesced_total == 2


def test_telemetry_drops_when_full_but_lossless_events_are_preserved():
    buffer = BoundedRequestEventBuffer(
        REQUEST_ID,
        ATTEMPT_ID,
        max_events=1,
        max_bytes=4096,
    )
    first = event(0, InferenceEventType.FINAL_DELTA, delta="first")
    assert buffer.publish(first) is PublishResult.ENQUEUED
    assert buffer.publish(
        event(1, InferenceEventType.HEARTBEAT, tokens=1)
    ) is PublishResult.DROPPED_TELEMETRY
    assert buffer.dropped_telemetry_total == 1
    assert buffer.drain() == (first,)
    assert buffer.next_sequence == 2


@pytest.mark.parametrize("limit", ["events", "bytes"])
def test_final_delta_overflow_returns_typed_slow_consumer_without_blocking(limit):
    first = event(0, InferenceEventType.FINAL_DELTA, delta="first")
    max_events = 1 if limit == "events" else 10
    max_bytes = 4096 if limit == "events" else first.encoded_size
    buffer = BoundedRequestEventBuffer(
        REQUEST_ID,
        ATTEMPT_ID,
        max_events=max_events,
        max_bytes=max_bytes,
    )
    assert buffer.publish(first) is PublishResult.ENQUEUED

    result = []
    producer = threading.Thread(
        target=lambda: result.append(
            buffer.publish(event(1, InferenceEventType.FINAL_DELTA, delta="second"))
        )
    )
    producer.start()
    producer.join(0.2)
    assert not producer.is_alive(), "publish must not wait for buffer capacity"
    assert result == [PublishResult.SLOW_CONSUMER]
    assert result[0].value == "slow_consumer"
    assert buffer.slow_consumer is True
    assert buffer.drain() == (first,)
    assert buffer.get(timeout=0) is None


def test_one_thousand_final_deltas_are_lossless_and_drain_in_linear_order():
    buffer = BoundedRequestEventBuffer(
        REQUEST_ID,
        ATTEMPT_ID,
        max_events=1000,
        max_bytes=1024 * 1024,
    )
    for sequence in range(1000):
        assert buffer.publish(
            event(sequence, InferenceEventType.FINAL_DELTA, delta=f"{sequence},")
        ) is PublishResult.ENQUEUED

    assert buffer.queued_events == 1000
    assert buffer.queued_bytes > 0
    drained = buffer.drain()
    assert [item.sequence for item in drained] == list(range(1000))
    assert "".join(item.delta or "" for item in drained).startswith("0,1,2,")
    assert buffer.queued_events == 0
    assert buffer.queued_bytes == 0


def test_null_sink_and_close_are_explicit_and_protocol_compatible():
    sink: EventSink = NullEventSink()
    assert isinstance(sink, EventSink)
    assert sink.publish(event(0, InferenceEventType.STARTED)) is PublishResult.ENQUEUED

    buffer = BoundedRequestEventBuffer(
        REQUEST_ID,
        ATTEMPT_ID,
        max_events=1,
        max_bytes=1024,
    )
    buffer.close()
    assert buffer.publish(event(0, InferenceEventType.STARTED)) is PublishResult.CLOSED
    assert buffer.get(timeout=0) is None
