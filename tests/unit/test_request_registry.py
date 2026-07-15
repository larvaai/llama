from __future__ import annotations

import pytest

from model_worker.events import NullEventSink
from model_worker.request_registry import Lifecycle, RequestRegistry


def test_registry_transitions_notify_and_terminal_state_is_immutable():
    registry = RequestRegistry()
    record = registry.create({"request": True}, 100, 200)
    assert registry.get(record.request_id) is record
    assert registry.snapshot() == (record,)
    assert isinstance(record.event_sink, NullEventSink)
    assert registry.transition(record, Lifecycle.PREFLIGHTED)
    assert registry.transition(record, Lifecycle.QUEUED)
    assert registry.transition(record, Lifecycle.RUNNING)
    assert registry.transition(record, Lifecycle.COMPLETED, result={"ok": True})
    assert record.result == {"ok": True}
    assert registry.transition(record, Lifecycle.FAILED) is False
    assert registry.cancel(record.request_id) is False


def test_registry_rejects_invalid_transition_and_handles_unknown_cancel():
    registry = RequestRegistry()
    record = registry.create({}, 100, 200)
    with pytest.raises(RuntimeError, match="invalid lifecycle transition"):
        registry.transition(record, Lifecycle.RUNNING)
    assert registry.cancel("missing") is False
    assert registry.cancel(record.request_id) is True
    assert record.cancel_event.is_set()


def test_queued_cancel_becomes_terminal_without_running():
    registry = RequestRegistry()
    record = registry.create({}, 100, 200)
    registry.transition(record, Lifecycle.PREFLIGHTED)
    registry.transition(record, Lifecycle.QUEUED)
    assert registry.cancel(record.request_id) is True
    assert record.lifecycle == Lifecycle.CANCELLED
    assert record.error == "cancelled"


def test_compare_and_transition_checks_expected_state_and_guard_atomically():
    registry = RequestRegistry()
    record = registry.create({}, 100, 200)
    registry.transition(record, Lifecycle.PREFLIGHTED)
    registry.transition(record, Lifecycle.QUEUED)

    assert not registry.compare_and_transition(
        record,
        Lifecycle.QUEUED,
        Lifecycle.RUNNING,
        predicate=lambda: False,
    )
    assert record.lifecycle == Lifecycle.QUEUED
    assert registry.compare_and_transition(
        record,
        Lifecycle.QUEUED,
        Lifecycle.RUNNING,
        predicate=lambda: True,
    )
    assert not registry.compare_and_transition(
        record,
        Lifecycle.QUEUED,
        Lifecycle.TIMED_OUT,
        error="queue_timeout",
    )
    assert record.lifecycle == Lifecycle.RUNNING
    assert Lifecycle.TIMED_OUT.value not in record.timestamps
