from __future__ import annotations

import threading

import pytest

from agent_runtime.errors import ConcurrencyConflict, IntegrityViolation
from agent_runtime.event_store import SqliteAgentEventStore
from agent_runtime.events import new_event
from agent_runtime.reducer import replay
from tests.unit.agent_runtime_h0.test_reducer import BUDGETS


def created(event_id="e1"):
    return new_event(event_id, "run_created", {"budgets": BUDGETS})


def test_reopen_at_every_event_boundary_replays_identically(tmp_path):
    path = tmp_path / "events.sqlite3"
    events = [created(), new_event("e2", "run_transition", {"target": "deciding"}), new_event("e3", "run_transition", {"target": "blocked"})]
    for index, event in enumerate(events):
        with SqliteAgentEventStore(path) as store:
            assert store.append("run-1", index, [event]) == index + 1
        with SqliteAgentEventStore(path) as reopened:
            loaded = reopened.load("run-1")
            assert replay("run-1", loaded).version == index + 1


def test_concurrent_append_same_version_has_exactly_one_winner(tmp_path):
    path = tmp_path / "events.sqlite3"
    with SqliteAgentEventStore(path) as store:
        store.append("run-1", 0, [created()])
    barrier = threading.Barrier(2)
    outcomes = []

    def writer(event_id, target):
        with SqliteAgentEventStore(path) as store:
            barrier.wait()
            try:
                store.append("run-1", 1, [new_event(event_id, "run_transition", {"target": target})])
                outcomes.append("won")
            except ConcurrencyConflict:
                outcomes.append("lost")

    threads = [threading.Thread(target=writer, args=("e2a", "deciding")), threading.Thread(target=writer, args=("e2b", "cancelled"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["lost", "won"]


def test_duplicate_event_action_and_invocation_claims_are_conflicts(tmp_path):
    path = tmp_path / "events.sqlite3"
    with SqliteAgentEventStore(path) as store:
        store.append("run-1", 0, [created(), new_event("ready", "run_transition", {"target": "deciding"}), new_event("waiting", "run_transition", {"target": "waiting_tool"})])
        event = new_event("e2", "tool_proposed", {"action_id": "a1", "invocation_id": "i1", "tool_catalog_digest": "sha256:x"})
        store.append("run-1", 3, [event])
        with pytest.raises(IntegrityViolation):
            store.append("run-2", 0, [event])
        duplicate_claim = new_event("e3", "tool_proposed", {"action_id": "a1", "invocation_id": "i2", "tool_catalog_digest": "sha256:x"})
        with pytest.raises(IntegrityViolation):
            store.append("run-1", 4, [duplicate_claim])


def test_illegal_append_rolls_back_without_changing_durable_state(tmp_path):
    path = tmp_path / "events.sqlite3"
    with SqliteAgentEventStore(path) as store:
        store.append("run-1", 0, [created()])
        with pytest.raises(IntegrityViolation, match="illegal run transition"):
            store.append("run-1", 1, [new_event("illegal", "run_transition", {"target": "succeeded", "authority": "acceptance_gate"})])
        assert len(store.load("run-1")) == 1
        assert store.append("run-1", 1, [new_event("legal", "run_transition", {"target": "deciding"})]) == 2


@pytest.mark.parametrize("corruption", ["hash", "gap", "version"])
def test_corruption_gap_and_unknown_version_are_detected(tmp_path, corruption):
    path = tmp_path / "events.sqlite3"
    with SqliteAgentEventStore(path) as store:
        store.append("run-1", 0, [created()])
        if corruption == "hash":
            store._connection.execute("UPDATE agent_run_events SET payload_sha256='sha256:bad'")
        elif corruption == "gap":
            store._connection.execute("UPDATE agent_run_events SET sequence=2")
        else:
            store._connection.execute("UPDATE agent_run_events SET event_version=99")
        with pytest.raises(IntegrityViolation):
            store.load("run-1")


def test_durable_mode_and_sqlite_safety_pragmas_are_explicit(tmp_path):
    with SqliteAgentEventStore(tmp_path / "events.sqlite3", durable=True, busy_timeout_ms=1234) as store:
        assert store._connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert store._connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert store._connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert store._connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1234
