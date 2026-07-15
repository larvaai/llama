from __future__ import annotations

from dataclasses import replace
from tempfile import TemporaryDirectory

from hypothesis import given, settings, strategies as st

from agent_runtime.event_store import SqliteAgentEventStore
from agent_runtime.events import new_event
from agent_runtime.reducer import replay
from tests.unit.agent_runtime_h0.test_reducer import BUDGETS


@given(st.lists(st.integers(min_value=1, max_value=5), min_size=0, max_size=5))
@settings(deadline=None, max_examples=30)
def test_sqlite_reopen_matches_in_memory_budget_replay(amounts):
    remaining = BUDGETS["turns"]
    accepted = []
    for amount in amounts:
        if amount <= remaining:
            accepted.append(amount)
            remaining -= amount
    events = [new_event("budget-event-0", "run_created", {"budgets": BUDGETS})]
    events += [new_event(f"budget-event-{index}", "budget_consumed", {"claim_id": f"claim-{index}", "category": "turns", "amount": amount}) for index, amount in enumerate(accepted, 1)]
    memory = replay("run-budget", tuple(replace(event, sequence=index) for index, event in enumerate(events, 1)))
    with TemporaryDirectory() as directory:
        path = f"{directory}/events.sqlite3"
        version = 0
        for event in events:
            with SqliteAgentEventStore(path) as store:
                version = store.append("run-budget", version, [event])
        with SqliteAgentEventStore(path) as store:
            durable = replay("run-budget", store.load("run-budget"))
    assert durable == memory
