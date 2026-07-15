from __future__ import annotations

from dataclasses import replace

import pytest

from agent_runtime.contracts import AgentRunLifecycle, ToolInvocationLifecycle
from agent_runtime.errors import IntegrityViolation
from agent_runtime.events import new_event
from agent_runtime.reducer import replay


BUDGETS = {"turns": 5, "inference_tokens": 100, "inference_ms": 1000, "tool_calls": 2, "tool_attempts": 3, "resolver_calls": 1, "retries": 1, "result_bytes": 4096, "context_bytes": 4096, "deadline_utc": None}


def sequenced(*events):
    return tuple(replace(event, sequence=index) for index, event in enumerate(events, 1))


def legal_stream():
    return sequenced(
        new_event("e1", "run_created", {"budgets": BUDGETS}),
        new_event("e2", "run_transition", {"target": "deciding"}),
        new_event("e3", "run_transition", {"target": "waiting_tool"}),
        new_event("e4", "tool_proposed", {"action_id": "a1", "invocation_id": "i1", "tool_catalog_digest": "sha256:catalog"}),
        new_event("e5", "tool_transition", {"invocation_id": "i1", "target": "args_ready"}),
        new_event("e6", "tool_transition", {"invocation_id": "i1", "target": "authorized"}),
        new_event("e7", "tool_transition", {"invocation_id": "i1", "target": "dispatched"}),
        new_event("e8", "tool_transition", {"invocation_id": "i1", "target": "succeeded"}),
        new_event("e9", "budget_consumed", {"claim_id": "b1", "category": "tool_calls", "amount": 1}),
        new_event("e10", "run_transition", {"target": "observing"}),
        new_event("e11", "run_transition", {"target": "verifying"}),
        new_event("e12", "run_transition", {"target": "succeeded", "authority": "acceptance_gate"}),
    )


def test_legal_lifecycle_replays_deterministically_and_reconciles_budget_once():
    first = replay("run-1", legal_stream())
    second = replay("run-1", legal_stream())
    assert first == second
    assert first.lifecycle is AgentRunLifecycle.SUCCEEDED
    assert first.tool_invocations["i1"] is ToolInvocationLifecycle.SUCCEEDED
    assert first.budgets.tool_calls == 1


def test_submit_cannot_directly_succeed_without_acceptance_authority():
    events = list(legal_stream())
    events[-1] = replace(events[-1], payload={"target": "succeeded"})
    with pytest.raises(IntegrityViolation, match="acceptance"):
        replay("run-1", events)


def test_duplicate_budget_claim_and_terminalization_fail_closed():
    base = sequenced(
        new_event("e1", "run_created", {"budgets": BUDGETS}),
        new_event("e2", "run_transition", {"target": "deciding"}),
        new_event("e3", "budget_consumed", {"claim_id": "b1", "category": "turns", "amount": 1}),
        new_event("e4", "budget_consumed", {"claim_id": "b1", "category": "turns", "amount": 1}),
    )
    with pytest.raises(IntegrityViolation, match="budget claim"):
        replay("run-1", base)


def test_illegal_transition_does_not_mutate_preceding_durable_state():
    valid = sequenced(new_event("e1", "run_created", {"budgets": BUDGETS}))
    before = replay("run-1", valid)
    invalid = valid + (replace(new_event("e2", "run_transition", {"target": "succeeded", "authority": "acceptance_gate"}), sequence=2),)
    with pytest.raises(IntegrityViolation):
        replay("run-1", invalid)
    assert replay("run-1", valid) == before
