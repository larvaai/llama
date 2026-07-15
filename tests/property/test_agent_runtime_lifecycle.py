from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given, strategies as st

from agent_runtime.contracts import AgentRunLifecycle
from agent_runtime.errors import IntegrityViolation
from agent_runtime.events import new_event
from agent_runtime.reducer import RUN_TRANSITIONS, replay
from tests.unit.agent_runtime_h0.test_reducer import BUDGETS


@given(st.lists(st.sampled_from(list(AgentRunLifecycle)), min_size=1, max_size=20))
def test_random_illegal_run_sequences_fail_closed(targets):
    events = [new_event("event-0", "run_created", {"budgets": BUDGETS})]
    current = AgentRunLifecycle.READY
    legal = True
    for index, target in enumerate(targets, 1):
        if target not in RUN_TRANSITIONS.get(current, set()) or target is AgentRunLifecycle.SUCCEEDED:
            legal = False
        events.append(new_event(f"event-{index}", "run_transition", {"target": target.value}))
        current = target
    stream = tuple(replace(event, sequence=index) for index, event in enumerate(events, 1))
    if legal:
        replay("run-property", stream)
    else:
        with pytest.raises(IntegrityViolation):
            replay("run-property", stream)


@given(st.sampled_from([(AgentRunLifecycle.READY, AgentRunLifecycle.DECIDING), (AgentRunLifecycle.DECIDING, AgentRunLifecycle.WAITING_TOOL), (AgentRunLifecycle.WAITING_TOOL, AgentRunLifecycle.OBSERVING), (AgentRunLifecycle.OBSERVING, AgentRunLifecycle.SYNTHESIZING), (AgentRunLifecycle.OBSERVING, AgentRunLifecycle.VERIFYING)]))
def test_representative_legal_edges_replay_deterministically(edge):
    source, target = edge
    path = [AgentRunLifecycle.READY]
    while path[-1] != source:
        path.append({AgentRunLifecycle.DECIDING: AgentRunLifecycle.READY, AgentRunLifecycle.WAITING_TOOL: AgentRunLifecycle.DECIDING, AgentRunLifecycle.OBSERVING: AgentRunLifecycle.WAITING_TOOL}.get(path[-1], source))
        if len(path) > 10:
            break
    transitions = {
        AgentRunLifecycle.READY: [],
        AgentRunLifecycle.DECIDING: [AgentRunLifecycle.DECIDING],
        AgentRunLifecycle.WAITING_TOOL: [AgentRunLifecycle.DECIDING, AgentRunLifecycle.WAITING_TOOL],
        AgentRunLifecycle.OBSERVING: [AgentRunLifecycle.DECIDING, AgentRunLifecycle.WAITING_TOOL, AgentRunLifecycle.OBSERVING],
    }[source] + [target]
    events = [new_event("edge-0", "run_created", {"budgets": BUDGETS})]
    events += [new_event(f"edge-{i}", "run_transition", {"target": value.value}) for i, value in enumerate(transitions, 1)]
    stream = tuple(replace(event, sequence=index) for index, event in enumerate(events, 1))
    assert replay("run-edge", stream) == replay("run-edge", stream)
