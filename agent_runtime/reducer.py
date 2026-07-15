from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Iterable

from .contracts import AgentRunLifecycle, ToolInvocationLifecycle
from .errors import IntegrityViolation
from .events import AgentEvent
from .ids import validate_identifier

RUN_TERMINAL = frozenset({AgentRunLifecycle.SUCCEEDED, AgentRunLifecycle.BLOCKED, AgentRunLifecycle.FAILED, AgentRunLifecycle.CANCELLED, AgentRunLifecycle.TIMED_OUT, AgentRunLifecycle.BUDGET_EXHAUSTED})
RUN_TRANSITIONS = {
    AgentRunLifecycle.READY: {AgentRunLifecycle.DECIDING, AgentRunLifecycle.CANCELLED, AgentRunLifecycle.TIMED_OUT},
    AgentRunLifecycle.DECIDING: {AgentRunLifecycle.WAITING_TOOL, AgentRunLifecycle.WAITING_USER, AgentRunLifecycle.PAUSED, AgentRunLifecycle.BLOCKED, AgentRunLifecycle.FAILED, AgentRunLifecycle.CANCELLED, AgentRunLifecycle.TIMED_OUT, AgentRunLifecycle.BUDGET_EXHAUSTED},
    AgentRunLifecycle.WAITING_TOOL: {AgentRunLifecycle.OBSERVING, AgentRunLifecycle.PAUSED, AgentRunLifecycle.FAILED, AgentRunLifecycle.CANCELLED, AgentRunLifecycle.TIMED_OUT, AgentRunLifecycle.BUDGET_EXHAUSTED},
    AgentRunLifecycle.OBSERVING: {AgentRunLifecycle.DECIDING, AgentRunLifecycle.SYNTHESIZING, AgentRunLifecycle.VERIFYING, AgentRunLifecycle.PAUSED, AgentRunLifecycle.BLOCKED, AgentRunLifecycle.FAILED, AgentRunLifecycle.CANCELLED, AgentRunLifecycle.TIMED_OUT, AgentRunLifecycle.BUDGET_EXHAUSTED},
    AgentRunLifecycle.SYNTHESIZING: {AgentRunLifecycle.DECIDING, AgentRunLifecycle.SUCCEEDED, AgentRunLifecycle.BLOCKED, AgentRunLifecycle.FAILED, AgentRunLifecycle.CANCELLED, AgentRunLifecycle.TIMED_OUT, AgentRunLifecycle.BUDGET_EXHAUSTED},
    AgentRunLifecycle.VERIFYING: {AgentRunLifecycle.DECIDING, AgentRunLifecycle.SUCCEEDED, AgentRunLifecycle.BLOCKED, AgentRunLifecycle.FAILED, AgentRunLifecycle.CANCELLED, AgentRunLifecycle.TIMED_OUT, AgentRunLifecycle.BUDGET_EXHAUSTED},
    AgentRunLifecycle.WAITING_USER: {AgentRunLifecycle.DECIDING, AgentRunLifecycle.CANCELLED, AgentRunLifecycle.TIMED_OUT},
    AgentRunLifecycle.PAUSED: {AgentRunLifecycle.DECIDING, AgentRunLifecycle.CANCELLED, AgentRunLifecycle.TIMED_OUT},
}
TOOL_TRANSITIONS = {
    ToolInvocationLifecycle.PROPOSED: {ToolInvocationLifecycle.ARGS_READY, ToolInvocationLifecycle.NEEDS_RESOLUTION},
    ToolInvocationLifecycle.NEEDS_RESOLUTION: {ToolInvocationLifecycle.ARGS_READY, ToolInvocationLifecycle.FAILED},
    ToolInvocationLifecycle.ARGS_READY: {ToolInvocationLifecycle.AUTHORIZED, ToolInvocationLifecycle.WAITING_APPROVAL, ToolInvocationLifecycle.DENIED},
    ToolInvocationLifecycle.WAITING_APPROVAL: {ToolInvocationLifecycle.AUTHORIZED, ToolInvocationLifecycle.DENIED},
    ToolInvocationLifecycle.AUTHORIZED: {ToolInvocationLifecycle.DISPATCHED},
    ToolInvocationLifecycle.DISPATCHED: {ToolInvocationLifecycle.SUCCEEDED, ToolInvocationLifecycle.PARTIAL, ToolInvocationLifecycle.FAILED, ToolInvocationLifecycle.UNKNOWN_OUTCOME},
}


@dataclass(slots=True)
class BudgetState:
    turns: int
    inference_tokens: int
    inference_ms: int
    tool_calls: int
    tool_attempts: int
    resolver_calls: int
    retries: int
    result_bytes: int
    context_bytes: int
    deadline_utc: str | None


@dataclass(slots=True)
class AgentRunState:
    run_id: str
    lifecycle: AgentRunLifecycle
    version: int
    budgets: BudgetState
    tool_invocations: dict[str, ToolInvocationLifecycle] = field(default_factory=dict)
    action_claims: set[str] = field(default_factory=set)
    budget_claims: set[str] = field(default_factory=set)
    catalog_digests: dict[str, str] = field(default_factory=dict)
    h1_records: dict[str, dict[str, object]] = field(default_factory=dict)
    dispatched_invocations: set[str] = field(default_factory=set)
    terminal_results: set[str] = field(default_factory=set)
    acceptance_verdict: bool | None = None


def _fail(message: str) -> None:
    raise IntegrityViolation("illegal_event_stream", message)


def replay(run_id: str, events: Iterable[AgentEvent]) -> AgentRunState:
    validate_identifier(run_id, field="run_id")
    ordered = tuple(events)
    if not ordered or ordered[0].event_kind != "run_created":
        _fail("stream must begin with run_created")
    state: AgentRunState | None = None
    expected = 1
    for event in ordered:
        if event.sequence != expected:
            _fail("event sequence is not contiguous")
        expected += 1
        payload = event.payload
        if event.event_kind == "run_created":
            if state is not None:
                _fail("run_created may occur only once")
            try:
                raw_budgets = payload["budgets"]
                budgets = BudgetState(**raw_budgets)
            except (KeyError, TypeError) as error:
                raise IntegrityViolation("invalid_event_payload", "invalid run budget payload") from error
            for item in fields(budgets):
                name = item.name
                value = getattr(budgets, name)
                if name != "deadline_utc" and (type(value) is not int or value < 0):
                    _fail("budget values must be non-negative integers")
            state = AgentRunState(run_id, AgentRunLifecycle.READY, event.sequence, budgets)
        elif event.event_kind == "run_transition":
            try:
                target = AgentRunLifecycle(payload["target"])
            except (KeyError, ValueError) as error:
                raise IntegrityViolation("invalid_event_payload", "invalid run target") from error
            if target not in RUN_TRANSITIONS.get(state.lifecycle, set()):
                _fail(f"illegal run transition {state.lifecycle.value}->{target.value}")
            if target is AgentRunLifecycle.SUCCEEDED and payload.get("authority") != "acceptance_gate":
                _fail("only deterministic acceptance may succeed a run")
            state.lifecycle = target
            state.version = event.sequence
        elif event.event_kind == "tool_proposed":
            invocation_id = payload.get("invocation_id")
            action_id = payload.get("action_id")
            digest = payload.get("tool_catalog_digest")
            if not isinstance(invocation_id, str) or not isinstance(action_id, str) or not isinstance(digest, str) or state.lifecycle is not AgentRunLifecycle.WAITING_TOOL:
                _fail("invalid tool proposal")
            validate_identifier(invocation_id, field="invocation_id")
            validate_identifier(action_id, field="action_id")
            if invocation_id in state.tool_invocations or action_id in state.action_claims:
                _fail("duplicate action or invocation claim")
            state.tool_invocations[invocation_id] = ToolInvocationLifecycle.PROPOSED
            state.action_claims.add(action_id)
            state.catalog_digests[action_id] = digest
            state.version = event.sequence
        elif event.event_kind == "tool_transition":
            invocation_id = payload.get("invocation_id")
            if not isinstance(invocation_id, str):
                _fail("invalid invocation identifier")
            try:
                current = state.tool_invocations[invocation_id]
                target = ToolInvocationLifecycle(payload["target"])
            except (KeyError, ValueError) as error:
                raise IntegrityViolation("invalid_event_payload", "invalid invocation transition") from error
            if target not in TOOL_TRANSITIONS.get(current, set()):
                _fail(f"illegal tool transition {current.value}->{target.value}")
            state.tool_invocations[invocation_id] = target
            state.version = event.sequence
        elif event.event_kind == "budget_consumed":
            claim_id = payload.get("claim_id")
            category = payload.get("category")
            amount = payload.get("amount")
            if not isinstance(claim_id, str) or not isinstance(category, str) or claim_id in state.budget_claims or not hasattr(state.budgets, category):
                _fail("invalid or duplicate budget claim")
            validate_identifier(claim_id, field="claim_id")
            if category == "deadline_utc" or type(amount) is not int or amount <= 0:
                _fail("invalid budget decrement")
            remaining = getattr(state.budgets, category)
            if amount > remaining:
                _fail("budget cannot become negative")
            setattr(state.budgets, category, remaining - amount)
            state.budget_claims.add(claim_id)
            if payload.get("terminalize"):
                if state.lifecycle in RUN_TERMINAL:
                    _fail("terminalization may occur only once")
                state.lifecycle = AgentRunLifecycle.BUDGET_EXHAUSTED
            state.version = event.sequence
        elif event.event_kind in {
            "decision_recorded",
            "inference_completed",
            "arguments_compiled",
            "authorization_recorded",
            "dispatch_started",
            "result_recorded",
            "flow_decided",
            "acceptance_decided",
        }:
            if payload.get("record_version") != "h1.v1":
                _fail(f"{event.event_kind} has an unsupported record version")
            record_id = payload.get("record_id")
            if not isinstance(record_id, str) or record_id in state.h1_records:
                _fail("invalid or duplicate H1 record")
            validate_identifier(record_id, field="record_id")
            invocation_id = payload.get("invocation_id")
            if event.event_kind in {
                "arguments_compiled",
                "authorization_recorded",
                "dispatch_started",
                "result_recorded",
            }:
                if not isinstance(invocation_id, str) or invocation_id not in state.tool_invocations:
                    _fail("H1 invocation record does not reference a proposal")
            if event.event_kind == "dispatch_started":
                if state.tool_invocations[invocation_id] is not ToolInvocationLifecycle.AUTHORIZED:
                    _fail("dispatch boundary requires an authorized invocation")
                if invocation_id in state.dispatched_invocations:
                    _fail("dispatch boundary may be persisted only once")
                state.dispatched_invocations.add(invocation_id)
            elif event.event_kind == "result_recorded":
                if invocation_id not in state.dispatched_invocations:
                    _fail("terminal result requires a persisted dispatch boundary")
                if invocation_id in state.terminal_results:
                    _fail("terminal result may be persisted only once")
                state.terminal_results.add(invocation_id)
            elif event.event_kind == "acceptance_decided":
                accepted = payload.get("accepted")
                if type(accepted) is not bool or payload.get("authority") != "acceptance_gate":
                    _fail("acceptance verdict requires deterministic authority")
                state.acceptance_verdict = accepted
            state.h1_records[record_id] = dict(payload)
            state.version = event.sequence
    assert state is not None
    return state
