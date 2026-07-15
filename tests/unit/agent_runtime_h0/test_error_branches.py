from __future__ import annotations

import sqlite3

import pytest

from agent_runtime.catalog import ImmutableToolCatalog
from agent_runtime.contracts import (
    AgentError,
    AgentRunLifecycle,
    CompiledToolCall,
    EffectClass,
    FlowPolicy,
    FlowTransition,
    SideEffectState,
    ToolResultEnvelope,
    ToolInvocationLifecycle,
)
from agent_runtime.errors import ContractError, IntegrityViolation
from agent_runtime.event_store import SqliteAgentEventStore
from agent_runtime.events import AgentEvent, canonical_json, new_event, parse_canonical_payload, payload_digest
from agent_runtime.ids import CorrelationIds, validate_identifier
from agent_runtime.reducer import replay
from tests.unit.agent_runtime_h0.test_catalog import tool
from tests.unit.agent_runtime_h0.test_reducer import BUDGETS, sequenced


@pytest.mark.parametrize("value", [None, "", "x" * 129, "bad\nvalue"])
def test_identifier_validation_rejects_empty_oversized_and_control_values(value):
    with pytest.raises(ContractError):
        validate_identifier(value, field="test")


def test_correlation_ids_validate_every_identifier():
    ids = CorrelationIds("w", "t", "r", "turn", "action", "invocation", "attempt")
    assert ids.run_id == "r"
    with pytest.raises(ContractError):
        CorrelationIds("w", "t", "r", "turn", "action", "invocation", "")


def test_compiled_call_deep_freezes_json_and_rejects_nonfinite():
    call = CompiledToolCall("a", "i", "internal", "read", "1", {"items": [1, {"x": 2}]}, "key", EffectClass.READ_ONLY)
    assert call.native_arguments["items"] == (1, {"x": 2})
    with pytest.raises(TypeError):
        call.native_arguments["items"][1]["x"] = 3
    with pytest.raises(ContractError):
        CompiledToolCall("a", "i", "internal", "read", "1", {"value": float("nan")}, "key", EffectClass.READ_ONLY)
    with pytest.raises(ContractError):
        CompiledToolCall("a", "i", "internal", "read", "1", {1: "bad"}, "key", EffectClass.READ_ONLY)


def test_result_error_and_flow_value_objects_fail_closed():
    error = AgentError("tool_failed", "failure", False)
    failed = ToolResultEnvelope("i", "read", "1", ToolInvocationLifecycle.FAILED, "failed", None, None, False, SideEffectState.NONE, error, "fake")
    assert failed.trust_label == "untrusted"
    assert FlowTransition(FlowPolicy.PERSIST_AND_STOP, AgentRunLifecycle.FAILED, "stop").reason == "stop"
    with pytest.raises(ContractError):
        AgentError("bad", "failure", 1)
    with pytest.raises(ContractError):
        ToolResultEnvelope("i", "read", "1", ToolInvocationLifecycle.DISPATCHED, "x", None, None, False, SideEffectState.NONE, None, "fake")
    with pytest.raises(ContractError):
        ToolResultEnvelope("i", "read", "1", ToolInvocationLifecycle.FAILED, "x", None, None, False, SideEffectState.NONE, None, "fake")
    with pytest.raises(ContractError):
        ToolResultEnvelope("i", "read", "1", ToolInvocationLifecycle.SUCCEEDED, "x", None, None, 1, SideEffectState.NONE, None, "fake")
    with pytest.raises(ContractError):
        FlowTransition(FlowPolicy.HANDOFF, AgentRunLifecycle.PAUSED, "")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CompiledToolCall("a", "i", "internal", "read", "1", {}, "key", "future"),
        lambda: ToolResultEnvelope("i", "read", "1", "future", "x", None, None, False, SideEffectState.NONE, None, "fake"),
        lambda: ToolResultEnvelope("i", "read", "1", ToolInvocationLifecycle.SUCCEEDED, "x", None, None, False, "future", None, "fake"),
        lambda: FlowTransition("future", AgentRunLifecycle.PAUSED, "x"),
    ],
)
def test_value_objects_reject_raw_or_unknown_enum_values(factory):
    with pytest.raises(ContractError):
        factory()


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '[1]', '{ "a":1}'])
def test_persisted_payload_parser_rejects_duplicate_nonfinite_nonobject_noncanonical(raw):
    with pytest.raises(ContractError):
        parse_canonical_payload(raw)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"event_kind": "future"},
        {"event_version": 2},
        {"sequence": 0},
        {"occurred_at_utc": "not-a-date"},
        {"occurred_at_utc": "2026-07-15T00:00:00+07:00"},
    ],
)
def test_event_contract_rejects_unknown_versions_kinds_sequence_and_non_utc(kwargs):
    values = {"event_id": "e", "event_kind": "run_created", "payload": {}, "occurred_at_utc": "2026-07-15T00:00:00Z"}
    values.update(kwargs)
    with pytest.raises(ContractError):
        AgentEvent(**values)
    with pytest.raises(ContractError):
        canonical_json({"bad": object()})


def test_store_rejects_configuration_and_unknown_database_version(tmp_path):
    with pytest.raises(ValueError):
        SqliteAgentEventStore(tmp_path / "bad.sqlite3", busy_timeout_ms=True)
    path = tmp_path / "future.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version=99")
    connection.close()
    with pytest.raises(IntegrityViolation, match="version"):
        SqliteAgentEventStore(path)


def test_store_rejects_invalid_append_and_noncanonical_persisted_payload(tmp_path):
    path = tmp_path / "events.sqlite3"
    with SqliteAgentEventStore(path, durable=False) as store:
        with pytest.raises(ValueError):
            store.append("run", 0, [])
        store.append("run", 0, [new_event("e", "run_created", {"budgets": BUDGETS})])
        raw = '{ "budgets":{}}'
        store._connection.execute("UPDATE agent_run_events SET payload_json=?, payload_sha256=?", (raw, payload_digest(raw)))
        with pytest.raises(IntegrityViolation, match="persisted"):
            store.load("run")


def test_catalog_duplicate_and_lookup_paths():
    catalog = ImmutableToolCatalog([tool()])
    assert catalog.get("read_file").tool_id == "read_file"
    with pytest.raises(ValueError, match="duplicate"):
        ImmutableToolCatalog([tool(), tool()])


@pytest.mark.parametrize(
    "events",
    [
        (),
        sequenced(new_event("e1", "run_created", {})),
        sequenced(new_event("e1", "run_created", {"budgets": {**BUDGETS, "turns": True}})),
        sequenced(new_event("e1", "run_created", {"budgets": BUDGETS}), new_event("e2", "run_created", {"budgets": BUDGETS})),
        sequenced(new_event("e1", "run_created", {"budgets": BUDGETS}), new_event("e2", "run_transition", {"target": "future"})),
    ],
)
def test_reducer_rejects_malformed_streams(events):
    with pytest.raises(IntegrityViolation):
        replay("run", events)


def test_reducer_rejects_bad_tool_and_budget_payloads_and_terminalizes_once():
    prefix = [
        new_event("e1", "run_created", {"budgets": BUDGETS}),
        new_event("e2", "run_transition", {"target": "deciding"}),
        new_event("e3", "run_transition", {"target": "waiting_tool"}),
    ]
    bad_proposal = sequenced(*prefix, new_event("e4", "tool_proposed", {"action_id": 1, "invocation_id": "i", "tool_catalog_digest": "d"}))
    with pytest.raises(IntegrityViolation):
        replay("run", bad_proposal)
    valid = prefix + [new_event("e4", "tool_proposed", {"action_id": "a", "invocation_id": "i", "tool_catalog_digest": "d"})]
    with pytest.raises(IntegrityViolation):
        replay("run", sequenced(*valid, new_event("e5", "tool_transition", {"invocation_id": "i", "target": "dispatched"})))
    with pytest.raises(IntegrityViolation):
        replay("run", sequenced(*prefix, new_event("e4", "budget_consumed", {"claim_id": "b", "category": "turns", "amount": 99})))
    terminal = sequenced(*prefix, new_event("e4", "budget_consumed", {"claim_id": "b", "category": "turns", "amount": 1, "terminalize": True}))
    assert replay("run", terminal).lifecycle is AgentRunLifecycle.BUDGET_EXHAUSTED
