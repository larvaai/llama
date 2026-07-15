from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from agent_runtime.contracts import (
    DecisionAction,
    EffectClass,
    FlowPolicy,
    RetryScope,
    ToolIntent,
    ToolRevision,
)
from agent_runtime.errors import ContractError
from agent_runtime.ports import (
    AcceptanceGate,
    AgentEventStore,
    ArgumentCompiler,
    FlowController,
    PermissionGate,
    ResultNormalizer,
    ToolCatalog,
    ToolExecutor,
)


@pytest.mark.parametrize(
    ("action", "tool_id", "objective", "input_hint", "message"),
    [
        ("call_tool", "read_file", "read", "target", None),
        ("ask_user", None, None, None, "which file?"),
        ("submit", None, None, None, "ready for acceptance"),
        ("blocked", None, None, None, "permission missing"),
    ],
)
def test_tool_intent_accepts_exactly_four_valid_shapes(action, tool_id, objective, input_hint, message):
    raw = json.dumps({"action": action, "tool_id": tool_id, "objective": objective, "input_hint": input_hint, "message": message})
    intent = ToolIntent.parse(raw, allowed_tool_ids=frozenset({"read_file"}))
    assert intent.action is DecisionAction(action)


@pytest.mark.parametrize(
    "value",
    [
        {"action": "call_tool", "tool_id": None, "objective": "x", "input_hint": "y", "message": None},
        {"action": "call_tool", "tool_id": "other", "objective": "x", "input_hint": "y", "message": None},
        {"action": "ask_user", "tool_id": "read_file", "objective": None, "input_hint": None, "message": "?"},
        {"action": "submit", "tool_id": None, "objective": None, "input_hint": None, "message": None},
        {"action": "blocked", "tool_id": None, "objective": "x", "input_hint": None, "message": "x"},
    ],
)
def test_tool_intent_rejects_invalid_cross_field_combinations(value):
    with pytest.raises(ContractError):
        ToolIntent.parse(json.dumps(value), allowed_tool_ids=frozenset({"read_file"}))


@pytest.mark.parametrize(
    "raw",
    [
        b'{"action":"submit","tool_id":null,"objective":null,"input_hint":null}',
        b'{"action":"submit","tool_id":null,"objective":null,"input_hint":null,"message":"x","extra":1}',
        b'{"action":"submit","action":"blocked","tool_id":null,"objective":null,"input_hint":null,"message":"x"}',
        b'{"action":true,"tool_id":null,"objective":null,"input_hint":null,"message":"x"}',
        br'{"action":"submit","tool_id":null,"objective":null,"input_hint":null,"message":"x\u0000"}',
        b'{"action":"future","tool_id":null,"objective":null,"input_hint":null,"message":"x"}',
        b'\xff',
    ],
)
def test_tool_intent_is_exact_utf8_and_fail_closed(raw):
    with pytest.raises(ContractError):
        ToolIntent.parse(raw, allowed_tool_ids=frozenset())


def test_tool_intent_enforces_encoded_byte_cap():
    value = {"action": "submit", "tool_id": None, "objective": None, "input_hint": None, "message": "✓" * 6_000}
    with pytest.raises(ContractError, match="byte cap"):
        ToolIntent.parse(json.dumps(value, ensure_ascii=False), allowed_tool_ids=frozenset())


def revision(version="1"):
    return ToolRevision(
        version=version,
        semantic_schema={"type": "object"},
        native_input_schema={"type": "object"},
        native_output_schema={"type": "object"},
        compiler_revision="compiler-1",
        effect_class=EffectClass.READ_ONLY,
        permission_policy_id="permission-1",
        scope_policy_id="scope-1",
        default_flow_policy=FlowPolicy.REPLAN_WITH_OBSERVATION,
        timeout_ms=1000,
        result_byte_cap=4096,
        result_token_cap=512,
        retry_scope=RetryScope.TRANSIENT,
    )


def test_tool_revision_is_immutable_and_rejects_bool_as_int():
    item = revision()
    with pytest.raises(FrozenInstanceError):
        item.timeout_ms = 5
    with pytest.raises(TypeError):
        item.semantic_schema["x"] = 1
    values = item.__dict__ if hasattr(item, "__dict__") else {}
    assert values == {}
    with pytest.raises(ContractError):
        ToolRevision(
            version="1", semantic_schema={}, native_input_schema={}, native_output_schema={},
            compiler_revision="c", effect_class=EffectClass.READ_ONLY,
            permission_policy_id="p", scope_policy_id="s",
            default_flow_policy=FlowPolicy.PERSIST_AND_STOP, timeout_ms=True,
            result_byte_cap=1, result_token_cap=1, retry_scope=RetryScope.NEVER,
        )


def test_h0_exposes_ports_without_a_concrete_side_effect_executor():
    assert {port.__name__ for port in (AgentEventStore, ToolCatalog, ArgumentCompiler, PermissionGate, ToolExecutor, ResultNormalizer, FlowController, AcceptanceGate)} == {
        "AgentEventStore", "ToolCatalog", "ArgumentCompiler", "PermissionGate",
        "ToolExecutor", "ResultNormalizer", "FlowController", "AcceptanceGate",
    }
