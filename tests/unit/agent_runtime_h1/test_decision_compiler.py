from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime.builtin_tools import readonly_catalog
from agent_runtime.compiler import CompileContext, DeterministicCompiler
from agent_runtime.contracts import DecisionAction, ToolIntent
from agent_runtime.decision import (
    DecisionContextBuilder,
    DecisionEngine,
    DecisionRequest,
    DecisionResponse,
    SemanticToolCard,
    tool_intent_schema,
)
from agent_runtime.errors import AgentRuntimeError, ContractError
from agent_runtime.tool_adapters import ReadFileAdapter, SearchTextAdapter
from agent_runtime.tool_adapters.scope import AllowlistedPathScope


class FakeInference:
    def __init__(self, output: dict[str, object]) -> None:
        self.output = output
        self.released = False

    def infer(self, request: DecisionRequest) -> DecisionResponse:
        self.released = True
        return DecisionResponse(json.dumps(self.output), 10, 5, 2)


@pytest.mark.parametrize(
    ("action", "values"),
    [
        ("call_tool", {"tool_id": "read_file", "objective": "read", "input_hint": "target_file", "message": None}),
        ("ask_user", {"tool_id": None, "objective": None, "input_hint": None, "message": "which file?"}),
        ("submit", {"tool_id": None, "objective": None, "input_hint": None, "message": "ready"}),
        ("blocked", {"tool_id": None, "objective": None, "input_hint": None, "message": "no safe path"}),
    ],
)
def test_decision_engine_strictly_parses_all_actions(action, values):
    fake = FakeInference({"action": action, **values})
    schema = tool_intent_schema(["read_file"])
    request = DecisionRequest("req", "att", {}, schema, "sha256:ctx", "sha256:cat")
    intent, _ = DecisionEngine(fake).decide(request, current_catalog_digest="sha256:cat")
    assert intent.action is DecisionAction(action)
    assert fake.released


@pytest.mark.parametrize(
    "output",
    [
        b"not-json",
        json.dumps({"action": "call_tool", "tool_id": "unknown", "objective": "x", "input_hint": "x", "message": None}),
        json.dumps({"action": "submit", "tool_id": "read_file", "objective": None, "input_hint": None, "message": "x"}),
        json.dumps([{"action": "submit"}]),
    ],
)
def test_malformed_unknown_and_cross_field_decisions_fail_closed(output):
    class Port:
        def infer(self, request):
            return DecisionResponse(output)

    request = DecisionRequest("req", "att", {}, tool_intent_schema(["read_file"]), "sha256:ctx", "sha256:cat")
    with pytest.raises(ContractError):
        DecisionEngine(Port()).decide(request, current_catalog_digest="sha256:cat")


def test_context_is_bounded_and_schema_matches_shortlist():
    card = SemanticToolCard("read_file", "1", "read", ("target_ref",), "read_only", "utf8")
    context, schema, digest = DecisionContextBuilder().build(
        task_objective="inspect",
        acceptance_criteria=("hash present",),
        state_refs={"target_file": "file_ref"},
        tool_cards=(card,),
        catalog_digest="sha256:cat",
    )
    assert schema["properties"]["tool_id"]["enum"] == ["read_file", None]
    assert context["latest_observation"] is None
    assert digest.startswith("sha256:")


def test_compiler_uses_only_allowlisted_refs_and_stable_ids(tmp_path: Path):
    target = tmp_path / "task.txt"
    target.write_text("hello", encoding="utf-8")
    scope = AllowlistedPathScope({"workspace": tmp_path}, {"file_ref": target})
    catalog = readonly_catalog()
    compiler = DeterministicCompiler(catalog, {"read_file": ReadFileAdapter(scope), "search_text": SearchTextAdapter(scope)})
    intent = ToolIntent(DecisionAction.CALL_TOOL, "read_file", "read", "target_file", None)
    context = CompileContext("run", "turn", catalog.digest, {"target_file": "file_ref"})
    first = compiler.compile(intent, context)
    second = compiler.compile(intent, context)
    assert first == second
    assert first.native_arguments["path"] == str(target.resolve())
    with pytest.raises(AgentRuntimeError, match="semantic binding"):
        compiler.compile(intent, CompileContext("run", "turn", catalog.digest, {}))


def test_symlink_escape_is_rejected_when_supported(tmp_path: Path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    scope = AllowlistedPathScope({"workspace": tmp_path}, {"file_ref": link})
    with pytest.raises(AgentRuntimeError, match="escapes"):
        scope.resolve_file("file_ref")
