from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent_runtime.acceptance import DeterministicAcceptanceGate
from agent_runtime.builtin_tools import readonly_catalog
from agent_runtime.compiler import CompileContext, DeterministicCompiler
from agent_runtime.contracts import (
    AuthorizationDecision,
    DecisionAction,
    EffectClass,
    SideEffectState,
    ToolInvocationLifecycle,
    ToolIntent,
)
from agent_runtime.errors import AgentRuntimeError, IntegrityViolation
from agent_runtime.event_store import SqliteAgentEventStore
from agent_runtime.events import new_event
from agent_runtime.executor import AllowlistedReadOnlyExecutor, CancellationToken, RawToolResult
from agent_runtime.flow import CodeOwnedFlowController
from agent_runtime.normalization import DeterministicResultNormalizer, ImmutableArtifactStore
from agent_runtime.permissions import DeterministicPermissionGate, PermissionContext
from agent_runtime.resolver import BoundedArgumentResolver, ResolutionRequest, ResolutionResponse
from agent_runtime.tool_adapters import ReadFileAdapter, SearchTextAdapter
from agent_runtime.tool_adapters.scope import AllowlistedPathScope

BUDGETS = {"turns": 1, "inference_tokens": 1, "inference_ms": 1, "tool_calls": 1, "tool_attempts": 1, "resolver_calls": 1, "retries": 1, "result_bytes": 1, "context_bytes": 1, "deadline_utc": None}


def compiled_call(tmp_path: Path):
    target = tmp_path / "target.txt"
    target.write_text("abcdef", encoding="utf-8")
    scope = AllowlistedPathScope({"root": tmp_path}, {"file": target})
    catalog = readonly_catalog()
    compiler = DeterministicCompiler(catalog, {"read_file": ReadFileAdapter(scope), "search_text": SearchTextAdapter(scope)})
    intent = ToolIntent(DecisionAction.CALL_TOOL, "read_file", "read", "target", None)
    return compiler.compile(intent, CompileContext("run", "turn", catalog.digest, {"target": "file"}))


def test_permission_gate_is_structured_and_mutation_requires_approval(tmp_path: Path):
    call = compiled_call(tmp_path)
    gate = DeterministicPermissionGate()
    context = PermissionContext("worker", "tenant", frozenset({"read_file"}), "digest", "digest")
    assert gate.authorize(call, context).decision is AuthorizationDecision.ALLOW
    assert gate.authorize(call, replace(context, tainted=True)).decision is AuthorizationDecision.DENY
    mutation = replace(call, effect_class=EffectClass.IDEMPOTENT_MUTATION)
    assert gate.authorize(mutation, context).decision is AuthorizationDecision.REQUIRE_APPROVAL


def test_resolver_cannot_substitute_tool_and_has_bounded_budget():
    class Port:
        def __init__(self, tool_id):
            self.tool_id = tool_id

        def resolve(self, request):
            return ResolutionResponse(self.tool_id, {request.missing_slot: "file"})

    request = ResolutionRequest("read_file", {"required": ["target"]}, "target")
    resolver = BoundedArgumentResolver(Port("read_file"), max_calls=1)
    assert resolver.resolve(request) == {"target": "file"}
    with pytest.raises(AgentRuntimeError, match="exhausted"):
        resolver.resolve(request)
    with pytest.raises(AgentRuntimeError, match="change"):
        BoundedArgumentResolver(Port("search_text")).resolve(request)


def test_executor_deduplicates_and_propagates_truncation(tmp_path: Path):
    call = compiled_call(tmp_path)
    executor = AllowlistedReadOnlyExecutor(result_byte_cap=3)
    first = executor.execute(call)
    second = executor.execute(call)
    assert first == second
    assert first.payload == b"abc"
    assert first.truncated
    assert executor.dispatch_count == 1


def test_event_store_has_one_idempotency_claim_winner(tmp_path: Path):
    store = SqliteAgentEventStore(tmp_path / "events.db")
    prefix = [
        new_event("e1", "run_created", {"budgets": BUDGETS}),
        new_event("e2", "run_transition", {"target": "deciding"}),
        new_event("e3", "run_transition", {"target": "waiting_tool"}),
        new_event("e4", "tool_proposed", {"action_id": "action", "invocation_id": "inv", "tool_catalog_digest": "digest"}),
        new_event("e5", "tool_transition", {"invocation_id": "inv", "target": "args_ready"}),
    ]
    store.append("run", 0, prefix)
    payload = {"record_version": "h1.v1", "record_id": "auth1", "invocation_id": "inv", "decision": "allow", "idempotency_key": "idem", "arguments_digest": "digest"}
    store.append("run", 5, [new_event("e6", "authorization_recorded", payload)])
    with pytest.raises(IntegrityViolation, match="already exists"):
        store.append("run", 6, [new_event("e7", "authorization_recorded", {**payload, "record_id": "auth2"})])


def test_search_executor_is_bounded_without_shell(tmp_path: Path):
    (tmp_path / "a.txt").write_text("needle one\nother", encoding="utf-8")
    (tmp_path / "b.txt").write_text("needle two", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe")
    scope = AllowlistedPathScope({"root": tmp_path}, {})
    catalog = readonly_catalog()
    compiler = DeterministicCompiler(catalog, {"search_text": SearchTextAdapter(scope)})
    intent = ToolIntent(DecisionAction.CALL_TOOL, "search_text", "search", "query", None)
    call = compiler.compile(
        intent,
        CompileContext("run", "turn", catalog.digest, {"query": {"root_ref": "root", "pattern": "needle"}}),
    )
    result = AllowlistedReadOnlyExecutor().execute(call)
    assert result.payload.decode().splitlines() == ["a.txt:1:needle one", "b.txt:1:needle two"]


def test_executor_cancellation_and_unknown_tool_fail_closed(tmp_path: Path):
    call = compiled_call(tmp_path)
    token = CancellationToken()
    token.cancel()
    with pytest.raises(AgentRuntimeError, match="before dispatch"):
        AllowlistedReadOnlyExecutor().execute(call, cancellation=token)
    with pytest.raises(AgentRuntimeError, match="not executable"):
        AllowlistedReadOnlyExecutor().execute(replace(call, tool_id="not_allowlisted"))


def test_normalizer_rejects_non_utf8_and_acceptance_checks_hash(tmp_path: Path):
    normalizer = DeterministicResultNormalizer(ImmutableArtifactStore(tmp_path / "artifacts"))
    invalid = RawToolResult("inv", ToolInvocationLifecycle.SUCCEEDED, b"\xff", "invalid")
    with pytest.raises(AgentRuntimeError, match="not UTF-8"):
        normalizer.normalize(invalid, tool_id="read_file", tool_version="1")
    valid = RawToolResult("inv", ToolInvocationLifecycle.SUCCEEDED, b"ok", "ok", SideEffectState.NONE)
    envelope = normalizer.normalize(valid, tool_id="read_file", tool_version="1")
    gate = DeterministicAcceptanceGate()
    assert gate.evaluate(envelope, {"data_hash": "sha256:wrong"}).accepted is False
    assert gate.evaluate(envelope, {"tool_id": "search_text"}).accepted is False


def test_permission_deadline_catalog_and_scope_denials(tmp_path: Path):
    call = compiled_call(tmp_path)
    gate = DeterministicPermissionGate()
    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    base = PermissionContext("worker", "tenant", frozenset({"read_file"}), "same", "same")
    assert gate.authorize(call, replace(base, deadline_utc=expired)).decision is AuthorizationDecision.DENY
    assert gate.authorize(call, replace(base, catalog_digest="old")).decision is AuthorizationDecision.DENY
    assert gate.authorize(call, replace(base, allowed_tools=frozenset())).decision is AuthorizationDecision.DENY


def test_flow_controller_maps_failed_and_unknown_results(tmp_path: Path):
    normalizer = DeterministicResultNormalizer(ImmutableArtifactStore(tmp_path / "artifacts"))
    failed = normalizer.normalize(
        RawToolResult("inv", ToolInvocationLifecycle.FAILED, b"", "failed", SideEffectState.NONE, "failed"),
        tool_id="read_file",
        tool_version="1",
    )
    unknown = normalizer.normalize(
        RawToolResult("inv2", ToolInvocationLifecycle.UNKNOWN_OUTCOME, b"", "unknown", SideEffectState.UNKNOWN, "timeout"),
        tool_id="read_file",
        tool_version="1",
    )
    flow = CodeOwnedFlowController()
    assert flow.decide(failed).next_lifecycle.value == "deciding"
    assert flow.decide(unknown).next_lifecycle.value == "paused"
