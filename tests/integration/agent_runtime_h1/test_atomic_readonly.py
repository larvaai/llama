from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime.acceptance import DeterministicAcceptanceGate
from agent_runtime.builtin_tools import readonly_catalog
from agent_runtime.compiler import DeterministicCompiler
from agent_runtime.contracts import AgentRunLifecycle
from agent_runtime.decision import DecisionEngine, DecisionResponse
from agent_runtime.errors import AgentRuntimeError
from agent_runtime.event_store import SqliteAgentEventStore
from agent_runtime.events import new_event
from agent_runtime.executor import AllowlistedReadOnlyExecutor
from agent_runtime.flow import CodeOwnedFlowController
from agent_runtime.normalization import DeterministicResultNormalizer, ImmutableArtifactStore
from agent_runtime.permissions import DeterministicPermissionGate
from agent_runtime.reducer import replay
from agent_runtime.service import AtomicAgentService, AtomicTask, TurnContext
from agent_runtime.tool_adapters import ReadFileAdapter, SearchTextAdapter
from agent_runtime.tool_adapters.scope import AllowlistedPathScope

BUDGETS = {
    "turns": 4,
    "inference_tokens": 1000,
    "inference_ms": 1000,
    "tool_calls": 4,
    "tool_attempts": 4,
    "resolver_calls": 1,
    "retries": 1,
    "result_bytes": 100000,
    "context_bytes": 100000,
    "deadline_utc": None,
}


class TerminalInference:
    def __init__(self, output: dict[str, object]) -> None:
        self.output = output
        self.terminal = False
        self.calls = 0

    def infer(self, request):
        self.calls += 1
        self.terminal = True
        return DecisionResponse(json.dumps(self.output), 8, 4, 1)


class OrderingExecutor(AllowlistedReadOnlyExecutor):
    def __init__(self, inference, store, **kwargs):
        super().__init__(**kwargs)
        self.inference = inference
        self.store = store

    def execute(self, call, *, cancellation=None):
        assert self.inference.terminal, "inference must release before tool I/O"
        kinds = [event.event_kind for event in self.store.load("run-1")]
        assert kinds[-3:] == ["dispatch_started", "tool_transition", "budget_consumed"]
        return super().execute(call, cancellation=cancellation)


class FaultExecutor(AllowlistedReadOnlyExecutor):
    def execute(self, call, *, cancellation=None):
        self.dispatch_count += 1
        raise AgentRuntimeError("post_dispatch_timeout", "injected crash after dispatch")


def make_service(tmp_path: Path, inference, *, executor=None):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "task.txt"
    target.write_text("hello atomic world\napi_key=do-not-leak", encoding="utf-8")
    scope = AllowlistedPathScope({"workspace": workspace}, {"file_ref": target})
    catalog = readonly_catalog()
    adapters = {"read_file": ReadFileAdapter(scope), "search_text": SearchTextAdapter(scope)}
    store = SqliteAgentEventStore(tmp_path / "agent.db")
    store.append(
        "run-1",
        0,
        [
            new_event("created", "run_created", {"budgets": BUDGETS}),
            new_event("deciding", "run_transition", {"target": "deciding"}),
        ],
    )
    executor = executor or OrderingExecutor(inference, store)
    service = AtomicAgentService(
        store=store,
        catalog=catalog,
        decision=DecisionEngine(inference),
        compiler=DeterministicCompiler(catalog, adapters),
        permission=DeterministicPermissionGate(),
        executor=executor,
        normalizer=DeterministicResultNormalizer(ImmutableArtifactStore(tmp_path / "artifacts")),
        flow=CodeOwnedFlowController(),
        acceptance=DeterministicAcceptanceGate(),
    )
    task = AtomicTask(
        "read assigned file",
        ("read succeeds",),
        {"target_file": "file_ref"},
        {"target_file": "file_ref"},
        {"tool_id": "read_file"},
    )
    turn = TurnContext("run-1", "turn-1", "req-1", "infer-attempt-1", "atomic_worker", "tenant-1")
    return store, service, task, turn, executor


def test_end_to_end_read_file_persists_boundaries_and_accepts(tmp_path: Path):
    inference = TerminalInference({"action": "call_tool", "tool_id": "read_file", "objective": "read", "input_hint": "target_file", "message": None})
    store, service, task, turn, executor = make_service(tmp_path, inference)
    result = service.execute_turn(turn, task)
    events = store.load("run-1")
    kinds = [event.event_kind for event in events]
    assert kinds.index("result_recorded") < kinds.index("flow_decided") < kinds.index("acceptance_decided")
    state = replay("run-1", events)
    assert state.lifecycle is AgentRunLifecycle.SUCCEEDED
    assert state.acceptance_verdict is True
    artifact = next((tmp_path / "artifacts").rglob(result.data_hash.removeprefix("sha256:")))
    assert b"do-not-leak" not in artifact.read_bytes()
    assert b"[REDACTED]" in artifact.read_bytes()
    assert inference.calls == 1


def test_post_dispatch_fault_is_unknown_and_recovery_is_not_blind_retry(tmp_path: Path):
    inference = TerminalInference({"action": "call_tool", "tool_id": "read_file", "objective": "read", "input_hint": "target_file", "message": None})
    fault = FaultExecutor()
    store, service, task, turn, _ = make_service(tmp_path, inference, executor=fault)
    result = service.execute_turn(turn, task)
    assert result.status.value == "unknown_outcome"
    assert fault.dispatch_count == 1
    assert replay("run-1", store.load("run-1")).lifecycle is AgentRunLifecycle.PAUSED
    assert service.recover("run-1") == "terminal_or_paused"


def test_submit_cannot_succeed_without_acceptance_evidence(tmp_path: Path):
    inference = TerminalInference({"action": "submit", "tool_id": None, "objective": None, "input_hint": None, "message": "done"})
    store, service, task, turn, _ = make_service(tmp_path, inference)
    verdict = service.execute_turn(turn, task)
    assert verdict.accepted is False
    state = replay("run-1", store.load("run-1"))
    assert state.lifecycle is AgentRunLifecycle.BLOCKED
    assert state.acceptance_verdict is False


def test_catalog_race_fails_before_executor(tmp_path: Path):
    inference = TerminalInference({"action": "call_tool", "tool_id": "read_file", "objective": "read", "input_hint": "target_file", "message": None})
    _, service, task, turn, executor = make_service(tmp_path, inference)
    service.compiler.catalog = readonly_catalog()
    service.compiler.catalog._digest = "sha256:different"
    with pytest.raises(AgentRuntimeError, match="catalog changed"):
        service.execute_turn(turn, task)
    assert executor.dispatch_count == 0
