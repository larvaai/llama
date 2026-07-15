from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .acceptance import AcceptanceVerdict, DeterministicAcceptanceGate
from .catalog import ImmutableToolCatalog
from .compiler import CompileContext, DeterministicCompiler, arguments_digest
from .contracts import (
    AgentRunLifecycle,
    AuthorizationDecision,
    DecisionAction,
    SideEffectState,
    ToolInvocationLifecycle,
    ToolResultEnvelope,
)
from .decision import (
    DecisionContextBuilder,
    DecisionEngine,
    DecisionRequest,
    SemanticToolCard,
)
from .errors import AgentRuntimeError
from .event_store import SqliteAgentEventStore
from .events import canonical_json, new_event
from .executor import AllowlistedReadOnlyExecutor, CancellationToken, RawToolResult
from .flow import CodeOwnedFlowController
from .normalization import DeterministicResultNormalizer
from .permissions import DeterministicPermissionGate, PermissionContext
from .reducer import AgentRunState, replay

RECORD_VERSION = "h1.v1"


@dataclass(frozen=True, slots=True)
class AtomicTask:
    objective: str
    acceptance_criteria: tuple[str, ...]
    state_refs: Mapping[str, str]
    bindings: Mapping[str, Any]
    acceptance: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class TurnContext:
    run_id: str
    turn_id: str
    inference_request_id: str
    inference_attempt_id: str
    role: str
    tenant_id: str
    deadline_utc: str | None = None


def _id(prefix: str, value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:40]}"


class AtomicAgentService:
    def __init__(
        self,
        *,
        store: SqliteAgentEventStore,
        catalog: ImmutableToolCatalog,
        decision: DecisionEngine,
        compiler: DeterministicCompiler,
        permission: DeterministicPermissionGate,
        executor: AllowlistedReadOnlyExecutor,
        normalizer: DeterministicResultNormalizer,
        flow: CodeOwnedFlowController | None = None,
        acceptance: DeterministicAcceptanceGate | None = None,
    ) -> None:
        self.store = store
        self.catalog = catalog
        self.decision = decision
        self.compiler = compiler
        self.permission = permission
        self.executor = executor
        self.normalizer = normalizer
        self.flow = flow or CodeOwnedFlowController()
        self.acceptance = acceptance or DeterministicAcceptanceGate()
        self.context_builder = DecisionContextBuilder()

    def execute_turn(
        self,
        turn: TurnContext,
        task: AtomicTask,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ToolResultEnvelope | AcceptanceVerdict:
        state = self._state(turn.run_id)
        if state.lifecycle is not AgentRunLifecycle.DECIDING:
            raise AgentRuntimeError("illegal_state", "run must be deciding at turn start")
        self._check_deadline(turn.deadline_utc)
        if state.budgets.turns < 1:
            raise AgentRuntimeError("budget_exhausted", "no decision turns remain")
        cards = self._cards()
        context, schema, digest = self.context_builder.build(
            task_objective=task.objective,
            acceptance_criteria=task.acceptance_criteria,
            state_refs=task.state_refs,
            tool_cards=cards,
            catalog_digest=self.catalog.digest,
        )
        request = DecisionRequest(
            request_id=turn.inference_request_id,
            attempt_id=turn.inference_attempt_id,
            context=context,
            output_schema=schema,
            context_digest=digest,
            catalog_digest=self.catalog.digest,
        )
        intent, response = self.decision.decide(request, current_catalog_digest=self.catalog.digest)
        if request.catalog_digest != self.catalog.digest:
            raise AgentRuntimeError("catalog_changed", "catalog changed during inference")
        # infer() has returned terminally before any compiler, permission, or tool I/O below.
        state = self._append(
            state,
            "inference_completed",
            {
                "request_id": request.request_id,
                "attempt_id": request.attempt_id,
                "context_digest": digest,
                "tool_catalog_digest": self.catalog.digest,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "inference_ms": response.inference_ms,
            },
        )
        state = self._consume_budget(state, turn.turn_id, "turns", 1)
        token_usage = response.input_tokens + response.output_tokens
        if token_usage:
            state = self._consume_budget(state, request.attempt_id, "inference_tokens", token_usage)
        if response.inference_ms:
            state = self._consume_budget(state, request.attempt_id, "inference_ms", response.inference_ms)
        state = self._append(
            state,
            "decision_recorded",
            {
                "turn_id": turn.turn_id,
                "action": intent.action.value,
                "tool_id": intent.tool_id,
                "input_hint": intent.input_hint,
                "tool_catalog_digest": self.catalog.digest,
                "context_digest": digest,
            },
        )
        if intent.action is not DecisionAction.CALL_TOOL:
            return self._handle_non_tool(state, intent.action, intent.message or "")
        call = self.compiler.compile(
            intent,
            CompileContext(turn.run_id, turn.turn_id, self.catalog.digest, task.bindings),
        )
        state = self._append_plain(state, "run_transition", {"target": "waiting_tool"})
        state = self._append_plain(
            state,
            "tool_proposed",
            {
                "action_id": call.action_id,
                "invocation_id": call.invocation_id,
                "tool_catalog_digest": self.catalog.digest,
            },
        )
        state = self._append(
            state,
            "arguments_compiled",
            {
                "invocation_id": call.invocation_id,
                "action_id": call.action_id,
                "arguments_digest": arguments_digest(call.native_arguments),
                "compiler_revision": self.catalog.get(call.tool_id).revision.compiler_revision,
            },
        )
        state = self._append_plain(state, "tool_transition", {"invocation_id": call.invocation_id, "target": "args_ready"})
        authorization = self.permission.authorize(
            call,
            PermissionContext(
                turn.role,
                turn.tenant_id,
                frozenset(card.tool_id for card in cards),
                self.catalog.digest,
                request.catalog_digest,
                turn.deadline_utc,
            ),
        )
        state = self._append(
            state,
            "authorization_recorded",
            {
                "invocation_id": call.invocation_id,
                "decision": authorization.decision.value,
                "policy_id": authorization.policy_id,
                "arguments_digest": arguments_digest(call.native_arguments),
                "idempotency_key": call.idempotency_key,
            },
        )
        if authorization.decision is not AuthorizationDecision.ALLOW:
            target = "waiting_approval" if authorization.decision is AuthorizationDecision.REQUIRE_APPROVAL else "denied"
            self._append_plain(state, "tool_transition", {"invocation_id": call.invocation_id, "target": target})
            raise AgentRuntimeError("approval_required" if target == "waiting_approval" else "permission_denied", authorization.reason)
        state = self._append_plain(state, "tool_transition", {"invocation_id": call.invocation_id, "target": "authorized"})
        state = self._consume_budget(state, call.invocation_id, "tool_calls", 1)
        # This durable boundary and unique idempotency claim commit before execute().
        state = self._append(
            state,
            "dispatch_started",
            {
                "invocation_id": call.invocation_id,
                "attempt_id": _id("toolatt", {"invocation_id": call.invocation_id, "attempt": 1}),
                "idempotency_key": call.idempotency_key,
                "effect_class": call.effect_class.value,
                "side_effect_state": SideEffectState.NOT_DISPATCHED.value,
            },
        )
        state = self._append_plain(state, "tool_transition", {"invocation_id": call.invocation_id, "target": "dispatched"})
        state = self._consume_budget(state, call.invocation_id, "tool_attempts", 1)
        try:
            raw = self.executor.execute(call, cancellation=cancellation)
        except AgentRuntimeError as error:
            raw = RawToolResult(
                call.invocation_id,
                ToolInvocationLifecycle.UNKNOWN_OUTCOME if error.code.startswith("post_dispatch") else ToolInvocationLifecycle.FAILED,
                b"",
                str(error),
                SideEffectState.UNKNOWN if error.code.startswith("post_dispatch") else SideEffectState.NONE,
                error.code,
                False,
            )
        result = self.normalizer.normalize(raw, tool_id=call.tool_id, tool_version=call.tool_version)
        state = self._append(
            state,
            "result_recorded",
            {
                "invocation_id": call.invocation_id,
                "status": result.status.value,
                "data_ref": result.data_ref,
                "data_hash": result.data_hash,
                "truncated": result.truncated,
                "side_effect_state": result.side_effect_state.value,
            },
        )
        state = self._append_plain(state, "tool_transition", {"invocation_id": call.invocation_id, "target": result.status.value})
        state = self._append_plain(state, "run_transition", {"target": "observing"})
        transition = self.flow.decide(result, verify=True)
        state = self._append(
            state,
            "flow_decided",
            {
                "invocation_id": call.invocation_id,
                "policy": transition.policy.value,
                "target": transition.next_lifecycle.value,
            },
        )
        state = self._append_plain(state, "run_transition", {"target": transition.next_lifecycle.value})
        if transition.next_lifecycle is AgentRunLifecycle.VERIFYING:
            verdict = self.acceptance.evaluate(result, task.acceptance)
            state = self._append(
                state,
                "acceptance_decided",
                {
                    "accepted": verdict.accepted,
                    "authority": "acceptance_gate",
                    "invocation_id": call.invocation_id,
                    "reason": verdict.reason,
                },
            )
            target = "succeeded" if verdict.accepted else "blocked"
            self._append_plain(state, "run_transition", {"target": target, "authority": "acceptance_gate"})
        return result

    def recover(self, run_id: str) -> str:
        state = self._state(run_id)
        in_flight = state.dispatched_invocations - state.terminal_results
        if in_flight:
            return "reconcile_only"
        if state.lifecycle in {AgentRunLifecycle.DECIDING, AgentRunLifecycle.READY}:
            return "safe_retry"
        return "terminal_or_paused"

    def _cards(self) -> tuple[SemanticToolCard, ...]:
        return tuple(
            SemanticToolCard(
                tool.tool_id,
                tool.revision.version,
                tool.description,
                tuple(tool.revision.semantic_schema.get("required", ())),
                tool.revision.effect_class.value,
                str(tool.revision.native_output_schema.get("type", "object")),
            )
            for tool in self.catalog.all()
        )

    def _handle_non_tool(self, state: AgentRunState, action: DecisionAction, message: str) -> AcceptanceVerdict:
        if action is DecisionAction.ASK_USER:
            self._append_plain(state, "run_transition", {"target": "waiting_user", "reason": message})
            return AcceptanceVerdict(False, "waiting for user")
        if action is DecisionAction.BLOCKED:
            self._append_plain(state, "run_transition", {"target": "blocked", "reason": message})
            return AcceptanceVerdict(False, "model proposed blocked; code terminalized")
        verdict = AcceptanceVerdict(False, "submit has no deterministic tool evidence")
        state = self._append(
            state,
            "acceptance_decided",
            {"accepted": False, "authority": "acceptance_gate", "reason": verdict.reason},
        )
        self._append_plain(state, "run_transition", {"target": "blocked", "authority": "acceptance_gate"})
        return verdict

    def _state(self, run_id: str) -> AgentRunState:
        return replay(run_id, self.store.load(run_id))

    def _append(self, state: AgentRunState, kind: str, payload: Mapping[str, Any]) -> AgentRunState:
        record_id = _id("rec", {"run_id": state.run_id, "sequence": state.version + 1, "kind": kind, "payload": dict(payload)})
        return self._append_plain(state, kind, {"record_version": RECORD_VERSION, "record_id": record_id, **payload})

    def _append_plain(self, state: AgentRunState, kind: str, payload: Mapping[str, Any]) -> AgentRunState:
        event_id = _id("evt", {"run_id": state.run_id, "sequence": state.version + 1, "kind": kind, "payload": dict(payload)})
        self.store.append(state.run_id, state.version, [new_event(event_id, kind, payload)])
        return self._state(state.run_id)

    def _consume_budget(self, state: AgentRunState, namespace: str, category: str, amount: int) -> AgentRunState:
        remaining = getattr(state.budgets, category)
        if amount > remaining:
            raise AgentRuntimeError("budget_exhausted", f"{category} budget exhausted")
        claim_id = _id("budget", {"run_id": state.run_id, "namespace": namespace, "category": category})
        return self._append_plain(
            state,
            "budget_consumed",
            {"claim_id": claim_id, "category": category, "amount": amount},
        )

    @staticmethod
    def _check_deadline(deadline_utc: str | None) -> None:
        if deadline_utc is None:
            return
        deadline = datetime.fromisoformat(deadline_utc.replace("Z", "+00:00"))
        if datetime.now(timezone.utc) >= deadline:
            raise AgentRuntimeError("deadline_exhausted", "agent turn deadline expired")
