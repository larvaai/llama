from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from .errors import ContractError
from .ids import validate_identifier

MAX_DECISION_BYTES = 16_384
MAX_TEXT_BYTES = 8_192


class AgentRunLifecycle(StrEnum):
    READY = "ready"
    DECIDING = "deciding"
    WAITING_TOOL = "waiting_tool"
    OBSERVING = "observing"
    SYNTHESIZING = "synthesizing"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    FAILED = "failed"
    WAITING_USER = "waiting_user"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    BUDGET_EXHAUSTED = "budget_exhausted"


class ToolInvocationLifecycle(StrEnum):
    PROPOSED = "proposed"
    ARGS_READY = "args_ready"
    NEEDS_RESOLUTION = "needs_resolution"
    AUTHORIZED = "authorized"
    WAITING_APPROVAL = "waiting_approval"
    DENIED = "denied"
    DISPATCHED = "dispatched"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    UNKNOWN_OUTCOME = "unknown_outcome"


class EffectClass(StrEnum):
    READ_ONLY = "read_only"
    IDEMPOTENT_MUTATION = "idempotent_mutation"
    NON_IDEMPOTENT_MUTATION = "non_idempotent_mutation"


class SideEffectState(StrEnum):
    NONE = "none"
    NOT_DISPATCHED = "not_dispatched"
    APPLIED = "applied"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class RetryScope(StrEnum):
    NEVER = "never"
    TRANSIENT = "transient"
    IDEMPOTENT = "idempotent"
    RECONCILE_ONLY = "reconcile_only"


class AuthorizationDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class FlowPolicy(StrEnum):
    DIRECT_RETURN_SAFE = "direct_return_safe"
    SYNTHESIZE_NO_TOOLS = "synthesize_no_tools"
    REPLAN_WITH_OBSERVATION = "replan_with_observation"
    VERIFY_THEN_REPLAN = "verify_then_replan"
    PAUSE_FOR_APPROVAL = "pause_for_approval"
    PERSIST_AND_STOP = "persist_and_stop"
    HANDOFF = "handoff"


class DecisionAction(StrEnum):
    CALL_TOOL = "call_tool"
    ASK_USER = "ask_user"
    SUBMIT = "submit"
    BLOCKED = "blocked"


def _enum(value: object, expected: type[StrEnum], field: str) -> None:
    if not isinstance(value, expected):
        raise ContractError("invalid_contract", f"{field} must be a known {expected.__name__}")


def _text(value: object, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value:
        raise ContractError("invalid_contract", f"{field} must be a non-empty string")
    if len(value.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ContractError("contract_too_large", f"{field} exceeds its UTF-8 byte cap")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ContractError("invalid_contract", f"{field} contains a control character")
    return value


def _strict_object(raw: str | bytes, *, max_bytes: int) -> dict[str, Any]:
    encoded = raw.encode("utf-8") if isinstance(raw, str) else raw
    if len(encoded) > max_bytes:
        raise ContractError("contract_too_large", "JSON exceeds its UTF-8 byte cap")
    try:
        text = encoded.decode("utf-8", errors="strict")

        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise ContractError("invalid_json", f"duplicate key: {key}")
                result[key] = value
            return result

        value = json.loads(
            text,
            object_pairs_hook=pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ContractError("invalid_json", f"non-finite number: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("invalid_json", "invalid canonical JSON input") from error
    if not isinstance(value, dict):
        raise ContractError("invalid_contract", "root must be an object")
    return value


@dataclass(frozen=True, slots=True)
class ToolIntent:
    action: DecisionAction
    tool_id: str | None
    objective: str | None
    input_hint: str | None
    message: str | None

    def __post_init__(self) -> None:
        _enum(self.action, DecisionAction, "action")
        for field in ("tool_id", "objective", "input_hint", "message"):
            _text(getattr(self, field), field, nullable=True)
        self._validate_shape()

    @classmethod
    def parse(cls, raw: str | bytes, *, allowed_tool_ids: frozenset[str]) -> ToolIntent:
        value = _strict_object(raw, max_bytes=MAX_DECISION_BYTES)
        required = {"action", "tool_id", "objective", "input_hint", "message"}
        if set(value) != required:
            raise ContractError("invalid_contract", "ToolIntent must contain exactly five fields")
        try:
            action = DecisionAction(value["action"])
        except (TypeError, ValueError) as error:
            raise ContractError("invalid_contract", "unknown ToolIntent action") from error
        result = cls(
            action=action,
            tool_id=_text(value["tool_id"], "tool_id", nullable=True),
            objective=_text(value["objective"], "objective", nullable=True),
            input_hint=_text(value["input_hint"], "input_hint", nullable=True),
            message=_text(value["message"], "message", nullable=True),
        )
        result.validate(allowed_tool_ids)
        return result

    def validate(self, allowed_tool_ids: frozenset[str]) -> None:
        self._validate_shape()
        if self.action is DecisionAction.CALL_TOOL and self.tool_id not in allowed_tool_ids:
            raise ContractError("unknown_tool", "tool_id is not in the authorized shortlist")

    def _validate_shape(self) -> None:
        if self.action is DecisionAction.CALL_TOOL:
            if not self.tool_id or not self.objective or not self.input_hint or self.message is not None:
                raise ContractError("invalid_contract", "call_tool fields violate the v1 invariant")
        elif any((self.tool_id, self.objective, self.input_hint)) or not self.message:
            raise ContractError("invalid_contract", f"{self.action.value} fields violate the v1 invariant")


def _mapping(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ContractError("invalid_contract", f"{field} must be a string-keyed object")
    reject_nonfinite(value)
    return _freeze(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class CompiledToolCall:
    action_id: str
    invocation_id: str
    internal_call_id: str
    tool_id: str
    tool_version: str
    native_arguments: Mapping[str, Any]
    idempotency_key: str
    effect_class: EffectClass

    def __post_init__(self) -> None:
        for field in ("action_id", "invocation_id", "internal_call_id", "tool_id", "tool_version", "idempotency_key"):
            validate_identifier(getattr(self, field), field=field)
        object.__setattr__(self, "native_arguments", _mapping(self.native_arguments, "native_arguments"))
        _enum(self.effect_class, EffectClass, "effect_class")


@dataclass(frozen=True, slots=True)
class AgentError:
    code: str
    message: str
    retryable: bool
    retry_scope: RetryScope = RetryScope.NEVER
    side_effect_state: SideEffectState = SideEffectState.NONE

    def __post_init__(self) -> None:
        validate_identifier(self.code, field="code")
        _text(self.message, "message")
        if type(self.retryable) is not bool:
            raise ContractError("invalid_contract", "retryable must be a boolean")
        _enum(self.retry_scope, RetryScope, "retry_scope")
        _enum(self.side_effect_state, SideEffectState, "side_effect_state")


@dataclass(frozen=True, slots=True)
class AuthorizationResult:
    decision: AuthorizationDecision
    reason: str
    policy_id: str

    def __post_init__(self) -> None:
        _enum(self.decision, AuthorizationDecision, "decision")
        _text(self.reason, "reason")
        validate_identifier(self.policy_id, field="policy_id")


@dataclass(frozen=True, slots=True)
class ToolResultEnvelope:
    invocation_id: str
    tool_id: str
    tool_version: str
    status: ToolInvocationLifecycle
    summary: str
    data_ref: str | None
    data_hash: str | None
    truncated: bool
    side_effect_state: SideEffectState
    error: AgentError | None
    provenance: str
    trust_label: str = "untrusted"

    def __post_init__(self) -> None:
        for field in ("invocation_id", "tool_id", "tool_version", "provenance"):
            validate_identifier(getattr(self, field), field=field)
        _text(self.summary, "summary")
        _enum(self.status, ToolInvocationLifecycle, "status")
        _enum(self.side_effect_state, SideEffectState, "side_effect_state")
        _text(self.data_ref, "data_ref", nullable=True)
        _text(self.data_hash, "data_hash", nullable=True)
        if self.status not in {ToolInvocationLifecycle.SUCCEEDED, ToolInvocationLifecycle.PARTIAL, ToolInvocationLifecycle.FAILED, ToolInvocationLifecycle.UNKNOWN_OUTCOME}:
            raise ContractError("invalid_contract", "result status must be terminal")
        if type(self.truncated) is not bool or self.trust_label != "untrusted":
            raise ContractError("invalid_contract", "result metadata is invalid")
        if self.status in {ToolInvocationLifecycle.FAILED, ToolInvocationLifecycle.UNKNOWN_OUTCOME} and self.error is None:
            raise ContractError("invalid_contract", "failed results require a typed error")


@dataclass(frozen=True, slots=True)
class FlowTransition:
    policy: FlowPolicy
    next_lifecycle: AgentRunLifecycle
    reason: str

    def __post_init__(self) -> None:
        _enum(self.policy, FlowPolicy, "policy")
        _enum(self.next_lifecycle, AgentRunLifecycle, "next_lifecycle")
        _text(self.reason, "reason")


@dataclass(frozen=True, slots=True)
class ToolRevision:
    version: str
    semantic_schema: Mapping[str, Any]
    native_input_schema: Mapping[str, Any]
    native_output_schema: Mapping[str, Any]
    compiler_revision: str
    effect_class: EffectClass
    permission_policy_id: str
    scope_policy_id: str
    default_flow_policy: FlowPolicy
    timeout_ms: int
    result_byte_cap: int
    result_token_cap: int
    retry_scope: RetryScope

    def __post_init__(self) -> None:
        for field in ("version", "compiler_revision", "permission_policy_id", "scope_policy_id"):
            validate_identifier(getattr(self, field), field=field)
        for field in ("timeout_ms", "result_byte_cap", "result_token_cap"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ContractError("invalid_contract", f"{field} must be a positive integer")
        _enum(self.effect_class, EffectClass, "effect_class")
        _enum(self.default_flow_policy, FlowPolicy, "default_flow_policy")
        _enum(self.retry_scope, RetryScope, "retry_scope")
        for field in ("semantic_schema", "native_input_schema", "native_output_schema"):
            object.__setattr__(self, field, _mapping(getattr(self, field), field))


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    tool_id: str
    revision: ToolRevision
    description: str

    def __post_init__(self) -> None:
        validate_identifier(self.tool_id, field="tool_id")
        if not isinstance(self.revision, ToolRevision):
            raise ContractError("invalid_contract", "revision must be a ToolRevision")
        _text(self.description, "description")


def reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError("invalid_contract", "non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError("invalid_contract", "object key must be a string")
            reject_nonfinite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_nonfinite(item)
