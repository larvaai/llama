from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .catalog import ImmutableToolCatalog
from .contracts import CompiledToolCall, DecisionAction, ToolIntent
from .errors import AgentRuntimeError
from .events import canonical_json


class ToolAdapter(Protocol):
    def compile(self, input_hint: str, bindings: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class CompileContext:
    run_id: str
    turn_id: str
    catalog_digest: str
    bindings: Mapping[str, Any]


def _stable_id(prefix: str, value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:40]}"


def arguments_digest(arguments: Mapping[str, Any]) -> str:
    def plain(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [plain(item) for item in value]
        return value

    return "sha256:" + hashlib.sha256(canonical_json(plain(arguments)).encode("utf-8")).hexdigest()


class DeterministicCompiler:
    def __init__(self, catalog: ImmutableToolCatalog, adapters: Mapping[str, ToolAdapter]) -> None:
        self.catalog = catalog
        self.adapters = dict(adapters)

    def compile(self, intent: ToolIntent, context: CompileContext) -> CompiledToolCall:
        if intent.action is not DecisionAction.CALL_TOOL or intent.tool_id is None or intent.input_hint is None:
            raise AgentRuntimeError("not_tool_action", "only call_tool decisions can be compiled")
        if context.catalog_digest != self.catalog.digest:
            raise AgentRuntimeError("catalog_changed", "catalog changed before compilation")
        try:
            definition = self.catalog.get(intent.tool_id)
            adapter = self.adapters[intent.tool_id]
        except KeyError as error:
            raise AgentRuntimeError("unknown_tool", "tool has no allowlisted adapter") from error
        try:
            native = dict(adapter.compile(intent.input_hint, context.bindings))
        except KeyError as error:
            raise AgentRuntimeError("needs_resolution", "semantic binding is missing") from error
        identity = {
            "run_id": context.run_id,
            "turn_id": context.turn_id,
            "catalog_digest": context.catalog_digest,
            "tool_id": intent.tool_id,
            "tool_version": definition.revision.version,
            "input_hint": intent.input_hint,
            "arguments_digest": arguments_digest(native),
        }
        action_id = _stable_id("act", identity)
        invocation_id = _stable_id("inv", {**identity, "action_id": action_id})
        return CompiledToolCall(
            action_id=action_id,
            invocation_id=invocation_id,
            internal_call_id=_stable_id("call", {"invocation_id": invocation_id}),
            tool_id=intent.tool_id,
            tool_version=definition.revision.version,
            native_arguments=native,
            idempotency_key=_stable_id("idem", {**identity, "effect": definition.revision.effect_class.value}),
            effect_class=definition.revision.effect_class,
        )
