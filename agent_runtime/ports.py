from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .contracts import (
    AuthorizationResult,
    CompiledToolCall,
    FlowTransition,
    ToolIntent,
    ToolResultEnvelope,
)
from .events import AgentEvent


class AgentEventStore(Protocol):  # pragma: no cover - interface declaration
    def append(self, run_id: str, expected_version: int, events: Sequence[AgentEvent]) -> int: ...
    def load(self, run_id: str) -> tuple[AgentEvent, ...]: ...


class ToolCatalog(Protocol):  # pragma: no cover - interface declaration
    @property
    def digest(self) -> str: ...


class ArgumentCompiler(Protocol):  # pragma: no cover - interface declaration
    def compile(self, intent: ToolIntent, state: Mapping[str, Any]) -> CompiledToolCall: ...


class PermissionGate(Protocol):  # pragma: no cover - interface declaration
    def authorize(
        self, call: CompiledToolCall, state: Mapping[str, Any]
    ) -> AuthorizationResult: ...


class ToolExecutor(Protocol):  # pragma: no cover - interface declaration
    def execute(self, call: CompiledToolCall) -> ToolResultEnvelope: ...


class ResultNormalizer(Protocol):  # pragma: no cover - interface declaration
    def normalize(self, value: object) -> ToolResultEnvelope: ...


class FlowController(Protocol):  # pragma: no cover - interface declaration
    def decide(self, result: ToolResultEnvelope, state: Mapping[str, Any]) -> FlowTransition: ...


class AcceptanceGate(Protocol):  # pragma: no cover - interface declaration
    def accept(self, state: Mapping[str, Any]) -> bool: ...
