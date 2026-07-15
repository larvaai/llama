from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .errors import AgentRuntimeError


@dataclass(frozen=True, slots=True)
class ResolutionRequest:
    tool_id: str
    semantic_schema: Mapping[str, Any]
    missing_slot: str


@dataclass(frozen=True, slots=True)
class ResolutionResponse:
    tool_id: str
    bindings: Mapping[str, Any]


class ResolutionPort(Protocol):
    def resolve(self, request: ResolutionRequest) -> ResolutionResponse: ...


class BoundedArgumentResolver:
    def __init__(self, port: ResolutionPort, *, max_calls: int = 1) -> None:
        if max_calls < 0:
            raise ValueError("max_calls must be non-negative")
        self.port = port
        self.remaining = max_calls

    def resolve(self, request: ResolutionRequest) -> Mapping[str, Any]:
        if self.remaining <= 0:
            raise AgentRuntimeError("resolver_exhausted", "argument resolver budget exhausted")
        self.remaining -= 1
        response = self.port.resolve(request)
        if not isinstance(response, ResolutionResponse):
            raise AgentRuntimeError("resolver_invalid", "resolver returned invalid output")
        if response.tool_id != request.tool_id:
            raise AgentRuntimeError("tool_substitution", "resolver may not change the selected tool")
        if request.missing_slot not in response.bindings:
            raise AgentRuntimeError("resolver_invalid", "resolver did not bind the missing slot")
        return dict(response.bindings)
