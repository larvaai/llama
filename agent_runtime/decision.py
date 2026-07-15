from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .contracts import MAX_DECISION_BYTES, ToolIntent
from .errors import AgentRuntimeError, ContractError
from .events import canonical_json

MAX_CONTEXT_BYTES = 32_768
MAX_STATE_REFS = 32
MIN_TOOL_CARDS = 1
MAX_TOOL_CARDS = 8


@dataclass(frozen=True, slots=True)
class SemanticToolCard:
    tool_id: str
    version: str
    description: str
    input_slots: tuple[str, ...]
    effect_class: str
    result_shape: str


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    request_id: str
    attempt_id: str
    context: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    context_digest: str
    catalog_digest: str


@dataclass(frozen=True, slots=True)
class DecisionResponse:
    output: str | bytes
    input_tokens: int = 0
    output_tokens: int = 0
    inference_ms: int = 0


class DecisionInferencePort(Protocol):
    def infer(self, request: DecisionRequest) -> DecisionResponse: ...


def tool_intent_schema(tool_ids: Sequence[str]) -> dict[str, Any]:
    if not 1 <= len(tool_ids) <= MAX_TOOL_CARDS or len(set(tool_ids)) != len(tool_ids):
        raise ContractError("invalid_shortlist", "tool shortlist must contain 1-8 unique tools")
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["call_tool", "ask_user", "submit", "blocked"],
            },
            "tool_id": {"type": ["string", "null"], "enum": [*tool_ids, None]},
            "objective": {"type": ["string", "null"]},
            "input_hint": {"type": ["string", "null"]},
            "message": {"type": ["string", "null"]},
        },
        "required": ["action", "tool_id", "objective", "input_hint", "message"],
        "additionalProperties": False,
    }


class DecisionContextBuilder:
    def __init__(self, *, instruction_version: str = "atomic-tool-decision.v1") -> None:
        self.instruction_version = instruction_version

    def build(
        self,
        *,
        task_objective: str,
        acceptance_criteria: Sequence[str],
        state_refs: Mapping[str, str],
        tool_cards: Sequence[SemanticToolCard],
        catalog_digest: str,
        latest_observation: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        if len(state_refs) > MAX_STATE_REFS:
            raise ContractError("context_too_large", "too many state references")
        if not MIN_TOOL_CARDS <= len(tool_cards) <= MAX_TOOL_CARDS:
            raise ContractError("invalid_shortlist", "decision context requires 1-8 tools")
        ids = [card.tool_id for card in tool_cards]
        schema = tool_intent_schema(ids)
        context: dict[str, Any] = {
            "instruction_version": self.instruction_version,
            "task": {
                "objective": task_objective,
                "acceptance_criteria": list(acceptance_criteria),
            },
            "state_refs": dict(sorted(state_refs.items())),
            "tool_cards": [
                {
                    "tool_id": card.tool_id,
                    "version": card.version,
                    "description": card.description,
                    "input_slots": list(card.input_slots),
                    "effect_class": card.effect_class,
                    "result_shape": card.result_shape,
                }
                for card in tool_cards
            ],
            "latest_observation": latest_observation,
            "tool_catalog_digest": catalog_digest,
        }
        encoded = canonical_json(context).encode("utf-8")
        if len(encoded) > MAX_CONTEXT_BYTES:
            raise ContractError("context_too_large", "decision context exceeds byte cap")
        digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
        return context, schema, digest


class DecisionEngine:
    def __init__(self, inference: DecisionInferencePort) -> None:
        self.inference = inference

    def decide(self, request: DecisionRequest, *, current_catalog_digest: str) -> tuple[ToolIntent, DecisionResponse]:
        if current_catalog_digest != request.catalog_digest:
            raise AgentRuntimeError("catalog_changed", "catalog changed before inference")
        response = self.inference.infer(request)
        if not isinstance(response, DecisionResponse):
            raise AgentRuntimeError("inference_invalid", "inference returned an invalid response")
        if len(response.output if isinstance(response.output, bytes) else response.output.encode("utf-8")) > MAX_DECISION_BYTES:
            raise ContractError("contract_too_large", "decision exceeds byte cap")
        if current_catalog_digest != request.catalog_digest:
            raise AgentRuntimeError("catalog_changed", "catalog changed during inference")
        allowed = frozenset(request.output_schema["properties"]["tool_id"]["enum"][:-1])
        return ToolIntent.parse(response.output, allowed_tool_ids=allowed), response
