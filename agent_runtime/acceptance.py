from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contracts import ToolResultEnvelope


@dataclass(frozen=True, slots=True)
class AcceptanceVerdict:
    accepted: bool
    reason: str


class DeterministicAcceptanceGate:
    def evaluate(self, result: ToolResultEnvelope, criteria: Mapping[str, str]) -> AcceptanceVerdict:
        if result.status.value != "succeeded" or result.data_hash is None:
            return AcceptanceVerdict(False, "tool result is not a verified success")
        expected_hash = criteria.get("data_hash")
        if expected_hash is not None and expected_hash != result.data_hash:
            return AcceptanceVerdict(False, "artifact hash does not match acceptance criteria")
        required_tool = criteria.get("tool_id")
        if required_tool is not None and required_tool != result.tool_id:
            return AcceptanceVerdict(False, "tool does not match acceptance criteria")
        return AcceptanceVerdict(True, "deterministic acceptance criteria passed")
