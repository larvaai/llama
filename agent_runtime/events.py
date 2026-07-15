from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .contracts import reject_nonfinite
from .errors import ContractError
from .ids import validate_identifier

EVENT_SCHEMA_VERSION = 1
KNOWN_EVENT_KINDS = frozenset(
    {
        "run_created",
        "run_transition",
        "tool_proposed",
        "tool_transition",
        "budget_consumed",
        "decision_recorded",
        "inference_completed",
        "arguments_compiled",
        "authorization_recorded",
        "dispatch_started",
        "result_recorded",
        "flow_decided",
        "acceptance_decided",
    }
)


def canonical_json(value: Mapping[str, Any]) -> str:
    reject_nonfinite(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ContractError("invalid_event", "event payload is not canonical JSON data") from error


def payload_digest(payload_json: str) -> str:
    return "sha256:" + hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def parse_canonical_payload(payload_json: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise IntegrityError(f"duplicate key: {key}")
            result[key] = value
        return result

    class IntegrityError(ValueError):
        pass

    try:
        value = json.loads(
            payload_json,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(IntegrityError(value)),
        )
    except (json.JSONDecodeError, IntegrityError) as error:
        raise ContractError("invalid_event", "persisted payload is not strict JSON") from error
    if not isinstance(value, dict) or canonical_json(value) != payload_json:
        raise ContractError("invalid_event", "persisted payload is not canonical JSON")
    return value


@dataclass(frozen=True, slots=True)
class AgentEvent:
    event_id: str
    event_kind: str
    payload: Mapping[str, Any]
    occurred_at_utc: str
    event_version: int = EVENT_SCHEMA_VERSION
    sequence: int | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.event_id, field="event_id")
        if self.event_kind not in KNOWN_EVENT_KINDS:
            raise ContractError("unknown_event_kind", f"unknown event kind: {self.event_kind}")
        if type(self.event_version) is not int or self.event_version != EVENT_SCHEMA_VERSION:
            raise ContractError("unknown_event_version", "unsupported event schema version")
        if self.sequence is not None and (type(self.sequence) is not int or self.sequence < 1):
            raise ContractError("invalid_event", "sequence must be a positive integer")
        try:
            parsed = datetime.fromisoformat(self.occurred_at_utc.replace("Z", "+00:00"))
        except (TypeError, ValueError) as error:
            raise ContractError("invalid_event", "occurred_at_utc must be ISO-8601") from error
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ContractError("invalid_event", "occurred_at_utc must be UTC")
        canonical_json(self.payload)


def new_event(event_id: str, event_kind: str, payload: Mapping[str, Any]) -> AgentEvent:
    occurred = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return AgentEvent(event_id=event_id, event_kind=event_kind, payload=dict(payload), occurred_at_utc=occurred)
