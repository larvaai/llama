from __future__ import annotations

from dataclasses import dataclass, fields

from .errors import ContractError

MAX_ID_BYTES = 128


def validate_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError("invalid_identifier", f"{field} must be a non-empty string")
    if len(value.encode("utf-8")) > MAX_ID_BYTES:
        raise ContractError("invalid_identifier", f"{field} exceeds {MAX_ID_BYTES} UTF-8 bytes")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ContractError("invalid_identifier", f"{field} contains a control character")
    return value


@dataclass(frozen=True, slots=True)
class CorrelationIds:
    workflow_id: str
    task_id: str
    run_id: str
    turn_id: str
    action_id: str
    invocation_id: str
    attempt_id: str

    def __post_init__(self) -> None:
        for item in fields(self):
            validate_identifier(getattr(self, item.name), field=item.name)
