from __future__ import annotations

import json
from typing import Any

from .errors import WorkerError


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _has_unpaired_surrogate(value: Any) -> bool:
    if isinstance(value, str):
        return any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    if isinstance(value, list):
        return any(_has_unpaired_surrogate(item) for item in value)
    if isinstance(value, dict):
        return any(
            _has_unpaired_surrogate(key) or _has_unpaired_surrogate(item)
            for key, item in value.items()
        )
    return False


def loads(data: bytes | str, *, too_large: int | None = None) -> Any:
    try:
        raw = data if isinstance(data, bytes) else data.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise WorkerError("invalid_request", "invalid JSON Unicode scalar") from exc
    if too_large is not None and len(raw) > too_large:
        raise WorkerError("request_too_large", "request body exceeds byte limit")
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        if _has_unpaired_surrogate(value):
            raise ValueError("unpaired Unicode surrogate is forbidden")
        return value
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkerError("invalid_request", f"invalid JSON: {exc}") from exc
