from __future__ import annotations

import hashlib
from dataclasses import fields, is_dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .contracts import ToolDefinition
from .events import canonical_json


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class ImmutableToolCatalog:
    def __init__(self, tools: Iterable[ToolDefinition]) -> None:
        indexed: dict[str, ToolDefinition] = {}
        for tool in tools:
            if tool.tool_id in indexed:
                raise ValueError(f"duplicate tool_id: {tool.tool_id}")
            indexed[tool.tool_id] = tool
        self._tools = MappingProxyType(indexed)
        encoded = canonical_json({key: _plain(value) for key, value in sorted(indexed.items())})
        self._digest = "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def digest(self) -> str:
        return self._digest

    def get(self, tool_id: str) -> ToolDefinition:
        return self._tools[tool_id]

    def all(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._tools[key] for key in sorted(self._tools))
