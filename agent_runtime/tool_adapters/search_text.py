from __future__ import annotations

from typing import Any, Mapping

from ..errors import AgentRuntimeError
from .scope import AllowlistedPathScope

MAX_PATTERN_BYTES = 512


class SearchTextAdapter:
    def __init__(self, scope: AllowlistedPathScope) -> None:
        self.scope = scope

    def compile(self, input_hint: str, bindings: Mapping[str, Any]) -> Mapping[str, Any]:
        value = bindings[input_hint]
        if not isinstance(value, Mapping):
            raise KeyError(input_hint)
        ref = value.get("root_ref")
        pattern = value.get("pattern")
        if not isinstance(ref, str) or not isinstance(pattern, str) or not pattern:
            raise KeyError(input_hint)
        if len(pattern.encode("utf-8")) > MAX_PATTERN_BYTES or any(ord(c) < 0x20 for c in pattern):
            raise AgentRuntimeError("invalid_arguments", "search pattern is not bounded text")
        return {"root": str(self.scope.resolve_root(ref)), "root_ref": ref, "pattern": pattern}
