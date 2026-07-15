from __future__ import annotations

from typing import Any, Mapping

from .scope import AllowlistedPathScope


class ReadFileAdapter:
    def __init__(self, scope: AllowlistedPathScope) -> None:
        self.scope = scope

    def compile(self, input_hint: str, bindings: Mapping[str, Any]) -> Mapping[str, Any]:
        ref = bindings[input_hint]
        if not isinstance(ref, str):
            raise KeyError(input_hint)
        return {"path": str(self.scope.resolve_file(ref)), "ref": ref}
