from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..errors import AgentRuntimeError


@dataclass(frozen=True, slots=True)
class AllowlistedPathScope:
    roots: Mapping[str, Path]
    files: Mapping[str, Path]

    def resolve_file(self, ref: str) -> Path:
        try:
            candidate = self.files[ref]
        except KeyError as error:
            raise AgentRuntimeError("scope_violation", "file ref is not allowlisted") from error
        return self._resolve(candidate, self._owning_root(candidate))

    def resolve_root(self, ref: str) -> Path:
        try:
            candidate = self.roots[ref]
        except KeyError as error:
            raise AgentRuntimeError("scope_violation", "root ref is not allowlisted") from error
        return self._resolve(candidate, candidate)

    def _owning_root(self, candidate: Path) -> Path:
        raw = candidate.absolute()
        owners = [root for root in self.roots.values() if raw.is_relative_to(root.absolute())]
        if not owners:
            raise AgentRuntimeError("scope_violation", "file ref has no allowlisted root")
        return max(owners, key=lambda path: len(path.parts))

    @staticmethod
    def _resolve(candidate: Path, root: Path) -> Path:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root):
            raise AgentRuntimeError("scope_violation", "resolved path escapes allowlisted root")
        return resolved
