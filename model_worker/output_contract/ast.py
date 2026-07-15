from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PrimitiveRule:
    name: str
    types: tuple[str, ...]
    enum: tuple[Any, ...] | None


@dataclass(frozen=True, slots=True)
class ContractAST:
    version: str
    properties: tuple[PrimitiveRule, ...]
