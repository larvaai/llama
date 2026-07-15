from __future__ import annotations

from typing import Any

from ..errors import ErrorDetail
from .ast import ContractAST, PrimitiveRule


def _matches(value: Any, rule: PrimitiveRule) -> bool:
    checks = {"string": lambda: type(value) is str, "integer": lambda: type(value) is int, "boolean": lambda: type(value) is bool, "null": lambda: value is None}
    return any(checks[item]() for item in rule.types) and (rule.enum is None or value in rule.enum)


def validate_output(value: Any, contract: ContractAST) -> list[ErrorDetail]:
    if type(value) is not dict:
        return [ErrorDetail("$", "output must be an object")]
    expected = [rule.name for rule in contract.properties]
    if list(value) != expected:
        return [ErrorDetail("$", "output keys must exactly match canonical contract order")]
    return [ErrorDetail(f"$.{rule.name}", "value violates type or enum") for rule in contract.properties if not _matches(value[rule.name], rule)]
