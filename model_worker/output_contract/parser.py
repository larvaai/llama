from __future__ import annotations

import math
from typing import Any

from ..errors import ErrorDetail, WorkerError
from .ast import ContractAST, PrimitiveRule

SUPPORTED_TYPES = {"string", "integer", "boolean", "null"}
ROOT_KEYS = {"type", "properties", "required", "additionalProperties"}
PROPERTY_KEYS = {"type", "enum"}


def _fail(path: str, message: str) -> None:
    raise WorkerError("unsupported_contract", message, details=[ErrorDetail(path, message)])


def _matches(value: Any, declared: str) -> bool:
    return {"string": type(value) is str, "integer": type(value) is int, "boolean": type(value) is bool, "null": value is None}[declared]


def parse_contract(schema: Any, *, max_properties: int = 64) -> ContractAST:
    if type(schema) is not dict:
        _fail("$", "schema must be an object")
    unknown = set(schema) - ROOT_KEYS
    if unknown:
        _fail(f"$.{sorted(unknown)[0]}", "unsupported schema keyword")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        _fail("$", "root must be object with additionalProperties=false")
    properties = schema.get("properties")
    required = schema.get("required")
    if type(properties) is not dict or not properties or len(properties) > max_properties:
        _fail("$.properties", "properties must be a bounded non-empty object")
    if type(required) is not list or any(type(name) is not str for name in required) or len(set(required)) != len(required):
        _fail("$.required", "required must contain unique strings")
    if required != list(properties):
        _fail("$.required", "all properties must be required in canonical property order")
    rules = []
    for name, spec in properties.items():
        path = f"$.properties.{name}"
        if type(name) is not str or not name or type(spec) is not dict:
            _fail(path, "property name and rule must be valid")
        if set(spec) - PROPERTY_KEYS:
            _fail(path, "unsupported property keyword")
        raw_types = spec.get("type")
        if type(raw_types) is str:
            types = (raw_types,)
        elif type(raw_types) is list and raw_types and all(type(item) is str for item in raw_types):
            types = tuple(raw_types)
        else:
            _fail(path + ".type", "type is required")
        if len(set(types)) != len(types) or not set(types) <= SUPPORTED_TYPES:
            _fail(path + ".type", "duplicate or unsupported type")
        if len(types) > 2 or (len(types) == 2 and "null" not in types):
            _fail(path + ".type", "only a primitive optionally unioned with null is supported")
        enum = spec.get("enum")
        enum_tuple = None
        if "enum" in spec:
            if type(enum) is not list or not enum:
                _fail(path + ".enum", "enum must be non-empty")
            for value in enum:
                if type(value) is float and not math.isfinite(value):
                    _fail(path + ".enum", "non-finite enum value")
                if not any(_matches(value, item) for item in types):
                    _fail(path + ".enum", "enum value does not match declared type")
            if len({repr(item) for item in enum}) != len(enum):
                _fail(path + ".enum", "duplicate enum value")
            enum_tuple = tuple(enum)
        rules.append(PrimitiveRule(name, types, enum_tuple))
    return ContractAST("structured-output.v1", tuple(rules))
