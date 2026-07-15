from __future__ import annotations

import json
import math
import random

import pytest

from model_worker.errors import WorkerError
from model_worker.output_contract import compile_gbnf, parse_contract, validate_output
from model_worker.strict_json import loads


def schema(spec=None, *, name="value"):
    return {
        "type": "object",
        "properties": {name: spec or {"type": "string"}},
        "required": [name],
        "additionalProperties": False,
    }


@pytest.mark.parametrize(
    ("candidate", "path"),
    [
        (schema({"type": "string", "maxLength": 3}), "$.properties.value"),
        (schema({"enum": ["x"]}), "$.properties.value.type"),
        (schema({"type": "array"}), "$.properties.value.type"),
        (schema({"type": "object"}), "$.properties.value.type"),
        (schema({"type": "string", "pattern": "x"}), "$.properties.value"),
        ({**schema(), "title": "ignored?"}, "$.title"),
        ({**schema(), "additionalProperties": True}, "$"),
        ({**schema(), "required": []}, "$.required"),
    ],
)
def test_unsupported_contracts_fail_closed_with_location(candidate, path):
    with pytest.raises(WorkerError) as captured:
        parse_contract(candidate)
    assert captured.value.code == "unsupported_contract"
    assert len(captured.value.details) == 1
    assert captured.value.details[0].path == path


@pytest.mark.parametrize(
    "spec",
    [
        {"type": []},
        {"type": ["string", "string"]},
        {"type": ["string", "integer"]},
        {"type": ["string", "null", "integer"]},
        {"type": "string", "enum": []},
        {"type": "integer", "enum": [True]},
        {"type": "string", "enum": ["x", "x"]},
        {"type": "integer", "enum": [math.inf]},
    ],
)
def test_invalid_type_and_enum_definitions_are_rejected(spec):
    with pytest.raises(WorkerError) as captured:
        parse_contract(schema(spec))
    assert captured.value.code == "unsupported_contract"


def test_property_count_and_shape_bounds_are_enforced():
    with pytest.raises(WorkerError):
        parse_contract("not-an-object")
    with pytest.raises(WorkerError):
        parse_contract({"type": "object", "properties": {}, "required": [], "additionalProperties": False})
    with pytest.raises(WorkerError):
        parse_contract(
            {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"], "additionalProperties": False},
            max_properties=1,
        )
    with pytest.raises(WorkerError):
        parse_contract({**schema(), "required": ["value", "value"]})
    with pytest.raises(WorkerError):
        parse_contract({"type": "object", "properties": {1: {"type": "string"}}, "required": [1], "additionalProperties": False})
    with pytest.raises(WorkerError):
        parse_contract({"type": "object", "properties": {"value": "not-a-rule"}, "required": ["value"], "additionalProperties": False})


def test_enum_type_canonical_validation_and_gbnf_serialization():
    ast = parse_contract(schema({"type": ["string", "null"], "enum": ["ok", None]}, name='résult"'))
    assert validate_output({'résult"': "ok"}, ast) == []
    assert validate_output({'résult"': None}, ast) == []
    assert validate_output({'résult"': 1}, ast)
    grammar = compile_gbnf(ast)
    assert "value-0-r-sult" in grammar
    assert json.dumps(json.dumps('résult"', ensure_ascii=False), ensure_ascii=False) in grammar
    assert json.dumps(json.dumps("ok", ensure_ascii=False, separators=(",", ":")), ensure_ascii=False) in grammar


def test_validator_requires_exact_object_keys_and_canonical_order():
    ast = parse_contract(
        {
            "type": "object",
            "properties": {"first": {"type": "integer"}, "second": {"type": "boolean"}},
            "required": ["first", "second"],
            "additionalProperties": False,
        }
    )
    assert validate_output({"first": 1, "second": True}, ast) == []
    assert validate_output("not-an-object", ast)[0].path == "$"
    assert validate_output({"second": True, "first": 1}, ast)[0].path == "$"
    assert validate_output({"first": 1}, ast)[0].path == "$"
    assert validate_output({"first": 1, "second": True, "extra": None}, ast)[0].path == "$"
    assert validate_output({"first": True, "second": True}, ast)[0].path == "$.first"


@pytest.mark.parametrize(
    "raw",
    [
        '{"a":1,"a":2}',
        '{"a":1} trailing',
        '{"a":NaN}',
        '{"a":Infinity}',
        b'\xff',
        b'"\\uD800"',
        b'{"\\uDFFF":"value"}',
    ],
)
def test_strict_json_rejects_duplicate_trailing_nonfinite_and_invalid_utf8(raw):
    with pytest.raises(WorkerError) as captured:
        loads(raw)
    assert captured.value.code == "invalid_request"


def test_strict_json_enforces_byte_limit_before_decode():
    with pytest.raises(WorkerError) as captured:
        loads("✓", too_large=2)
    assert captured.value.code == "request_too_large"


def test_ten_thousand_generated_values_match_the_primitive_validator_oracle():
    rng = random.Random(7)
    options = [
        ("string", lambda: rng.choice(["", "x", "✓"])),
        ("integer", lambda: rng.randint(-99, 99)),
        ("boolean", lambda: rng.choice([True, False])),
        ("null", lambda: None),
    ]
    candidates = ["x", 1, True, None]
    for _ in range(10_000):
        declared, factory = rng.choice(options)
        ast = parse_contract(schema({"type": declared}))
        good = factory()
        assert validate_output({"value": good}, ast) == []
        for candidate in candidates:
            expected_valid = type(candidate) is type(good)
            assert (validate_output({"value": candidate}, ast) == []) is expected_valid
