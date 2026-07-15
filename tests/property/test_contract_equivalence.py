from __future__ import annotations

import json

from hypothesis import given, strategies as st

from model_worker.output_contract import compile_gbnf, parse_contract, validate_output


PRIMITIVES = st.one_of(st.text(), st.integers(), st.booleans(), st.none())


def declared_type(value):
    return "null" if value is None else "boolean" if type(value) is bool else "integer" if type(value) is int else "string"


@given(PRIMITIVES)
def test_normalized_contract_accepts_every_declared_primitive(value):
    declared = declared_type(value)
    schema = {
        "type": "object",
        "properties": {"value": {"type": declared}},
        "required": ["value"],
        "additionalProperties": False,
    }
    contract = parse_contract(schema)
    assert validate_output({"value": value}, contract) == []
    grammar = compile_gbnf(contract)
    assert grammar.startswith("root ::=")
    assert "value-0-value ::= " in grammar


@given(PRIMITIVES, PRIMITIVES)
def test_validator_matches_exact_python_json_primitive_types(declared_value, candidate):
    schema = {
        "type": "object",
        "properties": {"value": {"type": declared_type(declared_value)}},
        "required": ["value"],
        "additionalProperties": False,
    }
    contract = parse_contract(schema)
    expected_valid = type(candidate) is type(declared_value)
    assert (validate_output({"value": candidate}, contract) == []) is expected_valid


@given(st.lists(st.text(min_size=1).filter(lambda name: "\x00" not in name), min_size=1, max_size=8, unique=True))
def test_compiler_emits_each_property_once_in_canonical_order(names):
    schema = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in names},
        "required": names,
        "additionalProperties": False,
    }
    contract = parse_contract(schema)
    grammar = compile_gbnf(contract)
    cursor = -1
    for name in names:
        encoded = json.dumps(json.dumps(name, ensure_ascii=False), ensure_ascii=False)
        next_cursor = grammar.index(encoded)
        assert next_cursor > cursor
        cursor = next_cursor
