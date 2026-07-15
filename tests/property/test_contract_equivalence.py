from __future__ import annotations

from hypothesis import given, strategies as st

from model_worker.output_contract import compile_gbnf, parse_contract, validate_output


@given(st.one_of(st.text(), st.integers(), st.booleans(), st.none()))
def test_normalized_contract_validator_accepts_declared_primitive(value):
    declared = "null" if value is None else "boolean" if type(value) is bool else "integer" if type(value) is int else "string"
    schema={"type":"object","properties":{"value":{"type":declared}},"required":["value"],"additionalProperties":False}
    contract=parse_contract(schema)
    assert validate_output({"value":value},contract)==[]
    assert compile_gbnf(contract).startswith("root ::=")
