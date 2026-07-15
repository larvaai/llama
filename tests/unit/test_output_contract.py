from __future__ import annotations

import json
import random

import pytest

from model_worker.errors import WorkerError
from model_worker.output_contract import compile_gbnf, parse_contract, validate_output
from model_worker.strict_json import loads


def schema(spec=None):
    return {"type": "object", "properties": {"value": spec or {"type": "string"}}, "required": ["value"], "additionalProperties": False}


@pytest.mark.parametrize("spec", [{"type":"string","maxLength":3}, {"enum":["x"]}, {"type":"array"}, {"type":"object"}, {"type":"string","pattern":"x"}])
def test_unsupported_keywords_fail_closed(spec):
    with pytest.raises(WorkerError) as error: parse_contract(schema(spec))
    assert error.value.code == "unsupported_contract" and error.value.details


def test_enum_type_and_canonical_validation():
    ast = parse_contract(schema({"type":["string","null"], "enum":["ok", None]}))
    assert not validate_output({"value":"ok"}, ast)
    assert validate_output({"value":1}, ast)
    assert "value-0-value" in compile_gbnf(ast)


def test_strict_json_duplicate_trailing_and_nonfinite():
    for raw in ('{"a":1,"a":2}', '{"a":1} trailing', '{"a":NaN}'):
        with pytest.raises(WorkerError): loads(raw)


def test_ten_thousand_generated_schema_value_cases_have_no_validator_divergence():
    rng = random.Random(7)
    options = [("string", lambda: rng.choice(["", "x", "✓"])), ("integer", lambda: rng.randint(-99,99)), ("boolean", lambda: rng.choice([True,False])), ("null", lambda: None)]
    for _ in range(10_000):
        declared, factory = rng.choice(options)
        ast = parse_contract(schema({"type": declared}))
        good = {"value": factory()}
        assert validate_output(good, ast) == []
        bad_candidates = ["x", 1, True, None]
        bad = next(value for value in bad_candidates if type(value) is not type(good["value"]))
        assert validate_output({"value": bad}, ast)
        json.dumps(good, ensure_ascii=False)
