"""Compile a deliberately small, deterministic JSON Schema subset to llama.cpp GBNF."""

import argparse
import json
import re
from pathlib import Path


SUPPORTED_TYPES = {"string", "integer", "boolean", "null"}


def terminal(text: str) -> str:
    """GBNF terminal containing exactly text."""
    return json.dumps(text, ensure_ascii=False)


def rule_name(property_name: str, index: int) -> str:
    safe = re.sub(r"[^a-zA-Z0-9-]", "-", property_name).strip("-").lower() or "field"
    return f"value-{index}-{safe}"


def value_expression(spec: dict) -> str:
    if "enum" in spec:
        values = spec["enum"]
        if not isinstance(values, list) or not values:
            raise ValueError("enum must be a non-empty array")
        return " | ".join(terminal(json.dumps(value, ensure_ascii=False, separators=(",", ":"))) for value in values)

    types = spec.get("type")
    if isinstance(types, str):
        types = [types]
    if not isinstance(types, list) or not types or not set(types) <= SUPPORTED_TYPES:
        raise ValueError(f"unsupported type: {types!r}")
    mapping = {
        "string": "json-string",
        "integer": "json-integer",
        "boolean": '("true" | "false")',
        "null": '"null"',
    }
    expressions = [mapping[item] for item in types]
    return expressions[0] if len(expressions) == 1 else "(" + " | ".join(expressions) + ")"


def compile_schema(schema: dict) -> str:
    if schema.get("type") != "object":
        raise ValueError("root type must be object")
    if schema.get("additionalProperties") is not False:
        raise ValueError("additionalProperties must be false")
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not properties:
        raise ValueError("properties must be a non-empty object")
    if required != list(properties):
        raise ValueError("Phase D subset requires every property, in properties order")

    members = []
    value_rules = []
    for index, (name, spec) in enumerate(properties.items()):
        value_rule = rule_name(name, index)
        members.append(f"{terminal(json.dumps(name, ensure_ascii=False))} ws \":\" ws {value_rule}")
        value_rules.append(f"{value_rule} ::= {value_expression(spec)}")

    root_parts = ['"{" ws']
    for index, member in enumerate(members):
        if index:
            root_parts.append('"," ws')
        root_parts.append(member)
    root_parts.append('"}" ws')

    common = [
        'json-integer ::= "-"? ("0" | [1-9] [0-9]*)',
        'json-string ::= "\\\"" json-char* "\\\""',
        'json-char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F]{4})',
        'ws ::= [ \\t\\n\\r]*',
    ]
    return "\n".join(["root ::= " + " ".join(root_parts), *value_rules, *common, ""])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("schema", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    grammar = compile_schema(schema)
    args.output.write_text(grammar, encoding="utf-8")
    print(grammar, end="")


if __name__ == "__main__":
    main()
