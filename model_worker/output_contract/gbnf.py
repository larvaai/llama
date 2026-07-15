from __future__ import annotations

import json
import re

from .ast import ContractAST, PrimitiveRule


def _terminal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _value(rule: PrimitiveRule) -> str:
    if rule.enum is not None:
        return " | ".join(_terminal(json.dumps(item, ensure_ascii=False, separators=(",", ":"))) for item in rule.enum)
    mapping = {"string": "json-string", "integer": "json-integer", "boolean": '("true" | "false")', "null": '"null"'}
    options = [mapping[item] for item in rule.types]
    return options[0] if len(options) == 1 else "(" + " | ".join(options) + ")"


def compile_gbnf(contract: ContractAST) -> str:
    members, rules = [], []
    for index, rule in enumerate(contract.properties):
        rule_name = "value-%d-%s" % (index, re.sub(r"[^a-zA-Z0-9-]", "-", rule.name).strip("-").lower() or "field")
        members.append(f'{_terminal(json.dumps(rule.name, ensure_ascii=False))} ws ":" ws {rule_name}')
        rules.append(f"{rule_name} ::= {_value(rule)}")
    root = 'root ::= "{" ws ' + ' "," ws '.join(members) + ' "}" ws'
    common = ['json-integer ::= "-"? ("0" | [1-9] [0-9]*)', 'json-string ::= "\\\"" json-char* "\\\""', 'json-char ::= [^"\\\\\\x7F\\x00-\\x1F] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F]{4})', 'ws ::= [ \\t\\n\\r]*']
    return "\n".join([root, *rules, *common, ""])
