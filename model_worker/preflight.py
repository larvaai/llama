from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import GenerateRequest
from .manifest import ModelManifest, enforce_request_envelope
from .output_contract import compile_gbnf, parse_contract
from .output_contract.ast import ContractAST


@dataclass(frozen=True, slots=True)
class PreflightedRequest:
    request: GenerateRequest
    contract: ContractAST
    grammar: str

    @property
    def limits(self):
        return self.request.limits


def preflight(value: Any, manifest: ModelManifest) -> PreflightedRequest:
    request = GenerateRequest.parse(value)
    enforce_request_envelope(request, manifest)
    contract = parse_contract(request.output_contract.schema, max_properties=manifest.limits["max_properties"])
    grammar = compile_gbnf(contract)
    return PreflightedRequest(request, contract, grammar)
