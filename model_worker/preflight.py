from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import GenerateRequest, Message
from .manifest import ModelManifest, enforce_message_envelope, enforce_request_envelope
from .output_contract import compile_gbnf, parse_contract
from .output_contract.ast import ContractAST
from .prompt import PROMPT_CONTRACT_VERSION, build_model_messages, model_prompt_hash


@dataclass(frozen=True, slots=True)
class PreflightedRequest:
    request: GenerateRequest
    contract: ContractAST
    grammar: str
    model_messages: tuple[Message, ...]
    prompt_hash: str
    prompt_version: str

    @property
    def limits(self):
        return self.request.limits


def preflight(value: Any, manifest: ModelManifest) -> PreflightedRequest:
    request = GenerateRequest.parse(value)
    enforce_request_envelope(request, manifest)
    contract = parse_contract(request.output_contract.schema, max_properties=manifest.limits["max_properties"])
    grammar = compile_gbnf(contract)
    model_messages = build_model_messages(
        request,
        tuple(rule.name for rule in contract.properties),
    )
    enforce_message_envelope(model_messages, manifest)
    return PreflightedRequest(
        request=request,
        contract=contract,
        grammar=grammar,
        model_messages=model_messages,
        prompt_hash=model_prompt_hash(model_messages),
        prompt_version=PROMPT_CONTRACT_VERSION,
    )
