from __future__ import annotations

from .contracts import GenerateRequest


def contract_instruction(request: GenerateRequest, property_names: tuple[str, ...]) -> str:
    fields = ", ".join(property_names)
    suffix = f"\nField semantics: {request.output_contract.instructions}" if request.output_contract.instructions else ""
    return f"Return exactly one JSON object matching structured-output.v1. Canonical fields: {fields}.{suffix}"
