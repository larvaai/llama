from __future__ import annotations

import hashlib
import json

from .contracts import GenerateRequest, Message

PROMPT_CONTRACT_VERSION = "model-worker-prompt.v1"
BEGIN_CONTRACT = "<model-worker-output-contract-v1>"
END_CONTRACT = "</model-worker-output-contract-v1>"


def contract_instruction(request: GenerateRequest, property_names: tuple[str, ...]) -> str:
    fields = ", ".join(property_names)
    lines = [
        BEGIN_CONTRACT,
        "Return exactly one JSON object matching structured-output.v1.",
        f"Canonical fields: {fields}.",
    ]
    if request.output_contract.instructions:
        lines.extend(("Field semantics:", request.output_contract.instructions))
    lines.append(END_CONTRACT)
    return "\n".join(lines)


def build_model_messages(
    request: GenerateRequest,
    property_names: tuple[str, ...],
) -> tuple[Message, ...]:
    instruction = contract_instruction(request, property_names)
    return (Message("system", instruction), *request.messages)


def model_prompt_hash(messages: tuple[Message, ...]) -> str:
    canonical = json.dumps(
        {
            "version": PROMPT_CONTRACT_VERSION,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()
