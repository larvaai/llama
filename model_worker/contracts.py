from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import ErrorDetail, WorkerError

PROTOCOL_VERSION = "model-worker.v1"
ALLOWED_ROLES = {"system", "user", "assistant"}


def _exact_object(value: Any, path: str, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if type(value) is not dict:
        raise WorkerError("invalid_request", f"{path} must be an object")
    unknown = set(value) - allowed
    missing = required - set(value)
    details = [ErrorDetail(f"{path}.{key}", "unknown field") for key in sorted(unknown)]
    details += [ErrorDetail(f"{path}.{key}", "required field missing") for key in sorted(missing)]
    if details:
        raise WorkerError("invalid_request", f"invalid fields at {path}", details=details)
    return value


def _strict_int(value: Any, path: str, low: int = 1) -> int:
    if type(value) is not int or value < low:
        raise WorkerError("invalid_request", f"{path} must be an integer >= {low}")
    return value


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class OutputContract:
    version: str
    schema: dict[str, Any]
    instructions: str = ""


@dataclass(frozen=True, slots=True)
class Limits:
    reasoning_tokens: int
    final_tokens: int
    total_tokens: int
    queue_timeout_ms: int
    execution_timeout_ms: int


@dataclass(frozen=True, slots=True)
class StreamOptions:
    enabled: bool = False
    include_reasoning: bool = False


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    protocol_version: str
    model_id: str
    messages: tuple[Message, ...]
    output_contract: OutputContract
    limits: Limits
    stream: StreamOptions = field(default_factory=StreamOptions)
    client_request_id: str | None = None

    @classmethod
    def parse(cls, value: Any) -> "GenerateRequest":
        body = _exact_object(value, "$", {"protocol_version", "model_id", "messages", "output_contract", "limits", "stream", "metadata"}, {"protocol_version", "model_id", "messages", "output_contract", "limits"})
        if body["protocol_version"] != PROTOCOL_VERSION:
            raise WorkerError("invalid_request", f"protocol_version must be {PROTOCOL_VERSION}")
        if type(body["model_id"]) is not str or not body["model_id"]:
            raise WorkerError("invalid_request", "model_id must be a non-empty string")
        if type(body["messages"]) is not list or not body["messages"]:
            raise WorkerError("invalid_request", "messages must be a non-empty array")
        messages = []
        for index, item in enumerate(body["messages"]):
            obj = _exact_object(item, f"$.messages[{index}]", {"role", "content"}, {"role", "content"})
            if obj["role"] not in ALLOWED_ROLES:
                raise WorkerError("invalid_request", f"invalid role at $.messages[{index}].role")
            if type(obj["content"]) is not str or not obj["content"]:
                raise WorkerError("invalid_request", f"content must be a non-empty string at $.messages[{index}]")
            messages.append(Message(obj["role"], obj["content"]))
        contract = _exact_object(body["output_contract"], "$.output_contract", {"version", "schema", "instructions"}, {"version", "schema"})
        if contract["version"] != "structured-output.v1" or type(contract["schema"]) is not dict:
            raise WorkerError("invalid_request", "invalid output contract version or schema type")
        instructions = contract.get("instructions", "")
        if type(instructions) is not str:
            raise WorkerError("invalid_request", "output_contract.instructions must be a string")
        raw_limits = _exact_object(body["limits"], "$.limits", {"reasoning_tokens", "final_tokens", "total_tokens", "queue_timeout_ms", "execution_timeout_ms"}, {"reasoning_tokens", "final_tokens", "total_tokens", "queue_timeout_ms", "execution_timeout_ms"})
        limits = Limits(**{name: _strict_int(raw_limits[name], f"$.limits.{name}") for name in raw_limits})
        if limits.reasoning_tokens > limits.total_tokens or limits.final_tokens > limits.total_tokens or limits.total_tokens > limits.reasoning_tokens + limits.final_tokens:
            raise WorkerError("invalid_request", "token budgets violate reasoning/final/total invariants")
        stream_obj = _exact_object(body.get("stream", {}), "$.stream", {"enabled", "include_reasoning"}, set())
        for name in ("enabled", "include_reasoning"):
            if name in stream_obj and type(stream_obj[name]) is not bool:
                raise WorkerError("invalid_request", f"$.stream.{name} must be a boolean")
        metadata = _exact_object(body.get("metadata", {}), "$.metadata", {"client_request_id"}, set())
        client_id = metadata.get("client_request_id")
        if client_id is not None and type(client_id) is not str:
            raise WorkerError("invalid_request", "metadata.client_request_id must be a string")
        return cls(
            PROTOCOL_VERSION, body["model_id"], tuple(messages),
            OutputContract(contract["version"], contract["schema"], instructions), limits,
            StreamOptions(stream_obj.get("enabled", False), stream_obj.get("include_reasoning", False)), client_id,
        )


@dataclass(frozen=True, slots=True)
class GenerateResult:
    request_id: str
    attempt_id: str
    termination: str
    protocol_valid: bool
    output_valid: bool
    output: Any
    usage: dict[str, int]
    timing: dict[str, int | float]
    model: dict[str, Any]
    error: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        if self.termination == "completed" and not (self.protocol_valid and self.output_valid):
            raise ValueError("completed requires protocol_valid and output_valid")
        return {"protocol_version": PROTOCOL_VERSION, **{name: getattr(self, name) for name in self.__dataclass_fields__}}
