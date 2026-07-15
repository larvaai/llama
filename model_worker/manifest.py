from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .errors import WorkerError
from .strict_json import loads


class CapabilityProvider(Protocol):
    def tokenize(self, text: str) -> list[int]: ...
    def has_chat_template(self) -> bool: ...
    def model_context(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ModelManifest:
    path: Path
    raw: dict[str, Any]
    digest: str

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def context(self) -> dict[str, int]:
        return self.raw["context"]

    @property
    def limits(self) -> dict[str, int]:
        return self.raw["limits"]

    @property
    def reasoning(self) -> dict[str, Any]:
        return self.raw["reasoning"]


REQUIRED_TOP = {"manifest_version", "id", "gguf_path", "gguf_sha256", "runtime_build", "runtime", "context", "gpu", "sampling", "reasoning", "limits"}


def load_manifest(path: Path, *, verify_files: bool = True) -> ModelManifest:
    raw_bytes = path.read_bytes()
    data = loads(raw_bytes)
    if type(data) is not dict or set(data) != REQUIRED_TOP:
        raise WorkerError("worker_not_ready", "manifest has missing or unknown top-level fields")
    if data["manifest_version"] != "model-manifest.v1" or data["runtime_build"] != "b10012":
        raise WorkerError("worker_not_ready", "unsupported manifest or runtime build")
    if type(data["id"]) is not str or not data["id"]:
        raise WorkerError("worker_not_ready", "manifest id must be non-empty")
    context = data["context"]
    if type(context) is not dict or any(type(context.get(k)) is not int or context[k] <= 0 for k in ("n_ctx", "n_batch", "n_ubatch", "training_context")):
        raise WorkerError("worker_not_ready", "invalid context envelope")
    if context["n_ubatch"] > context["n_batch"] or context["n_batch"] > context["n_ctx"]:
        raise WorkerError("worker_not_ready", "n_ubatch <= n_batch <= n_ctx is required")
    if context["n_ctx"] > context["training_context"] and not context.get("rope_scaling"):
        raise WorkerError("worker_not_ready", "context above training context requires explicit rope_scaling")
    if data.get("sampling") != {"profile": "greedy-v1"}:
        raise WorkerError("worker_not_ready", "only greedy-v1 sampling is supported")
    reasoning = data["reasoning"]
    if type(reasoning) is not dict or reasoning.get("mode") not in {"none", "required_marker_sequence"}:
        raise WorkerError("worker_not_ready", "unsupported reasoning mode")
    if reasoning["mode"] == "required_marker_sequence" and any(type(reasoning.get(k)) is not str or not reasoning[k] for k in ("start_text", "end_text")):
        raise WorkerError("worker_not_ready", "reasoning markers must be non-empty strings")
    if verify_files:
        _verify_sha(Path(data["gguf_path"]), data["gguf_sha256"], "model")
        _verify_sha(Path(data["runtime"]["directory"]) / "llama.dll", data["runtime"]["llama_dll_sha256"], "runtime")
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return ModelManifest(path.resolve(), data, "sha256:" + hashlib.sha256(canonical).hexdigest())


def _verify_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise WorkerError("worker_not_ready", f"{label} file not found")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != "sha256:" + digest:
        raise WorkerError("worker_not_ready", f"{label} hash mismatch")


def verify_capabilities(manifest: ModelManifest, provider: CapabilityProvider) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not provider.has_chat_template():
        raise WorkerError("worker_not_ready", "model chat template is missing")
    if provider.model_context() < manifest.context["n_ctx"]:
        raise WorkerError("worker_not_ready", "declared context exceeds model capability")
    if manifest.reasoning["mode"] == "none":
        return (), ()
    start = tuple(provider.tokenize(manifest.reasoning["start_text"]))
    end = tuple(provider.tokenize(manifest.reasoning["end_text"]))
    if not start or not end or start == end:
        raise WorkerError("worker_not_ready", "reasoning marker token sequences are invalid")
    return start, end


def enforce_request_envelope(request: Any, manifest: ModelManifest) -> None:
    if request.model_id != manifest.id:
        raise WorkerError("invalid_request", "model_id does not match resident model")
    limits = manifest.limits
    if len(request.messages) > limits["max_messages"]:
        raise WorkerError("request_too_large", "too many messages")
    total = 0
    for message in request.messages:
        size = len(message.content.encode("utf-8"))
        total += size
        if size > limits["message_bytes"]:
            raise WorkerError("request_too_large", "message exceeds byte limit")
    if total > limits["input_bytes"]:
        raise WorkerError("request_too_large", "input exceeds byte limit")
    if len(request.output_contract.instructions.encode("utf-8")) > limits["instructions_bytes"]:
        raise WorkerError("request_too_large", "contract instructions exceed byte limit")
    schema_size = len(json.dumps(request.output_contract.schema, ensure_ascii=False).encode("utf-8"))
    if schema_size > limits["schema_bytes"]:
        raise WorkerError("request_too_large", "schema exceeds byte limit")
    if request.client_request_id and len(request.client_request_id.encode("utf-8")) > limits["client_request_id_bytes"]:
        raise WorkerError("request_too_large", "client_request_id exceeds byte limit")
    pairs = ((request.limits.reasoning_tokens, "max_reasoning_tokens"), (request.limits.final_tokens, "max_final_tokens"), (request.limits.total_tokens, "max_total_tokens"))
    if any(actual > limits[name] for actual, name in pairs):
        raise WorkerError("invalid_request", "requested token budget exceeds manifest policy")
