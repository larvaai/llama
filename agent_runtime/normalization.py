from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from .contracts import AgentError, RetryScope, ToolResultEnvelope
from .errors import AgentRuntimeError
from .executor import RawToolResult

SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*([^\s]+)")


class ImmutableArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, payload: bytes) -> tuple[str, str]:
        digest_hex = hashlib.sha256(payload).hexdigest()
        digest = f"sha256:{digest_hex}"
        target = self.root / digest_hex[:2] / digest_hex
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            fd, temporary = tempfile.mkstemp(dir=target.parent, prefix="artifact-", suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest_hex:
            raise AgentRuntimeError("artifact_hash_mismatch", "artifact verification failed")
        return f"artifact://sha256/{digest_hex}", digest


class DeterministicResultNormalizer:
    def __init__(self, store: ImmutableArtifactStore, *, observation_byte_cap: int = 4_096) -> None:
        self.store = store
        self.observation_byte_cap = observation_byte_cap

    def normalize(self, raw: RawToolResult, *, tool_id: str, tool_version: str) -> ToolResultEnvelope:
        if not isinstance(raw, RawToolResult):
            raise AgentRuntimeError("result_schema_invalid", "executor result has invalid type")
        try:
            text = raw.payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise AgentRuntimeError("result_schema_invalid", "tool output is not UTF-8") from error
        redacted = SECRET_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
        payload = redacted.encode("utf-8")
        data_ref, data_hash = self.store.put(payload)
        bounded = payload[: self.observation_byte_cap]
        summary = raw.summary
        if bounded:
            preview = bounded.decode("utf-8", errors="ignore")
            summary = json.dumps({"summary": summary, "preview": preview}, ensure_ascii=False)
        truncated = raw.truncated or len(payload) > len(bounded)
        error = None
        if raw.status.value in {"failed", "unknown_outcome"}:
            error = AgentError(
                raw.error_code or "tool_failed",
                raw.summary,
                False,
                RetryScope.RECONCILE_ONLY if raw.status.value == "unknown_outcome" else RetryScope.NEVER,
                raw.side_effect_state,
            )
        return ToolResultEnvelope(
            invocation_id=raw.invocation_id,
            tool_id=tool_id,
            tool_version=tool_version,
            status=raw.status,
            summary=summary,
            data_ref=data_ref,
            data_hash=data_hash,
            truncated=truncated,
            side_effect_state=raw.side_effect_state,
            error=error,
            provenance="atomic_executor.v1",
        )
