from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .contracts import CompiledToolCall, SideEffectState, ToolInvocationLifecycle
from .errors import AgentRuntimeError


@dataclass(frozen=True, slots=True)
class RawToolResult:
    invocation_id: str
    status: ToolInvocationLifecycle
    payload: bytes
    summary: str
    side_effect_state: SideEffectState = SideEffectState.NONE
    error_code: str | None = None
    truncated: bool = False


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class AllowlistedReadOnlyExecutor:
    def __init__(self, *, timeout_ms: int = 5_000, result_byte_cap: int = 1_048_576) -> None:
        if timeout_ms <= 0 or result_byte_cap <= 0:
            raise ValueError("executor limits must be positive")
        self.timeout_ms = timeout_ms
        self.result_byte_cap = result_byte_cap
        self.dispatch_count = 0
        self._completed: dict[str, RawToolResult] = {}

    def execute(self, call: CompiledToolCall, *, cancellation: CancellationToken | None = None) -> RawToolResult:
        completed = self._completed.get(call.idempotency_key)
        if completed is not None:
            return completed
        if cancellation is not None and cancellation.cancelled:
            raise AgentRuntimeError("cancelled_pre_dispatch", "tool was cancelled before dispatch")
        started = time.monotonic()
        self.dispatch_count += 1
        if call.tool_id == "read_file":
            payload = Path(str(call.native_arguments["path"])).read_bytes()
            summary = f"read {len(payload)} bytes"
        elif call.tool_id == "search_text":
            payload, count = self._search(call.native_arguments, started, cancellation)
            summary = f"found {count} matching lines"
        else:
            raise AgentRuntimeError("executor_unavailable", "tool is not executable")
        if cancellation is not None and cancellation.cancelled:
            raise AgentRuntimeError("cancelled_post_dispatch", "tool was cancelled after dispatch")
        if (time.monotonic() - started) * 1000 > self.timeout_ms:
            raise AgentRuntimeError("post_dispatch_timeout", "tool timed out after dispatch")
        result = RawToolResult(
            invocation_id=call.invocation_id,
            status=ToolInvocationLifecycle.SUCCEEDED,
            payload=payload[: self.result_byte_cap],
            summary=summary,
            truncated=len(payload) > self.result_byte_cap,
        )
        self._completed[call.idempotency_key] = result
        return result

    def _search(
        self,
        arguments: Mapping[str, Any],
        started: float,
        cancellation: CancellationToken | None,
    ) -> tuple[bytes, int]:
        root = Path(str(arguments["root"]))
        pattern = str(arguments["pattern"])
        lines: list[str] = []
        for path in sorted(root.rglob("*")):
            if cancellation is not None and cancellation.cancelled:
                raise AgentRuntimeError("cancelled_post_dispatch", "search cancelled")
            if (time.monotonic() - started) * 1000 > self.timeout_ms:
                raise AgentRuntimeError("post_dispatch_timeout", "search timed out")
            if not path.is_file() or path.is_symlink():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if pattern in line:
                    lines.append(f"{path.relative_to(root).as_posix()}:{number}:{line}")
        return "\n".join(lines).encode("utf-8"), len(lines)
