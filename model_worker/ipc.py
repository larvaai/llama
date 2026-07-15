from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .errors import WorkerError
from .strict_json import loads

IPC_VERSION = "model-worker-ipc.v1"
FRAME_TYPES = {"ready", "started", "phase", "final_delta", "heartbeat", "progress", "completed", "failed"}


@dataclass(slots=True)
class FrameVerifier:
    request_id: str
    attempt_id: str
    next_sequence: int = 0

    def verify(self, raw: bytes | str | dict[str, Any]) -> dict[str, Any]:
        frame = loads(raw) if not isinstance(raw, dict) else raw
        if type(frame) is not dict or frame.get("protocol_version") != IPC_VERSION:
            raise WorkerError("worker_crashed", "invalid IPC version or frame")
        if frame.get("request_id") != self.request_id or frame.get("attempt_id") != self.attempt_id:
            raise WorkerError("worker_crashed", "IPC request/attempt identity mismatch")
        if type(frame.get("sequence")) is not int or frame["sequence"] != self.next_sequence:
            raise WorkerError("worker_crashed", "IPC sequence desynchronization")
        if frame.get("type") not in FRAME_TYPES:
            raise WorkerError("worker_crashed", "unknown IPC frame type")
        self.next_sequence += 1
        return frame


def encode_frame(frame_type: str, request_id: str, attempt_id: str, sequence: int, **payload: Any) -> str:
    if frame_type not in FRAME_TYPES and frame_type not in {"generate", "cancel", "shutdown"}:
        raise ValueError("unknown frame type")
    return json.dumps({"protocol_version": IPC_VERSION, "type": frame_type, "request_id": request_id, "attempt_id": attempt_id, "sequence": sequence, **payload}, ensure_ascii=False, separators=(",", ":"))
