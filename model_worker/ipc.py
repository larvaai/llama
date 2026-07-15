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
    started: bool = False
    final_phase: bool = False
    terminal: bool = False

    def verify(self, raw: bytes | str | dict[str, Any]) -> dict[str, Any]:
        try:
            frame = loads(raw) if not isinstance(raw, dict) else raw
        except WorkerError as exc:
            raise WorkerError("worker_crashed", "malformed IPC JSON") from exc
        if type(frame) is not dict or frame.get("protocol_version") != IPC_VERSION:
            raise WorkerError("worker_crashed", "invalid IPC version or frame")
        if frame.get("request_id") != self.request_id or frame.get("attempt_id") != self.attempt_id:
            raise WorkerError("worker_crashed", "IPC request/attempt identity mismatch")
        if type(frame.get("sequence")) is not int or frame["sequence"] != self.next_sequence:
            raise WorkerError("worker_crashed", "IPC sequence desynchronization")
        if frame.get("type") not in FRAME_TYPES:
            raise WorkerError("worker_crashed", "unknown IPC frame type")
        self._verify_request_frame(frame)
        self.next_sequence += 1
        return frame

    def _verify_request_frame(self, frame: dict[str, Any]) -> None:
        frame_type = frame["type"]
        if frame_type == "ready":
            raise WorkerError("worker_crashed", "unexpected ready frame during request")
        if self.terminal:
            raise WorkerError("worker_crashed", "IPC frame received after terminal")
        if not self.started:
            if frame_type != "started":
                raise WorkerError("worker_crashed", "IPC request did not start with started frame")
            self.started = True
            return
        if frame_type == "started":
            raise WorkerError("worker_crashed", "duplicate IPC started frame")
        if frame_type == "phase":
            if frame.get("phase") != "final" or self.final_phase:
                raise WorkerError("worker_crashed", "invalid IPC phase transition")
            self.final_phase = True
            return
        if frame_type == "final_delta":
            if not self.final_phase or type(frame.get("delta")) is not str or not frame["delta"]:
                raise WorkerError("worker_crashed", "invalid IPC final delta")
            return
        if frame_type == "progress":
            if (
                frame.get("phase") not in {"prompt_decode", "reasoning", "final"}
                or type(frame.get("tokens")) is not int
                or frame["tokens"] < 0
            ):
                raise WorkerError("worker_crashed", "invalid IPC progress payload")
            return
        if frame_type == "heartbeat":
            if type(frame.get("sampled_tokens")) is not int or frame["sampled_tokens"] < 0:
                raise WorkerError("worker_crashed", "invalid IPC heartbeat payload")
            return
        if frame_type == "completed":
            if (
                type(frame.get("final_text")) is not str
                or type(frame.get("usage")) is not dict
                or type(frame.get("timing")) is not dict
            ):
                raise WorkerError("worker_crashed", "invalid IPC completed payload")
            if not self.final_phase:
                raise WorkerError("worker_crashed", "IPC completed before final phase")
            self.terminal = True
            return
        if frame_type == "failed":
            if type(frame.get("error")) is not str or type(frame.get("detail")) is not str:
                raise WorkerError("worker_crashed", "invalid IPC failed payload")
            self.terminal = True


def encode_frame(frame_type: str, request_id: str, attempt_id: str, sequence: int, **payload: Any) -> str:
    if frame_type not in FRAME_TYPES and frame_type not in {"generate", "cancel", "shutdown"}:
        raise ValueError("unknown frame type")
    return json.dumps({"protocol_version": IPC_VERSION, "type": frame_type, "request_id": request_id, "attempt_id": attempt_id, "sequence": sequence, **payload}, ensure_ascii=False, separators=(",", ":"))
