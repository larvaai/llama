from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ERROR_HTTP_STATUS = {
    "invalid_request": 400,
    "request_too_large": 413,
    "unsupported_contract": 422,
    "context_overflow": 422,
    "queue_full": 429,
    "slow_consumer": 429,
    "queue_timeout": 408,
    "cancelled": 409,
    "deadline_exceeded": 504,
    "protocol_violation": 422,
    "output_invalid": 422,
    "decode_failed": 502,
    "worker_crashed": 502,
    "worker_not_ready": 503,
    "shutdown": 503,
}

RETRYABLE = {
    "queue_full", "queue_timeout", "slow_consumer", "deadline_exceeded", "decode_failed",
    "worker_crashed", "worker_not_ready", "shutdown",
}


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    path: str
    message: str


class WorkerError(Exception):
    def __init__(self, code: str, message: str, *, details: list[ErrorDetail] | None = None):
        if code not in ERROR_HTTP_STATUS:
            raise ValueError(f"unknown worker error code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or []

    @property
    def http_status(self) -> int:
        return ERROR_HTTP_STATUS[self.code]

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": [{"path": item.path, "message": item.message} for item in self.details],
        }
