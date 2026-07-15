from __future__ import annotations

import math
from typing import Any

from .errors import ERROR_HTTP_STATUS, WorkerError
from .metrics import Metrics
from .request_registry import Lifecycle, RequestRecord, TERMINAL


_TIMING_METRICS = ("prompt_decode_ms", "generation_ms", "total_ms")
_USAGE_METRICS = (
    "prompt_tokens",
    "reasoning_tokens",
    "final_tokens",
    "sampled_tokens",
    "context_headroom",
)


def terminal_error_class(record: RequestRecord) -> str | None:
    if record.lifecycle == Lifecycle.COMPLETED:
        return None
    if isinstance(record.error, WorkerError):
        return record.error.code
    if type(record.error) is str and record.error in ERROR_HTTP_STATUS:
        return record.error
    return "worker_crashed"


def _payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "as_dict"):
        result = result.as_dict()
    return result if isinstance(result, dict) else {}


def _finite_number(value: Any) -> float | None:
    if type(value) not in {int, float}:
        return None
    observed = float(value)
    return observed if math.isfinite(observed) else None


class TerminalMetricsObserver:
    """Reconcile request metrics at the lifecycle boundary, independent of HTTP."""

    def __init__(self, metrics: Metrics) -> None:
        self.metrics = metrics

    def __call__(self, record: RequestRecord) -> None:
        if record.lifecycle not in TERMINAL:
            raise ValueError("terminal metrics require a terminal request record")

        labels = {"termination": record.lifecycle.value.lower()}
        error_class = terminal_error_class(record)
        if error_class is not None:
            labels["error_class"] = error_class
        self.metrics.inc("requests_total", **labels)

        queued_at = record.timestamps.get(Lifecycle.QUEUED.value)
        terminal_at = record.timestamps.get(record.lifecycle.value)
        if queued_at is not None and terminal_at is not None:
            running_at = record.timestamps.get(Lifecycle.RUNNING.value, terminal_at)
            self.metrics.observe(
                "queue_wait_ms",
                max(0.0, running_at - queued_at) * 1000,
            )

        if record.lifecycle != Lifecycle.COMPLETED:
            return
        payload = _payload(record.result)
        for container_name, metric_names in (
            ("timing", _TIMING_METRICS),
            ("usage", _USAGE_METRICS),
        ):
            container = payload.get(container_name)
            if not isinstance(container, dict):
                continue
            for name in metric_names:
                observed = _finite_number(container.get(name))
                if observed is not None:
                    self.metrics.observe(name, observed)
