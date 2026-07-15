from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import dataclass, field

from .errors import ERROR_HTTP_STATUS


HISTOGRAM_BUCKETS = (
    0.0,
    1.0,
    2.5,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1_000.0,
    2_500.0,
    5_000.0,
    10_000.0,
    30_000.0,
    60_000.0,
    120_000.0,
    300_000.0,
)

LABEL_VALUE_ALLOWLIST = {
    "termination": frozenset({"completed", "failed", "cancelled", "timed_out"}),
    "error_class": frozenset(ERROR_HTTP_STATUS),
    "phase": frozenset(
        {"queue", "prefill", "prompt_decode", "reasoning", "decode", "final"}
    ),
}


@dataclass(slots=True)
class _Histogram:
    buckets: list[int] = field(
        default_factory=lambda: [0] * len(HISTOGRAM_BUCKETS)
    )
    count: int = 0
    total: float = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        for index, upper_bound in enumerate(HISTOGRAM_BUCKETS):
            if value <= upper_bound:
                self.buckets[index] += 1


def _safe_labels(labels: dict[str, str]) -> tuple[tuple[str, str], ...]:
    safe = []
    for key, value in labels.items():
        allowed_values = LABEL_VALUE_ALLOWLIST.get(key)
        if allowed_values is None:
            continue
        safe.append(
            (key, value if type(value) is str and value in allowed_values else "other")
        )
    return tuple(sorted(safe))


def _bucket_label(value: float) -> str:
    return format(value, "g")


class Metrics:
    """Bounded in-process metrics; IDs and raw observations are never retained."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
        self._histograms: dict[str, _Histogram] = {}
        self._gauges: dict[str, float] = {}

    def inc(self, name: str, **labels: str) -> None:
        safe = _safe_labels(labels)
        with self._lock:
            self._counters[(name, safe)] += 1

    def observe(self, name: str, value: float) -> None:
        observed = float(value)
        if not math.isfinite(observed):
            raise ValueError("metric observations must be finite")
        with self._lock:
            histogram = self._histograms.get(name)
            if histogram is None:
                histogram = _Histogram()
                self._histograms[name] = histogram
            histogram.observe(observed)

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def render(self) -> str:
        with self._lock:
            lines = []
            for (name, labels), value in sorted(self._counters.items()):
                suffix = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""
                lines.append(f"model_worker_{name}{suffix} {value}")
            for name, value in sorted(self._gauges.items()):
                lines.append(f"model_worker_{name} {value}")
            for name, histogram in sorted(self._histograms.items()):
                for upper_bound, count in zip(HISTOGRAM_BUCKETS, histogram.buckets, strict=True):
                    lines.append(
                        f'model_worker_{name}_bucket{{le="{_bucket_label(upper_bound)}"}} {count}'
                    )
                lines.extend(
                    (
                        f'model_worker_{name}_bucket{{le="+Inf"}} {histogram.count}',
                        f"model_worker_{name}_count {histogram.count}",
                        f"model_worker_{name}_sum {histogram.total:.9f}",
                    )
                )
            return "\n".join(lines) + "\n"
