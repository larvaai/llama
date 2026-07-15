from __future__ import annotations

import threading
from collections import Counter, defaultdict


class Metrics:
    """Low-cardinality in-process metrics; IDs are deliberately never labels."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counters: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.gauges: dict[str, float] = {}

    def inc(self, name: str, **labels: str) -> None:
        safe = tuple(sorted((key, value) for key, value in labels.items() if key in {"termination", "error_class", "phase"}))
        with self._lock: self.counters[(name, safe)] += 1

    def observe(self, name: str, value: float) -> None:
        with self._lock: self.samples[name].append(float(value))

    def gauge(self, name: str, value: float) -> None:
        with self._lock: self.gauges[name] = float(value)

    def render(self) -> str:
        with self._lock:
            lines = []
            for (name, labels), value in sorted(self.counters.items()):
                suffix = "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""
                lines.append(f"model_worker_{name}{suffix} {value}")
            for name, value in sorted(self.gauges.items()): lines.append(f"model_worker_{name} {value}")
            for name, values in sorted(self.samples.items()):
                lines.extend((f"model_worker_{name}_count {len(values)}", f"model_worker_{name}_sum {sum(values):.9f}"))
            return "\n".join(lines) + "\n"
