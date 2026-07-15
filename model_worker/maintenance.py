from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .artifacts import ArtifactCleanupStats
from .metrics import Metrics
from .request_registry import RegistryPruneStats, RequestRegistry


class ArtifactMaintenanceStore(Protocol):
    def cleanup_with_stats(
        self,
        now: float | None = None,
        *,
        max_removals: int | None = None,
    ) -> ArtifactCleanupStats: ...


@dataclass(frozen=True, slots=True)
class MaintenanceRunStats:
    started_at_unix: float
    duration_ms: float
    registry: RegistryPruneStats
    artifacts: ArtifactCleanupStats


def _finite_number(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(converted) or converted < 0 or (positive and converted == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return converted


class MaintenanceRunner:
    """Periodic bounded cleanup for terminal registry records and artifacts."""

    def __init__(
        self,
        registry: RequestRegistry,
        artifacts: ArtifactMaintenanceStore,
        *,
        interval_seconds: float,
        terminal_ttl_seconds: float,
        max_registry_prune: int,
        max_artifact_removals: int,
        metrics: Metrics | None = None,
        run_immediately: bool = True,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.interval_seconds = _finite_number(
            interval_seconds,
            "interval_seconds",
            positive=True,
        )
        self.terminal_ttl_seconds = _finite_number(
            terminal_ttl_seconds,
            "terminal_ttl_seconds",
        )
        if type(max_registry_prune) is not int or max_registry_prune < 0:
            raise ValueError("max_registry_prune must be a non-negative integer")
        if type(max_artifact_removals) is not int or max_artifact_removals < 0:
            raise ValueError("max_artifact_removals must be a non-negative integer")
        if type(run_immediately) is not bool:
            raise ValueError("run_immediately must be a boolean")
        if not callable(wall_clock) or not callable(monotonic_clock):
            raise TypeError("maintenance clocks must be callable")

        self.registry = registry
        self.artifacts = artifacts
        self.max_registry_prune = max_registry_prune
        self.max_artifact_removals = max_artifact_removals
        self.metrics = metrics
        self.run_immediately = run_immediately
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_run: MaintenanceRunStats | None = None
        self._failure_count = 0
        self._last_error_type: str | None = None

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def last_run(self) -> MaintenanceRunStats | None:
        with self._state_lock:
            return self._last_run

    @property
    def failure_count(self) -> int:
        with self._state_lock:
            return self._failure_count

    @property
    def last_error_type(self) -> str | None:
        with self._state_lock:
            return self._last_error_type

    def _read_clock(self, clock: Callable[[], float], name: str) -> float:
        return _finite_number(clock(), name)

    def run_once(self) -> MaintenanceRunStats:
        with self._run_lock:
            started_at = self._read_clock(self._wall_clock, "wall_clock")
            started_monotonic = self._read_clock(
                self._monotonic_clock,
                "monotonic_clock",
            )
            registry_stats = self.registry.prune_terminal(
                ttl_seconds=self.terminal_ttl_seconds,
                max_records=self.max_registry_prune,
            )
            artifact_stats = self.artifacts.cleanup_with_stats(
                now=started_at,
                max_removals=self.max_artifact_removals,
            )
            finished = self._read_clock(self._monotonic_clock, "monotonic_clock")
            stats = MaintenanceRunStats(
                started_at_unix=started_at,
                duration_ms=max(0.0, finished - started_monotonic) * 1000,
                registry=registry_stats,
                artifacts=artifact_stats,
            )
            with self._state_lock:
                self._last_run = stats
                self._last_error_type = None
            self._record_metrics(stats)
            return stats

    def _record_metrics(self, stats: MaintenanceRunStats) -> None:
        if self.metrics is None:
            return
        self.metrics.inc("maintenance_runs_total")
        self.metrics.observe("maintenance_duration_ms", stats.duration_ms)
        self.metrics.gauge("registry_records", stats.registry.remaining_records)
        self.metrics.gauge(
            "registry_pruned_records_last",
            stats.registry.removed_records,
        )
        self.metrics.gauge("artifact_bytes", stats.artifacts.bytes_after)
        self.metrics.gauge(
            "artifact_removed_attempts_last",
            stats.artifacts.removed_attempts,
        )
        self.metrics.gauge(
            "artifact_cleanup_errors_last",
            stats.artifacts.scan_errors + stats.artifacts.delete_errors,
        )

    def start(self) -> bool:
        with self._state_lock:
            if self._thread is not None:
                return False
            if self._stop_event.is_set():
                raise RuntimeError("maintenance runner cannot restart after stop")
            thread = threading.Thread(
                target=self._run,
                name="model-worker-maintenance",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return True

    def _run(self) -> None:
        if self.run_immediately and not self._stop_event.is_set():
            self._run_background_once()
        while not self._stop_event.wait(self.interval_seconds):
            self._run_background_once()

    def _run_background_once(self) -> None:
        try:
            self.run_once()
        except Exception as exc:
            with self._state_lock:
                self._failure_count += 1
                self._last_error_type = type(exc).__name__
            if self.metrics is not None:
                self.metrics.inc("maintenance_failures_total")

    def stop(self, timeout: float = 5.0) -> bool:
        bounded_timeout = _finite_number(timeout, "timeout")
        self._stop_event.set()
        with self._state_lock:
            thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(bounded_timeout)
        return not thread.is_alive()
