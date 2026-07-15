from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from model_worker.artifacts import ArtifactStore
from model_worker.maintenance import MaintenanceRunner
from model_worker.metrics import Metrics
from model_worker.request_registry import Lifecycle, RequestRegistry, TERMINAL
from model_worker.terminal_metrics import TerminalMetricsObserver


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def prepare_running(registry: RequestRegistry):
    record = registry.create({}, 1_000, 1_000)
    assert registry.transition(record, Lifecycle.PREFLIGHTED)
    assert registry.transition(record, Lifecycle.QUEUED)
    assert registry.transition(record, Lifecycle.RUNNING)
    return record


def wait_for_terminal(record, ready: threading.Barrier) -> Lifecycle:
    ready.wait()
    with record.condition:
        while record.lifecycle not in TERMINAL:
            record.condition.wait()
    return record.lifecycle


def test_terminal_observer_runs_once_across_duplicate_wait_and_disconnect_race():
    calls = []
    calls_lock = threading.Lock()

    def observe(record) -> None:
        with calls_lock:
            calls.append((record.request_id, record.lifecycle))

    registry = RequestRegistry(clock=lambda: 10.0, terminal_observers=(observe,))
    record = prepare_running(registry)
    waiter_ready = threading.Barrier(3)
    with ThreadPoolExecutor(max_workers=6) as executor:
        waiters = [
            executor.submit(wait_for_terminal, record, waiter_ready)
            for _ in range(2)
        ]
        waiter_ready.wait()
        transition_ready = threading.Barrier(5)

        def terminalize(target: Lifecycle, error=None) -> bool:
            transition_ready.wait()
            return registry.compare_and_transition(
                record,
                Lifecycle.RUNNING,
                target,
                error=error,
                result={"ok": True} if target == Lifecycle.COMPLETED else None,
            )

        races = [
            executor.submit(terminalize, Lifecycle.COMPLETED),
            executor.submit(terminalize, Lifecycle.FAILED, "worker_crashed"),
            executor.submit(terminalize, Lifecycle.TIMED_OUT, "deadline_exceeded"),
        ]

        def disconnect() -> bool:
            transition_ready.wait()
            registry.cancel(record.request_id)
            return registry.compare_and_transition(
                record,
                Lifecycle.RUNNING,
                Lifecycle.CANCELLED,
                error="cancelled",
            )

        races.append(executor.submit(disconnect))
        transition_ready.wait()
        assert sum(future.result() for future in races) == 1
        assert all(future.result() == record.lifecycle for future in waiters)

    assert calls == [(record.request_id, record.lifecycle)]
    assert registry.transition(record, Lifecycle.FAILED) is False
    assert registry.cancel(record.request_id) is False
    assert calls == [(record.request_id, record.lifecycle)]


def test_terminal_observer_failure_is_isolated_and_registration_is_deduplicated():
    observed = []

    def broken(_record) -> None:
        raise RuntimeError("metrics unavailable")

    def healthy(record) -> None:
        observed.append(record.lifecycle)

    registry = RequestRegistry(terminal_observers=(broken, healthy))
    assert registry.add_terminal_observer(healthy) is False
    record = prepare_running(registry)
    assert registry.transition(record, Lifecycle.COMPLETED, result={})
    assert observed == [Lifecycle.COMPLETED]
    assert registry.terminal_observer_failures == 1
    assert registry.remove_terminal_observer(healthy) is True
    assert registry.remove_terminal_observer(healthy) is False


def test_registry_ttl_prune_uses_injected_clock_and_hard_oldest_first_bound():
    clock = FakeClock()
    registry = RequestRegistry(clock=clock)
    oldest = prepare_running(registry)
    assert registry.transition(oldest, Lifecycle.COMPLETED, result={})

    clock.value = 5
    newer = prepare_running(registry)
    assert registry.transition(newer, Lifecycle.FAILED, error="worker_crashed")
    live = registry.create({}, 1_000, 1_000)

    clock.value = 10
    first = registry.prune_terminal(ttl_seconds=5, max_records=1)
    assert first.scanned_records == 3
    assert first.terminal_records == 2
    assert first.eligible_records == 2
    assert first.removed_records == 1
    assert first.eligible_remaining == 1
    assert first.limit_reached is True
    assert registry.get(oldest.request_id) is None
    assert registry.get(newer.request_id) is newer
    assert registry.get(live.request_id) is live

    second = registry.prune_terminal(ttl_seconds=5, max_records=1)
    assert second.removed_records == 1
    assert second.eligible_remaining == 0
    assert second.limit_reached is False
    assert registry.snapshot() == (live,)


def test_zero_prune_bound_reports_remaining_eligible_work_without_removal():
    registry = RequestRegistry(clock=lambda: 0.0)
    record = prepare_running(registry)
    registry.transition(record, Lifecycle.COMPLETED, result={})

    stats = registry.prune_terminal(ttl_seconds=0, max_records=0, now=0)
    assert stats.removed_records == 0
    assert stats.eligible_remaining == 1
    assert stats.limit_reached is True
    assert registry.get(record.request_id) is record


def test_terminal_metrics_reconcile_once_at_transition_not_at_wait_or_response():
    clock = FakeClock()
    metrics = Metrics()
    observer = TerminalMetricsObserver(metrics)
    registry = RequestRegistry(clock=clock, terminal_observers=(observer,))
    record = registry.create({}, 1_000, 1_000)
    clock.value = 1
    registry.transition(record, Lifecycle.PREFLIGHTED)
    clock.value = 2
    registry.transition(record, Lifecycle.QUEUED)
    clock.value = 5
    registry.transition(record, Lifecycle.RUNNING)
    clock.value = 10
    registry.transition(
        record,
        Lifecycle.COMPLETED,
        result={
            "timing": {"prompt_decode_ms": 2.5, "generation_ms": 4},
            "usage": {"prompt_tokens": 12, "sampled_tokens": 3},
        },
    )

    # Repeated reads/waits and a late disconnect cannot account the record twice.
    assert record.lifecycle == Lifecycle.COMPLETED
    assert registry.get(record.request_id) is record
    assert registry.cancel(record.request_id) is False
    rendered = metrics.render()
    assert 'model_worker_requests_total{termination="completed"} 1' in rendered
    assert "model_worker_queue_wait_ms_count 1" in rendered
    assert "model_worker_queue_wait_ms_sum 3000.000000000" in rendered
    assert "model_worker_prompt_decode_ms_count 1" in rendered
    assert "model_worker_prompt_tokens_count 1" in rendered


def test_terminal_metrics_include_typed_failure_without_http_serialization():
    metrics = Metrics()
    registry = RequestRegistry(terminal_observers=(TerminalMetricsObserver(metrics),))
    record = registry.create({}, 1_000, 1_000)
    registry.transition(record, Lifecycle.PREFLIGHTED)
    registry.transition(record, Lifecycle.QUEUED)
    registry.transition(record, Lifecycle.TIMED_OUT, error="queue_timeout")

    assert (
        'model_worker_requests_total{error_class="queue_timeout",termination="timed_out"} 1'
        in metrics.render()
    )


def test_maintenance_run_prunes_registry_and_artifacts_and_exports_metrics(tmp_path):
    clock = FakeClock()
    registry = RequestRegistry(clock=clock)
    record = prepare_running(registry)
    registry.transition(record, Lifecycle.COMPLETED, result={})
    clock.value = 20

    artifacts = ArtifactStore(
        tmp_path / "artifacts",
        total_quota=1_000_000,
        retention_seconds=5,
    )
    artifact = artifacts.begin("request", "attempt", 10_000)
    artifact.write_manifest(
        {},
        {},
        {"manifest_digest": "sha256:model", "runtime_build": "b10012"},
        {},
        {},
    )
    artifacts.finish(artifact)
    os.utime(artifact.path / "manifest.json", (1, 1))
    os.utime(artifact.path, (1, 1))
    metrics = Metrics()
    runner = MaintenanceRunner(
        registry,
        artifacts,
        interval_seconds=60,
        terminal_ttl_seconds=10,
        max_registry_prune=1,
        max_artifact_removals=1,
        metrics=metrics,
        wall_clock=lambda: 100.0,
        monotonic_clock=lambda: 5.0,
    )

    stats = runner.run_once()
    assert stats.registry.removed_records == 1
    assert stats.artifacts.removed_incomplete_attempts == 1
    assert registry.snapshot() == ()
    assert not artifact.path.exists()
    assert runner.last_run is stats
    rendered = metrics.render()
    assert "model_worker_maintenance_runs_total 1" in rendered
    assert "model_worker_registry_records 0.0" in rendered
    assert "model_worker_artifact_removed_attempts_last 1.0" in rendered


class BlockingArtifacts:
    def __init__(self, inner: ArtifactStore) -> None:
        self.inner = inner
        self.called = threading.Event()
        self.release = threading.Event()
        self.max_removals = None

    def cleanup_with_stats(self, now=None, *, max_removals=None):
        self.max_removals = max_removals
        self.called.set()
        self.release.wait(5)
        return self.inner.cleanup_with_stats(now=now, max_removals=max_removals)


def test_maintenance_stop_is_deadline_bounded_while_cleanup_is_blocked(tmp_path):
    registry = RequestRegistry()
    artifacts = BlockingArtifacts(
        ArtifactStore(
            tmp_path / "artifacts",
            total_quota=1_000,
            retention_seconds=60,
        )
    )
    runner = MaintenanceRunner(
        registry,
        artifacts,
        interval_seconds=60,
        terminal_ttl_seconds=60,
        max_registry_prune=4,
        max_artifact_removals=3,
        run_immediately=True,
    )
    assert runner.start() is True
    assert runner.start() is False
    assert artifacts.called.wait(1)

    started = time.monotonic()
    assert runner.stop(timeout=0.02) is False
    assert time.monotonic() - started < 0.25
    assert artifacts.max_removals == 3

    artifacts.release.set()
    assert runner.stop(timeout=1) is True
    assert runner.running is False
