from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from model_worker.metrics import HISTOGRAM_BUCKETS, Metrics


def test_histogram_renders_cumulative_buckets_count_and_sum():
    metrics = Metrics()
    for value in (0.5, 1, 3, 900_000):
        metrics.observe("latency_ms", value)

    rendered = metrics.render()
    assert 'model_worker_latency_ms_bucket{le="1"} 2' in rendered
    assert 'model_worker_latency_ms_bucket{le="2.5"} 2' in rendered
    assert 'model_worker_latency_ms_bucket{le="5"} 3' in rendered
    assert 'model_worker_latency_ms_bucket{le="300000"} 3' in rendered
    assert 'model_worker_latency_ms_bucket{le="+Inf"} 4' in rendered
    assert "model_worker_latency_ms_count 4" in rendered
    assert "model_worker_latency_ms_sum 900004.500000000" in rendered


def test_histogram_memory_does_not_grow_with_observation_count():
    metrics = Metrics()
    for value in range(20_000):
        metrics.observe("sampled_tokens", value)

    assert not hasattr(metrics, "samples")
    histogram = metrics._histograms["sampled_tokens"]
    assert histogram.count == 20_000
    assert len(histogram.buckets) == len(HISTOGRAM_BUCKETS)
    assert type(histogram).__slots__ == ("buckets", "count", "total")


def test_counter_labels_are_bounded_by_key_and_value_allowlists():
    metrics = Metrics()
    metrics.inc(
        "requests_total",
        termination="completed",
        request_id="must-not-be-retained",
    )
    for index in range(100):
        metrics.inc("requests_total", error_class=f"attacker-controlled-{index}")

    rendered = metrics.render()
    assert 'model_worker_requests_total{termination="completed"} 1' in rendered
    assert 'model_worker_requests_total{error_class="other"} 100' in rendered
    assert "request_id" not in rendered
    assert "attacker-controlled" not in rendered
    assert len(metrics._counters) == 2


def test_metrics_are_thread_safe_and_render_deterministically():
    metrics = Metrics()
    workers = 8
    observations_per_worker = 1_000

    def record(worker: int) -> None:
        for _ in range(observations_per_worker):
            metrics.observe("generation_ms", worker + 0.5)
            metrics.inc("requests_total", phase="final")
        metrics.gauge(f"worker_{worker}_queue_depth", worker)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(record, range(workers)))

    rendered = metrics.render()
    assert rendered == metrics.render()
    total = workers * observations_per_worker
    expected_sum = sum((worker + 0.5) * observations_per_worker for worker in range(workers))
    assert f'model_worker_generation_ms_bucket{{le="+Inf"}} {total}' in rendered
    assert f"model_worker_generation_ms_count {total}" in rendered
    assert f"model_worker_generation_ms_sum {expected_sum:.9f}" in rendered
    assert f'model_worker_requests_total{{phase="final"}} {total}' in rendered
    gauge_lines = [line for line in rendered.splitlines() if "_queue_depth " in line]
    assert gauge_lines == sorted(gauge_lines)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_histogram_rejects_non_finite_observations(value):
    metrics = Metrics()
    with pytest.raises(ValueError, match="must be finite"):
        metrics.observe("latency_ms", value)
    assert metrics.render() == "\n"
