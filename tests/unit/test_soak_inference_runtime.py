from __future__ import annotations

import pytest

from scripts.monitor_resources import (
    PROCESS_TREE_SCOPE,
    TOTAL_SYSTEM_FALLBACK_SCOPE,
)
from scripts.soak_inference_runtime import (
    linear_slope_per_hour,
    resource_report,
    soak_expected,
    stability_report,
)


def sample(index: int, rss: int, vram: int, scope: str) -> dict:
    return {
        "timestamp_unix": 1000.0 + index * 10,
        "process_tree_rss_bytes": rss,
        "gpu_vram_scope": scope,
        "gpu_vram_mib": vram,
    }


def test_linear_slope_is_reported_per_hour():
    assert linear_slope_per_hour([(0, 10), (60, 11), (120, 12)]) == pytest.approx(
        60.0
    )


def test_stability_report_rejects_sustained_positive_drift():
    samples = [sample(index, 100 + index * 100, 0, PROCESS_TREE_SCOPE) for index in range(10)]
    report = stability_report(
        samples,
        field="process_tree_rss_bytes",
        absolute_tolerance=10,
        relative_tolerance=0,
    )
    assert report["available"] is True
    assert report["passed"] is False
    assert report["reason"] == "positive_drift_exceeded"


def test_resource_report_uses_explicit_total_system_fallback_on_wddm():
    samples = [
        sample(index, 1024 * 1024 * 1024, 8000 + index % 2, TOTAL_SYSTEM_FALLBACK_SCOPE)
        for index in range(10)
    ]
    report = resource_report(samples)
    assert report["rss_process_tree"]["passed"] is True
    assert report["vram_process_tree"]["available"] is False
    assert report["vram_total_system_fallback"]["passed"] is True
    assert report["selected_vram_scope"] == TOTAL_SYSTEM_FALLBACK_SCOPE
    assert report["passed"] is True


def test_soak_uses_fixed_slot_shapes_instead_of_late_request_number_growth():
    first_wave = [soak_expected(index) for index in range(8)]
    assert len(set(first_wave)) == 8
    assert all(value.startswith("soak-") for value in first_wave)
    with pytest.raises(ValueError, match="outside"):
        soak_expected(8)
