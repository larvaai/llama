from __future__ import annotations

import json
import subprocess

from scripts import monitor_resources as monitor


def test_process_snapshot_and_tree_aggregate_service_with_all_descendants(monkeypatch):
    raw = json.dumps(
        [
            {"ProcessId": 10, "ParentProcessId": 1, "WorkingSetSize": "100", "Name": "python.exe"},
            {"ProcessId": 11, "ParentProcessId": 10, "WorkingSetSize": "200", "Name": "native.exe"},
            {"ProcessId": 12, "ParentProcessId": 11, "WorkingSetSize": "300", "Name": "helper.exe"},
            {"ProcessId": 99, "ParentProcessId": 1, "WorkingSetSize": "900", "Name": "other.exe"},
        ]
    )
    monkeypatch.setattr(monitor, "_run", lambda command: raw)
    snapshot = monitor.query_process_snapshot()
    assert snapshot is not None
    assert monitor.process_tree_pids(10, snapshot) == (10, 11, 12)
    assert monitor.process_tree_pids(99, snapshot) == (99,)
    assert monitor.process_tree_pids(404, snapshot) == ()


def test_process_snapshot_failure_is_distinct_from_empty_snapshot(monkeypatch):
    def fail(command):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(monitor, "_run", fail)
    assert monitor.query_process_snapshot() is None

    monkeypatch.setattr(monitor, "_run", lambda command: "[]")
    assert monitor.query_process_snapshot() == {}


def test_per_process_vram_parser_sums_multi_gpu_rows_and_fails_closed():
    assert monitor.parse_per_process_vram("11, 100\n11, 25 MiB\n99, 7\n") == {
        11: 125,
        99: 7,
    }
    assert monitor.parse_per_process_vram("") == {}
    assert monitor.parse_per_process_vram("11, [N/A]\n") is None
    assert monitor.parse_per_process_vram("not-a-pid, 10\n") is None


def test_per_process_query_uses_nvidia_compute_app_memory_field(monkeypatch):
    commands = []
    monkeypatch.setattr(
        monitor,
        "_run",
        lambda command: commands.append(command) or "11, 256\n",
    )
    usage, reason = monitor.query_per_process_vram()
    assert usage == {11: 256} and reason is None
    assert "--query-compute-apps=pid,used_gpu_memory" in commands[0]


def test_wddm_process_memory_parser_sums_instances_and_converts_bytes_to_mib():
    raw = json.dumps(
        [
            {"InstanceName": "pid_11_luid_a_phys_0", "CookedValue": 1048576},
            {"InstanceName": "pid_11_luid_b_phys_1", "CookedValue": 524288},
            {"InstanceName": "pid_99_luid_a_phys_0", "CookedValue": 2097152},
        ]
    )
    assert monitor.parse_wddm_process_vram(raw) == {11: 1.5, 99: 2.0}
    assert monitor.parse_wddm_process_vram("not-json") is None


def test_vram_uses_only_pid_tree_when_per_process_query_is_supported(monkeypatch):
    monkeypatch.setattr(
        monitor,
        "query_per_process_vram",
        lambda: ({10: 2, 11: 800, 99: 900}, None),
    )
    fallback_called = []
    monkeypatch.setattr(
        monitor,
        "query_total_system_vram",
        lambda: fallback_called.append(True),
    )
    measured = monitor.gpu_vram_measurement((10, 11))
    assert measured == {
        "scope": "service_process_tree",
        "mib": 802,
        "processes": [{"pid": 10, "mib": 2}, {"pid": 11, "mib": 800}],
        "fallback_reason": None,
        "backend": "nvidia-smi-compute-apps",
    }
    assert fallback_called == []


def test_vram_fallback_is_explicitly_labelled(monkeypatch):
    monkeypatch.setattr(
        monitor,
        "query_per_process_vram",
        lambda: (None, "per_process_memory_unavailable"),
    )
    monkeypatch.setattr(
        monitor,
        "query_wddm_process_vram",
        lambda: (None, "wddm_process_memory_unavailable"),
    )
    monkeypatch.setattr(monitor, "query_total_system_vram", lambda: 4096)
    measured = monitor.gpu_vram_measurement((10, 11))
    assert measured == {
        "scope": "total_system_fallback",
        "mib": 4096,
        "processes": [],
        "fallback_reason": (
            "per_process_memory_unavailable;wddm_process_memory_unavailable"
        ),
        "backend": "nvidia-smi-total-system",
    }

    measured = monitor.gpu_vram_measurement(None)
    assert measured["scope"] == "total_system_fallback"
    assert measured["fallback_reason"] == "process_tree_unavailable"


def test_vram_uses_wddm_process_counter_before_total_system_fallback(monkeypatch):
    monkeypatch.setattr(
        monitor,
        "query_per_process_vram",
        lambda: (None, "per_process_memory_unavailable"),
    )
    monkeypatch.setattr(
        monitor,
        "query_wddm_process_vram",
        lambda: ({10: 10.5, 11: 700.25, 99: 900.0}, None),
    )
    fallback_called = []
    monkeypatch.setattr(
        monitor,
        "query_total_system_vram",
        lambda: fallback_called.append(True),
    )
    measured = monitor.gpu_vram_measurement((10, 11))
    assert measured == {
        "scope": "service_process_tree",
        "mib": 710.75,
        "processes": [{"pid": 10, "mib": 10.5}, {"pid": 11, "mib": 700.25}],
        "fallback_reason": None,
        "backend": "windows-gpu-process-memory-counter",
    }
    assert fallback_called == []


def test_collect_sample_reports_process_tree_rss_and_vram_provenance(monkeypatch):
    snapshot = {
        10: monitor.ProcessInfo(10, 1, 100, "python.exe"),
        11: monitor.ProcessInfo(11, 10, 250, "native.exe"),
        99: monitor.ProcessInfo(99, 1, 900, "other.exe"),
    }
    monkeypatch.setattr(monitor, "query_process_snapshot", lambda: snapshot)
    monkeypatch.setattr(
        monitor,
        "gpu_vram_measurement",
        lambda pids: {
            "scope": "service_process_tree",
            "mib": 700,
            "processes": [{"pid": 11, "mib": 700}],
            "fallback_reason": None,
            "backend": "test",
        },
    )
    sample = monitor.collect_sample(10, timestamp=123.0)
    assert sample["process_tree_pids"] == [10, 11]
    assert sample["service_rss_bytes"] == 100
    assert sample["process_tree_rss_bytes"] == 350
    assert sample["gpu_vram_scope"] == "service_process_tree"
    assert sample["gpu_vram_mib"] == 700
    assert sample["gpu_vram_backend"] == "test"
    assert sample["gpu_vram_mib_total_system"] is None


def test_report_never_combines_process_vram_with_total_system_fallback():
    samples = [
        {
            "process_tree_rss_bytes": 100,
            "gpu_vram_scope": "service_process_tree",
            "gpu_vram_mib": 700,
        },
        {
            "process_tree_rss_bytes": 300,
            "gpu_vram_scope": "total_system_fallback",
            "gpu_vram_mib": 5000,
        },
    ]
    report = monitor.build_report(10, 10.0, 12.5, samples)
    assert report["rss_scope"] == "service_process_tree"
    assert report["peak_rss_bytes"] == 300
    assert report["peak_rss_bytes_process_tree"] == 300
    assert report["gpu_vram_measurement_scopes"] == [
        "service_process_tree",
        "total_system_fallback",
    ]
    assert report["peak_vram_mib_process_tree"] == 700
    assert report["peak_vram_mib_total_system_fallback"] == 5000
    assert report["peak_vram_mib_total_system"] == 5000
