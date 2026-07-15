from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROCESS_TREE_SCOPE = "service_process_tree"
TOTAL_SYSTEM_FALLBACK_SCOPE = "total_system_fallback"
UNAVAILABLE_SCOPE = "unavailable"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True, slots=True)
class ProcessInfo:
    pid: int
    parent_pid: int
    rss_bytes: int
    name: str


def _run(command: list[str]) -> str:
    return subprocess.check_output(
        command,
        text=True,
        stderr=subprocess.DEVNULL,
        creationflags=_NO_WINDOW,
    )


def query_process_snapshot() -> dict[int, ProcessInfo] | None:
    command = (
        "$items = @(Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,WorkingSetSize,Name); "
        "ConvertTo-Json -Compress -InputObject $items"
    )
    try:
        output = _run(["powershell", "-NoProfile", "-Command", command])
        raw = json.loads(output)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError):
        return None

    if isinstance(raw, dict):
        items: list[Any] = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return None

    snapshot: dict[int, ProcessInfo] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            pid = int(item["ProcessId"])
            parent_pid = int(item["ParentProcessId"])
            rss_bytes = int(item["WorkingSetSize"])
        except (KeyError, TypeError, ValueError):
            continue
        name = item.get("Name")
        if pid <= 0 or parent_pid < 0 or rss_bytes < 0 or type(name) is not str:
            continue
        snapshot[pid] = ProcessInfo(pid, parent_pid, rss_bytes, name)
    return snapshot


def process_tree_pids(root_pid: int, snapshot: dict[int, ProcessInfo]) -> tuple[int, ...]:
    if root_pid not in snapshot:
        return ()
    children: dict[int, list[int]] = defaultdict(list)
    for process in snapshot.values():
        children[process.parent_pid].append(process.pid)
    for child_pids in children.values():
        child_pids.sort()

    ordered: list[int] = []
    pending = deque([root_pid])
    seen: set[int] = set()
    while pending:
        pid = pending.popleft()
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append(pid)
        pending.extend(children.get(pid, ()))
    return tuple(ordered)


def _parse_memory_mib(value: str) -> int | None:
    matched = re.fullmatch(r"\s*(\d+)\s*(?:MiB)?\s*", value, flags=re.IGNORECASE)
    return int(matched.group(1)) if matched else None


def parse_per_process_vram(output: str) -> dict[int, int] | None:
    if not output.strip():
        return {}
    usage: dict[int, int] = defaultdict(int)
    for row in csv.reader(output.splitlines()):
        if len(row) != 2:
            return None
        try:
            pid = int(row[0].strip())
        except ValueError:
            return None
        memory = _parse_memory_mib(row[1])
        if pid <= 0 or memory is None:
            return None
        usage[pid] += memory
    return dict(usage)


def query_per_process_vram() -> tuple[dict[int, int] | None, str | None]:
    try:
        output = _run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.SubprocessError):
        return None, "per_process_query_failed"
    parsed = parse_per_process_vram(output)
    if parsed is None:
        return None, "per_process_memory_unavailable"
    return parsed, None


def parse_wddm_process_vram(output: str) -> dict[int, float] | None:
    if not output.strip():
        return {}
    try:
        raw = json.loads(output)
    except json.JSONDecodeError:
        return None
    items = [raw] if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return None
    usage_bytes: dict[int, float] = defaultdict(float)
    for item in items:
        if not isinstance(item, dict):
            return None
        instance = item.get("InstanceName")
        value = item.get("CookedValue")
        if type(instance) is not str or type(value) not in {int, float}:
            return None
        matched = re.match(r"^pid_(\d+)_", instance, flags=re.IGNORECASE)
        if matched is None:
            continue
        pid = int(matched.group(1))
        if pid <= 0 or value < 0:
            return None
        usage_bytes[pid] += float(value)
    return {
        pid: round(byte_count / (1024 * 1024), 3)
        for pid, byte_count in usage_bytes.items()
    }


def query_wddm_process_vram() -> tuple[dict[int, float] | None, str | None]:
    command = (
        "$items = @((Get-Counter "
        "'\\GPU Process Memory(*)\\Dedicated Usage' -ErrorAction Stop)."
        "CounterSamples | Select-Object InstanceName,CookedValue); "
        "ConvertTo-Json -Compress -InputObject $items"
    )
    try:
        output = _run(["powershell", "-NoProfile", "-Command", command])
    except (OSError, subprocess.SubprocessError):
        return None, "wddm_process_memory_query_failed"
    parsed = parse_wddm_process_vram(output)
    if parsed is None:
        return None, "wddm_process_memory_unavailable"
    return parsed, None


def query_total_system_vram() -> int | None:
    try:
        output = _run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        values = [_parse_memory_mib(line) for line in output.splitlines() if line.strip()]
        if not values or any(value is None for value in values):
            return None
        return sum(value for value in values if value is not None)
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_vram_measurement(pid_tree: tuple[int, ...] | None) -> dict[str, Any]:
    if pid_tree is not None:
        usage, reason = query_per_process_vram()
        backend = "nvidia-smi-compute-apps"
        if usage is None:
            nvidia_reason = reason
            usage, reason = query_wddm_process_vram()
            backend = "windows-gpu-process-memory-counter"
            if usage is None:
                reason = f"{nvidia_reason};{reason}"
        if usage is not None:
            selected = {pid: usage[pid] for pid in pid_tree if pid in usage}
            return {
                "scope": PROCESS_TREE_SCOPE,
                "mib": round(sum(selected.values()), 3),
                "processes": [
                    {"pid": pid, "mib": selected[pid]} for pid in sorted(selected)
                ],
                "fallback_reason": None,
                "backend": backend,
            }
    else:
        reason = "process_tree_unavailable"

    fallback = query_total_system_vram()
    if fallback is not None:
        return {
            "scope": TOTAL_SYSTEM_FALLBACK_SCOPE,
            "mib": fallback,
            "processes": [],
            "fallback_reason": reason,
            "backend": "nvidia-smi-total-system",
        }
    return {
        "scope": UNAVAILABLE_SCOPE,
        "mib": None,
        "processes": [],
        "fallback_reason": reason,
        "backend": None,
    }


def collect_sample(root_pid: int, *, timestamp: float | None = None) -> dict[str, Any]:
    snapshot = query_process_snapshot()
    pids = process_tree_pids(root_pid, snapshot) if snapshot is not None else ()
    process_tree_available = snapshot is not None and bool(pids)
    processes = [snapshot[pid] for pid in pids] if snapshot is not None else []
    vram = gpu_vram_measurement(pids if process_tree_available else None)
    process_tree_rss = sum(process.rss_bytes for process in processes) if processes else None
    service = snapshot.get(root_pid) if snapshot is not None else None
    return {
        "timestamp_unix": time.time() if timestamp is None else timestamp,
        "root_pid": root_pid,
        "rss_scope": PROCESS_TREE_SCOPE,
        "process_snapshot_available": snapshot is not None,
        "process_tree_pids": list(pids),
        "processes": [
            {
                "pid": process.pid,
                "parent_pid": process.parent_pid,
                "name": process.name,
                "rss_bytes": process.rss_bytes,
            }
            for process in processes
        ],
        "service_rss_bytes": service.rss_bytes if service is not None else None,
        "process_tree_rss_bytes": process_tree_rss,
        "gpu_vram_scope": vram["scope"],
        "gpu_vram_mib": vram["mib"],
        "gpu_vram_processes": vram["processes"],
        "gpu_vram_fallback_reason": vram["fallback_reason"],
        "gpu_vram_backend": vram["backend"],
        "gpu_vram_mib_total_system": (
            vram["mib"] if vram["scope"] == TOTAL_SYSTEM_FALLBACK_SCOPE else None
        ),
    }


def _stable(values: list[int], sample_count: int) -> list[int]:
    return values[-max(1, sample_count // 5) :] if values else []


def build_report(
    root_pid: int,
    started_at: float,
    finished_at: float,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    rss = [
        sample["process_tree_rss_bytes"]
        for sample in samples
        if sample["process_tree_rss_bytes"] is not None
    ]
    process_vram = [
        sample["gpu_vram_mib"]
        for sample in samples
        if sample["gpu_vram_scope"] == PROCESS_TREE_SCOPE
        and sample["gpu_vram_mib"] is not None
    ]
    fallback_vram = [
        sample["gpu_vram_mib"]
        for sample in samples
        if sample["gpu_vram_scope"] == TOTAL_SYSTEM_FALLBACK_SCOPE
        and sample["gpu_vram_mib"] is not None
    ]
    scopes = sorted({sample["gpu_vram_scope"] for sample in samples})
    stable_rss = _stable(rss, len(samples))
    stable_process_vram = _stable(process_vram, len(samples))
    stable_fallback_vram = _stable(fallback_vram, len(samples))
    return {
        "pid": root_pid,
        "rss_scope": PROCESS_TREE_SCOPE,
        "started_at_unix": started_at,
        "duration_seconds": finished_at - started_at,
        "samples": samples,
        # Backward-compatible alias; its scope is now explicit above.
        "peak_rss_bytes": max(rss, default=None),
        "peak_rss_bytes_process_tree": max(rss, default=None),
        "stable_rss_bytes": stable_rss,
        "stable_rss_bytes_process_tree": stable_rss,
        "gpu_vram_measurement_scopes": scopes,
        "peak_vram_mib_process_tree": max(process_vram, default=None),
        "stable_vram_mib_process_tree": stable_process_vram,
        "peak_vram_mib_total_system_fallback": max(fallback_vram, default=None),
        "stable_vram_mib_total_system_fallback": stable_fallback_vram,
        # Legacy field remains available only for explicitly-labelled fallback samples.
        "peak_vram_mib_total_system": max(fallback_vram, default=None),
        "stable_vram_mib_total_system": stable_fallback_vram,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()
    if args.pid <= 0:
        parser.error("--pid must be positive")
    if args.interval <= 0:
        parser.error("--interval must be positive")

    samples: list[dict[str, Any]] = []
    started = time.time()
    while not args.stop_file.exists():
        samples.append(collect_sample(args.pid))
        time.sleep(args.interval)
    finished = time.time()
    payload = build_report(args.pid, started, finished, samples)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
