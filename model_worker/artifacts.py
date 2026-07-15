from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import WorkerError


_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@dataclass(slots=True)
class AttemptArtifact:
    path: Path
    max_bytes: int
    _written: int = 0
    _terminal: bool = False

    def write_manifest(self, request_hash_source: Any, contract: Any, model: dict[str, Any], limits: Any, timestamps: dict[str, Any]) -> None:
        payload = {"request_hash": _canonical_hash(request_hash_source), "contract_hash": _canonical_hash(contract), "prompt_hash": model.get("prompt_hash"), "prompt_version": model.get("prompt_version"), "model_manifest_digest": model["manifest_digest"], "runtime_build": model["runtime_build"], "limits": limits, "timestamps": timestamps}
        self._write_new("manifest.json", payload)

    def write_result(self, result: dict[str, Any]) -> None:
        if self._terminal: raise RuntimeError("terminal artifact already written")
        self._write_new("result.json", result, atomic=True)
        self._terminal = True

    def _write_new(self, name: str, payload: Any, atomic: bool = False) -> None:
        target = self.path / name
        if target.exists(): raise FileExistsError(f"immutable artifact exists: {name}")
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if self._written + len(data) > self.max_bytes: raise WorkerError("request_too_large", "artifact byte quota exceeded")
        if atomic:
            fd, temp_name = tempfile.mkstemp(prefix=".result-", suffix=".tmp", dir=self.path)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data); stream.flush(); os.fsync(stream.fileno())
                os.replace(temp_name, target)
            finally:
                if os.path.exists(temp_name): os.unlink(temp_name)
        else:
            target.write_bytes(data)
        self._written += len(data)


@dataclass(frozen=True, slots=True)
class ArtifactCleanupStats:
    started_at_unix: float
    duration_ms: float
    scanned_attempts: int
    completed_attempts: int
    incomplete_attempts: int
    expired_candidates: int
    active_attempts_skipped: int
    unsafe_entries_skipped: int
    scan_errors: int
    delete_errors: int
    removed_attempts: int
    removed_completed_attempts: int
    removed_incomplete_attempts: int
    removed_expired_attempts: int
    removed_for_quota_attempts: int
    bytes_before: int
    bytes_after: int
    bytes_removed: int
    quota_bytes: int
    quota_satisfied: bool
    max_removals: int | None
    removal_limit_reached: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _AttemptInspection:
    size_bytes: int
    activity_mtime: float
    result_mtime: float | None


@dataclass(frozen=True, slots=True)
class _CleanupCandidate:
    path: Path
    relative_key: str
    modified: float
    size_bytes: int
    completed: bool
    expired: bool


def _is_link_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        _REPARSE_POINT and getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    return resolved == root or root in resolved.parents


class ArtifactStore:
    def __init__(self, root: Path, *, total_quota: int, retention_seconds: int) -> None:
        if type(total_quota) is not int or total_quota < 0:
            raise ValueError("total_quota must be a non-negative integer")
        if type(retention_seconds) is not int or retention_seconds < 0:
            raise ValueError("retention_seconds must be a non-negative integer")
        self.root = root.resolve()
        self.total_quota = total_quota
        self.retention_seconds = retention_seconds
        self.active: set[Path] = set()
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)
        try: os.chmod(self.root, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        except OSError: pass

    def begin(self, request_id: str, attempt_id: str, max_bytes: int) -> AttemptArtifact:
        if not request_id.isalnum() or not attempt_id.isalnum(): raise ValueError("server IDs must be path-safe")
        day = datetime.now(UTC).date().isoformat()
        path = (self.root / day / request_id / attempt_id).resolve()
        if self.root not in path.parents: raise ValueError("artifact path escaped root")
        with self.lock:
            path.mkdir(parents=True, exist_ok=False)
            self.active.add(path)
        return AttemptArtifact(path, max_bytes)

    def finish(self, artifact: AttemptArtifact) -> None:
        with self.lock: self.active.discard(artifact.path)

    def cleanup(self, now: float | None = None, *, max_removals: int | None = None) -> int:
        return self.cleanup_with_stats(now, max_removals=max_removals).removed_attempts

    def cleanup_with_stats(
        self,
        now: float | None = None,
        *,
        max_removals: int | None = None,
    ) -> ArtifactCleanupStats:
        if max_removals is not None and (
            type(max_removals) is not int or max_removals < 0
        ):
            raise ValueError("max_removals must be a non-negative integer or None")
        now = time.time() if now is None else now
        if type(now) not in {int, float}:
            raise ValueError("now must be a Unix timestamp")
        started_at = time.time()
        started_monotonic = time.monotonic()
        with self.lock:
            bytes_before, usage_errors = self._safe_tree_usage()
            attempt_paths, unsafe_entries, discovery_errors = self._discover_attempts()
            scan_errors = usage_errors + discovery_errors
            active_skipped = 0
            completed = 0
            incomplete = 0
            candidates: list[_CleanupCandidate] = []

            for attempt in attempt_paths:
                try:
                    resolved = attempt.resolve(strict=True)
                except (OSError, RuntimeError):
                    scan_errors += 1
                    continue
                if resolved in self.active:
                    active_skipped += 1
                    continue
                inspection, unsafe, errors = self._inspect_attempt(attempt)
                unsafe_entries += unsafe
                scan_errors += errors
                if inspection is None:
                    continue
                is_completed = inspection.result_mtime is not None
                completed += int(is_completed)
                incomplete += int(not is_completed)
                modified = (
                    inspection.result_mtime
                    if inspection.result_mtime is not None
                    else inspection.activity_mtime
                )
                candidates.append(
                    _CleanupCandidate(
                        attempt,
                        attempt.relative_to(self.root).as_posix(),
                        modified,
                        inspection.size_bytes,
                        is_completed,
                        now - modified > self.retention_seconds,
                    )
                )

            candidates.sort(key=lambda item: (item.modified, item.relative_key))
            expired_candidates = sum(candidate.expired for candidate in candidates)
            estimated_total = bytes_before
            removed_paths: set[Path] = set()
            removed_completed = 0
            removed_incomplete = 0
            removed_expired = 0
            removed_for_quota = 0
            delete_errors = 0

            def at_limit() -> bool:
                return max_removals is not None and len(removed_paths) >= max_removals

            def remove(candidate: _CleanupCandidate, *, expired: bool) -> bool:
                nonlocal estimated_total, unsafe_entries, scan_errors, delete_errors
                nonlocal removed_completed, removed_incomplete
                nonlocal removed_expired, removed_for_quota
                if at_limit():
                    return False
                # Reinspect immediately before deletion. A path that became a
                # symlink, reparse point, or escaped root is never handed to rmtree.
                inspection, unsafe, errors = self._inspect_attempt(candidate.path)
                unsafe_entries += unsafe
                scan_errors += errors
                if inspection is None:
                    return False
                try:
                    shutil.rmtree(candidate.path)
                except OSError:
                    delete_errors += 1
                    return False
                removed_paths.add(candidate.path)
                estimated_total = max(0, estimated_total - inspection.size_bytes)
                removed_completed += int(candidate.completed)
                removed_incomplete += int(not candidate.completed)
                removed_expired += int(expired)
                removed_for_quota += int(not expired)
                self._prune_empty_parents(candidate.path)
                return True

            for candidate in candidates:
                if candidate.expired:
                    if at_limit():
                        break
                    remove(candidate, expired=True)

            if estimated_total > self.total_quota:
                for candidate in candidates:
                    if estimated_total <= self.total_quota or at_limit():
                        break
                    if candidate.path not in removed_paths:
                        remove(candidate, expired=False)

            bytes_after, final_usage_errors = self._safe_tree_usage()
            scan_errors += final_usage_errors
            expired_remaining = any(
                candidate.expired and candidate.path not in removed_paths
                for candidate in candidates
            )
            limit_reached = at_limit() and (
                expired_remaining or bytes_after > self.total_quota
            )

        return ArtifactCleanupStats(
            started_at_unix=started_at,
            duration_ms=(time.monotonic() - started_monotonic) * 1000,
            scanned_attempts=len(attempt_paths),
            completed_attempts=completed,
            incomplete_attempts=incomplete,
            expired_candidates=expired_candidates,
            active_attempts_skipped=active_skipped,
            unsafe_entries_skipped=unsafe_entries,
            scan_errors=scan_errors,
            delete_errors=delete_errors,
            removed_attempts=len(removed_paths),
            removed_completed_attempts=removed_completed,
            removed_incomplete_attempts=removed_incomplete,
            removed_expired_attempts=removed_expired,
            removed_for_quota_attempts=removed_for_quota,
            bytes_before=bytes_before,
            bytes_after=bytes_after,
            bytes_removed=max(0, bytes_before - bytes_after),
            quota_bytes=self.total_quota,
            quota_satisfied=bytes_after <= self.total_quota,
            max_removals=max_removals,
            removal_limit_reached=limit_reached,
        )

    def _discover_attempts(self) -> tuple[list[Path], int, int]:
        unsafe = 0
        errors = 0
        attempts: list[Path] = []
        days, skipped, failed = self._safe_child_directories(self.root)
        unsafe += skipped
        errors += failed
        for day in days:
            requests, skipped, failed = self._safe_child_directories(day)
            unsafe += skipped
            errors += failed
            for request in requests:
                children, skipped, failed = self._safe_child_directories(request)
                unsafe += skipped
                errors += failed
                attempts.extend(children)
        attempts.sort(key=lambda path: path.relative_to(self.root).as_posix())
        return attempts, unsafe, errors

    def _safe_child_directories(self, parent: Path) -> tuple[list[Path], int, int]:
        directories: list[Path] = []
        unsafe = 0
        errors = 0
        try:
            with os.scandir(parent) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError:
            return directories, unsafe, 1
        for entry in entries:
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                errors += 1
                continue
            if _is_link_or_reparse(info):
                unsafe += 1
                continue
            if not stat.S_ISDIR(info.st_mode):
                continue
            path = Path(entry.path)
            if not _is_within(path, self.root):
                unsafe += 1
                continue
            directories.append(path)
        return directories, unsafe, errors

    def _inspect_attempt(
        self,
        attempt: Path,
    ) -> tuple[_AttemptInspection | None, int, int]:
        try:
            root_info = attempt.lstat()
        except OSError:
            return None, 0, 1
        if (
            _is_link_or_reparse(root_info)
            or not stat.S_ISDIR(root_info.st_mode)
            or not _is_within(attempt, self.root)
        ):
            return None, 1, 0

        size_bytes = 0
        activity_mtime = root_info.st_mtime
        result_mtime: float | None = None
        pending = [attempt]
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as iterator:
                    entries = sorted(iterator, key=lambda entry: entry.name)
            except OSError:
                return None, 0, 1
            for entry in entries:
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    return None, 0, 1
                if _is_link_or_reparse(info):
                    return None, 1, 0
                path = Path(entry.path)
                activity_mtime = max(activity_mtime, info.st_mtime)
                if stat.S_ISDIR(info.st_mode):
                    if not _is_within(path, self.root):
                        return None, 1, 0
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    size_bytes += info.st_size
                    if directory == attempt and entry.name == "result.json":
                        result_mtime = info.st_mtime
                else:
                    return None, 1, 0
        return _AttemptInspection(size_bytes, activity_mtime, result_mtime), 0, 0

    def _safe_tree_usage(self) -> tuple[int, int]:
        total = 0
        errors = 0
        pending = [self.root]
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as iterator:
                    entries = list(iterator)
            except OSError:
                errors += 1
                continue
            for entry in entries:
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    errors += 1
                    continue
                if _is_link_or_reparse(info):
                    continue
                path = Path(entry.path)
                if stat.S_ISDIR(info.st_mode):
                    if _is_within(path, self.root):
                        pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
        return total, errors

    def _prune_empty_parents(self, attempt: Path) -> None:
        parent = attempt.parent
        while parent != self.root and self.root in parent.parents:
            try:
                info = parent.lstat()
                if _is_link_or_reparse(info) or not _is_within(parent, self.root):
                    return
                parent.rmdir()
            except OSError:
                return
            parent = parent.parent
