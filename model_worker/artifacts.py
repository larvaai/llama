from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import WorkerError


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
        payload = {"request_hash": _canonical_hash(request_hash_source), "contract_hash": _canonical_hash(contract), "model_manifest_digest": model["manifest_digest"], "runtime_build": model["runtime_build"], "limits": limits, "timestamps": timestamps}
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


class ArtifactStore:
    def __init__(self, root: Path, *, total_quota: int, retention_seconds: int) -> None:
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

    def cleanup(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        removed = 0
        with self.lock:
            candidates = []
            for result in self.root.rglob("result.json"):
                attempt = result.parent.resolve()
                if self.root not in attempt.parents or attempt in self.active or attempt.is_symlink(): continue
                candidates.append((result.stat().st_mtime, attempt))
            total = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())
            for modified, attempt in sorted(candidates):
                size = sum(path.stat().st_size for path in attempt.rglob("*") if path.is_file())
                if now - modified > self.retention_seconds or total > self.total_quota:
                    shutil.rmtree(attempt); total -= size; removed += 1
        return removed
