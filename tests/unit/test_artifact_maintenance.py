from __future__ import annotations

import os

import pytest

from model_worker.artifacts import ArtifactCleanupStats, ArtifactStore


MODEL = {
    "manifest_digest": "sha256:model",
    "runtime_build": "b10012",
}


def make_completed(
    store: ArtifactStore,
    request_id: str,
    attempt_id: str,
    modified: float,
):
    artifact = store.begin(request_id, attempt_id, 10_000)
    artifact.write_manifest({}, {}, MODEL, {}, {})
    artifact.write_result({"termination": "completed", "padding": "x" * 32})
    store.finish(artifact)
    os.utime(artifact.path / "result.json", (modified, modified))
    return artifact


def make_incomplete(
    store: ArtifactStore,
    request_id: str,
    attempt_id: str,
    modified: float,
    *,
    active: bool = False,
):
    artifact = store.begin(request_id, attempt_id, 10_000)
    artifact.write_manifest({}, {}, MODEL, {}, {})
    os.utime(artifact.path / "manifest.json", (modified, modified))
    os.utime(artifact.path, (modified, modified))
    if not active:
        store.finish(artifact)
    return artifact


def test_cleanup_removes_expired_completed_and_crash_incomplete_but_never_active(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", total_quota=1_000_000, retention_seconds=10)
    completed = make_completed(store, "requestone", "attemptone", 10)
    incomplete = make_incomplete(store, "requesttwo", "attempttwo", 20)
    active = make_incomplete(store, "requestthree", "attemptthree", 5, active=True)

    stats = store.cleanup_with_stats(now=100)
    assert isinstance(stats, ArtifactCleanupStats)
    assert not completed.path.exists()
    assert not incomplete.path.exists()
    assert active.path.exists()
    assert stats.scanned_attempts == 3
    assert stats.completed_attempts == 1
    assert stats.incomplete_attempts == 1
    assert stats.active_attempts_skipped == 1
    assert stats.removed_attempts == 2
    assert stats.removed_completed_attempts == 1
    assert stats.removed_incomplete_attempts == 1
    assert stats.removed_expired_attempts == 2
    assert stats.removed_for_quota_attempts == 0
    assert stats.bytes_before > stats.bytes_after
    assert stats.as_dict()["quota_satisfied"] is True


def test_quota_eviction_is_deterministic_oldest_first_and_can_bound_removals(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", total_quota=1_000_000, retention_seconds=10_000)
    oldest = make_completed(store, "requestone", "attemptone", 100)
    middle = make_completed(store, "requesttwo", "attempttwo", 200)
    newest = make_completed(store, "requestthree", "attemptthree", 300)
    newest_size = sum(path.stat().st_size for path in newest.path.rglob("*") if path.is_file())
    store.total_quota = newest_size

    first = store.cleanup_with_stats(now=400, max_removals=1)
    assert not oldest.path.exists()
    assert middle.path.exists() and newest.path.exists()
    assert first.removed_for_quota_attempts == 1
    assert first.removal_limit_reached is True
    assert first.quota_satisfied is False

    second = store.cleanup_with_stats(now=400)
    assert not middle.path.exists()
    assert newest.path.exists()
    assert second.removed_for_quota_attempts == 1
    assert second.quota_satisfied is True


def test_quota_tie_breaker_uses_stable_relative_path_order(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", total_quota=1_000_000, retention_seconds=10_000)
    first = make_completed(store, "requesta", "attempt", 100)
    second = make_completed(store, "requestb", "attempt", 100)
    second_size = sum(path.stat().st_size for path in second.path.rglob("*") if path.is_file())
    store.total_quota = second_size

    stats = store.cleanup_with_stats(now=200)
    assert not first.path.exists()
    assert second.path.exists()
    assert stats.removed_for_quota_attempts == 1


def test_cleanup_legacy_count_api_delegates_to_stats_core(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", total_quota=1_000_000, retention_seconds=1)
    artifact = make_incomplete(store, "request", "attempt", 1)
    assert store.cleanup(now=10) == 1
    assert not artifact.path.exists()


def test_cleanup_never_follows_attempt_symlink_or_deletes_outside_root(tmp_path):
    root = tmp_path / "artifacts"
    store = ArtifactStore(root, total_quota=0, retention_seconds=0)
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    request = root / "2026-01-01" / "request"
    request.mkdir(parents=True)
    attempt_link = request / "attempt"
    try:
        os.symlink(outside, attempt_link, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")

    stats = store.cleanup_with_stats(now=100)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert attempt_link.is_symlink()
    assert stats.removed_attempts == 0
    assert stats.unsafe_entries_skipped >= 1


def test_cleanup_skips_safe_attempt_containing_a_symlink(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts", total_quota=0, retention_seconds=0)
    artifact = make_incomplete(store, "request", "attempt", 1)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")
    linked = artifact.path / "linked.txt"
    try:
        os.symlink(outside, linked)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"file symlink unavailable: {exc}")

    stats = store.cleanup_with_stats(now=100)
    assert artifact.path.exists()
    assert outside.read_text(encoding="utf-8") == "keep"
    assert stats.removed_attempts == 0
    assert stats.unsafe_entries_skipped >= 1


@pytest.mark.parametrize("max_removals", [-1, 1.5, True])
def test_cleanup_rejects_invalid_removal_bound(tmp_path, max_removals):
    store = ArtifactStore(tmp_path / "artifacts", total_quota=0, retention_seconds=0)
    with pytest.raises(ValueError, match="max_removals"):
        store.cleanup_with_stats(max_removals=max_removals)
