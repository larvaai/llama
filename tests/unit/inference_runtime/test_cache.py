from __future__ import annotations

import pytest

from inference_runtime import (
    CacheCapacityError,
    CacheKind,
    CacheNamespace,
    CacheScope,
    CacheVisibility,
    ExactTokenCacheKey,
    SequenceStateCache,
)


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def digest(character):
    return "sha256:" + character * 64


def namespace(**overrides):
    values = {
        "model_digest": digest("1"),
        "template_digest": digest("2"),
        "tokenizer_digest": digest("3"),
        "adapter_digest": digest("4"),
        "context_digest": digest("5"),
    }
    values.update(overrides)
    return CacheNamespace(**values)


def key(kind=CacheKind.PREFIX, tokens=(1, 2, 3), **namespace_overrides):
    return ExactTokenCacheKey(kind, namespace(**namespace_overrides), tokens)


def scope(
    tenant="tenant-a",
    workflow="workflow-a",
    agent="agent-a",
    visibility=CacheVisibility.PRIVATE,
):
    return CacheScope(tenant, workflow, agent, visibility)


def test_exact_token_and_every_runtime_digest_participate_in_key():
    original = key()
    assert original != key(tokens=(1, 2, 4))
    assert original != key(model_digest=digest("a"))
    assert original != key(template_digest=digest("b"))
    assert original != key(tokenizer_digest=digest("c"))
    assert original != key(adapter_digest=digest("d"))
    assert original != key(context_digest=digest("e"))
    assert original.digest.startswith("sha256:")


def test_scope_never_crosses_tenant_and_respects_private_workflow_tenant_levels():
    cache = SequenceStateCache(byte_budget=100, max_entries=8, ttl_seconds=10)
    cache.put(key(), scope(), b"private")
    assert cache.acquire(key(), scope(agent="other")) is None
    assert cache.acquire(key(), scope(tenant="tenant-b")) is None

    workflow_key = key(tokens=(4,))
    cache.put(
        workflow_key,
        scope(visibility=CacheVisibility.WORKFLOW),
        b"workflow",
    )
    lease = cache.acquire(workflow_key, scope(agent="other"))
    assert cache.read(lease) == b"workflow"
    assert cache.release(lease)

    tenant_key = key(tokens=(5,))
    cache.put(
        tenant_key,
        scope(visibility=CacheVisibility.TENANT),
        b"tenant",
    )
    assert cache.acquire(tenant_key, scope(workflow="other", agent="other"))
    assert cache.acquire(tenant_key, scope(tenant="tenant-b")) is None


def test_prefix_is_immutable_and_session_uses_copy_on_write():
    cache = SequenceStateCache(byte_budget=100, max_entries=8, ttl_seconds=10)
    prefix = key()
    cache.put(prefix, scope(), b"prefix")
    with pytest.raises(ValueError, match="immutable"):
        cache.acquire(prefix, scope(), mutable=True)

    session = key(CacheKind.SESSION, tokens=(7, 8))
    cache.put(session, scope(), b"session-v1")
    first = cache.acquire(session, scope(), mutable=True)
    second = cache.acquire(session, scope(), mutable=True)
    assert not first.copy_on_write
    assert second.copy_on_write
    assert first.entry_id != second.entry_id
    assert cache.read(first) == cache.read(second) == b"session-v1"

    updated = key(CacheKind.SESSION, tokens=(7, 8, 9))
    cache.commit_session(second, updated, b"session-v2", scope())
    assert cache.release(first)
    lease = cache.acquire(updated, scope())
    assert cache.read(lease) == b"session-v2"
    metrics = cache.metrics()
    assert metrics.cow_clones == 1


def test_ttl_lru_byte_budget_and_pinned_leases_are_bounded():
    clock = Clock()
    cache = SequenceStateCache(
        byte_budget=6,
        max_entries=2,
        ttl_seconds=5,
        clock=clock,
    )
    first_key = key(tokens=(1,))
    second_key = key(tokens=(2,))
    third_key = key(tokens=(3,))
    cache.put(first_key, scope(), b"111")
    clock.value = 1
    cache.put(second_key, scope(), b"222")
    lease = cache.acquire(first_key, scope())
    clock.value = 2
    cache.put(third_key, scope(), b"333")
    assert cache.acquire(second_key, scope()) is None
    second_lease = cache.acquire(first_key, scope())
    assert second_lease

    with pytest.raises(CacheCapacityError, match="pinned"):
        cache.put(key(tokens=(4,)), scope(), b"444444")
    assert cache.release(lease)
    assert cache.release(second_lease)
    clock.value = 20
    assert cache.prune() >= 1
    assert cache.metrics().bytes_used <= 6


def test_invalidation_skips_leased_entry_then_removes_it_after_release():
    cache = SequenceStateCache(byte_budget=100, max_entries=8, ttl_seconds=10)
    cache.put(key(), scope(), b"state")
    lease = cache.acquire(key(), scope())
    assert cache.invalidate(tenant_id="tenant-a") == 0
    assert cache.release(lease)
    assert cache.invalidate(tenant_id="tenant-a", kind=CacheKind.PREFIX) == 1
    assert cache.acquire(key(), scope()) is None
    metrics = cache.metrics()
    assert metrics.invalidations == 1
    assert metrics.saved_prefill_tokens == 3
