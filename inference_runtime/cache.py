from __future__ import annotations

import hashlib
import math
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Callable


def _identifier(value: str, name: str) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > 256:
        raise ValueError(f"{name} must be 1..256 UTF-8 bytes")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _digest(value: str, name: str) -> str:
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{name} must be a lowercase sha256 digest")
    return value


class CacheKind(str, Enum):
    PREFIX = "prefix"
    SESSION = "session"


class CacheVisibility(str, Enum):
    PRIVATE = "private"
    WORKFLOW = "workflow"
    TENANT = "tenant"


@dataclass(frozen=True, slots=True)
class CacheScope:
    tenant_id: str
    workflow_id: str
    agent_id: str
    visibility: CacheVisibility = CacheVisibility.PRIVATE

    def __post_init__(self) -> None:
        _identifier(self.tenant_id, "tenant_id")
        _identifier(self.workflow_id, "workflow_id")
        _identifier(self.agent_id, "agent_id")
        if type(self.visibility) is not CacheVisibility:
            raise TypeError("visibility must be CacheVisibility")

    def allows(self, requester: CacheScope) -> bool:
        if type(requester) is not CacheScope:
            return False
        if self.tenant_id != requester.tenant_id:
            return False
        if self.visibility is CacheVisibility.TENANT:
            return True
        if self.workflow_id != requester.workflow_id:
            return False
        if self.visibility is CacheVisibility.WORKFLOW:
            return True
        return self.agent_id == requester.agent_id


@dataclass(frozen=True, slots=True)
class CacheNamespace:
    model_digest: str
    template_digest: str
    tokenizer_digest: str
    adapter_digest: str
    context_digest: str

    def __post_init__(self) -> None:
        for name in (
            "model_digest",
            "template_digest",
            "tokenizer_digest",
            "adapter_digest",
            "context_digest",
        ):
            _digest(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ExactTokenCacheKey:
    kind: CacheKind
    namespace: CacheNamespace
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.kind) is not CacheKind:
            raise TypeError("kind must be CacheKind")
        if type(self.namespace) is not CacheNamespace:
            raise TypeError("namespace must be CacheNamespace")
        if type(self.token_ids) is not tuple or not self.token_ids:
            raise ValueError("token_ids must be a non-empty tuple")
        for token in self.token_ids:
            if type(token) is not int or token < 0:
                raise ValueError("token_ids must contain non-negative integers")

    @property
    def digest(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.kind.value.encode())
        for value in (
            self.namespace.model_digest,
            self.namespace.template_digest,
            self.namespace.tokenizer_digest,
            self.namespace.adapter_digest,
            self.namespace.context_digest,
        ):
            hasher.update(value.encode())
        for token in self.token_ids:
            hasher.update(token.to_bytes(4, "little", signed=False))
        return "sha256:" + hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class CacheLease:
    entry_id: str
    generation: int
    key: ExactTokenCacheKey
    mutable: bool
    copy_on_write: bool


@dataclass(frozen=True, slots=True)
class CacheMetrics:
    hits: int
    misses: int
    denied: int
    insertions: int
    evictions: int
    invalidations: int
    cow_clones: int
    saved_prefill_tokens: int
    bytes_used: int
    entries: int
    leased_entries: int


@dataclass(slots=True)
class _Entry:
    entry_id: str
    generation: int
    key: ExactTokenCacheKey
    scope: CacheScope
    payload: bytes
    byte_size: int
    created_at: float
    expires_at: float
    last_access: float
    refcount: int = 0
    ephemeral: bool = False


class CacheCapacityError(RuntimeError):
    pass


class SequenceStateCache:
    """Bounded exact-token prefix/session cache with tenant-safe COW leases."""

    def __init__(
        self,
        *,
        byte_budget: int,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(byte_budget) is not int or byte_budget <= 0:
            raise ValueError("byte_budget must be a positive integer")
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        if type(ttl_seconds) not in {int, float} or not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.byte_budget = byte_budget
        self.max_entries = max_entries
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry] = {}
        self._index: dict[tuple[ExactTokenCacheKey, CacheScope], str] = {}
        self._bytes_used = 0
        self._hits = 0
        self._misses = 0
        self._denied = 0
        self._insertions = 0
        self._evictions = 0
        self._invalidations = 0
        self._cow_clones = 0
        self._saved_prefill_tokens = 0

    def put(
        self,
        key: ExactTokenCacheKey,
        scope: CacheScope,
        payload: bytes,
    ) -> str:
        self._validate_values(key, scope, payload)
        now = self._now()
        with self._lock:
            self._prune_locked(now)
            index_key = (key, scope)
            existing_id = self._index.get(index_key)
            if existing_id is not None:
                existing = self._entries[existing_id]
                if existing.refcount:
                    raise CacheCapacityError("cannot replace a leased cache entry")
                self._remove_locked(existing, invalidation=True)
            self._ensure_capacity_locked(len(payload), now)
            entry_id = uuid.uuid4().hex
            entry = _Entry(
                entry_id,
                1,
                key,
                scope,
                bytes(payload),
                len(payload),
                now,
                now + self.ttl_seconds,
                now,
            )
            self._entries[entry_id] = entry
            self._index[index_key] = entry_id
            self._bytes_used += entry.byte_size
            self._insertions += 1
            return entry_id

    def acquire(
        self,
        key: ExactTokenCacheKey,
        requester: CacheScope,
        *,
        mutable: bool = False,
    ) -> CacheLease | None:
        if type(key) is not ExactTokenCacheKey or type(requester) is not CacheScope:
            raise TypeError("key and requester must be validated cache contracts")
        if type(mutable) is not bool:
            raise TypeError("mutable must be a bool")
        if mutable and key.kind is CacheKind.PREFIX:
            raise ValueError("prefix cache leases are immutable")
        now = self._now()
        with self._lock:
            self._prune_locked(now)
            matching = [
                entry
                for entry in self._entries.values()
                if not entry.ephemeral and entry.key == key
            ]
            allowed = [entry for entry in matching if entry.scope.allows(requester)]
            if not allowed:
                if matching:
                    self._denied += 1
                else:
                    self._misses += 1
                return None
            # Prefer the narrowest scope and then most-recently used entry.
            visibility_rank = {
                CacheVisibility.PRIVATE: 0,
                CacheVisibility.WORKFLOW: 1,
                CacheVisibility.TENANT: 2,
            }
            entry = min(
                allowed,
                key=lambda item: (visibility_rank[item.scope.visibility], -item.last_access),
            )
            copy_on_write = mutable and entry.refcount > 0
            if copy_on_write:
                self._ensure_capacity_locked(entry.byte_size, now)
                clone_id = uuid.uuid4().hex
                clone = _Entry(
                    clone_id,
                    entry.generation + 1,
                    entry.key,
                    requester,
                    bytes(entry.payload),
                    entry.byte_size,
                    now,
                    now + self.ttl_seconds,
                    now,
                    refcount=1,
                    ephemeral=True,
                )
                self._entries[clone_id] = clone
                self._bytes_used += clone.byte_size
                self._cow_clones += 1
                leased = clone
            else:
                entry.refcount += 1
                entry.last_access = now
                entry.expires_at = now + self.ttl_seconds
                leased = entry
            self._hits += 1
            self._saved_prefill_tokens += len(key.token_ids)
            return CacheLease(
                leased.entry_id,
                leased.generation,
                leased.key,
                mutable,
                copy_on_write,
            )

    def read(self, lease: CacheLease) -> bytes:
        if type(lease) is not CacheLease:
            raise TypeError("lease must be CacheLease")
        with self._lock:
            entry = self._lease_entry_locked(lease)
            return bytes(entry.payload)

    def release(self, lease: CacheLease) -> bool:
        if type(lease) is not CacheLease:
            raise TypeError("lease must be CacheLease")
        with self._lock:
            entry = self._entries.get(lease.entry_id)
            if entry is None or entry.generation != lease.generation or entry.refcount == 0:
                return False
            entry.refcount -= 1
            if entry.ephemeral and entry.refcount == 0:
                self._remove_locked(entry, invalidation=False)
            return True

    def commit_session(
        self,
        lease: CacheLease,
        new_key: ExactTokenCacheKey,
        payload: bytes,
        scope: CacheScope,
    ) -> str:
        if type(lease) is not CacheLease or not lease.mutable:
            raise ValueError("commit_session requires a mutable lease")
        if new_key.kind is not CacheKind.SESSION:
            raise ValueError("session commits require a session cache key")
        with self._lock:
            self._lease_entry_locked(lease)
        entry_id = self.put(new_key, scope, payload)
        self.release(lease)
        return entry_id

    def invalidate(
        self,
        *,
        tenant_id: str,
        kind: CacheKind | None = None,
        namespace: CacheNamespace | None = None,
    ) -> int:
        _identifier(tenant_id, "tenant_id")
        if kind is not None and type(kind) is not CacheKind:
            raise TypeError("kind must be CacheKind or None")
        if namespace is not None and type(namespace) is not CacheNamespace:
            raise TypeError("namespace must be CacheNamespace or None")
        with self._lock:
            targets = [
                entry
                for entry in self._entries.values()
                if entry.scope.tenant_id == tenant_id
                and (kind is None or entry.key.kind is kind)
                and (namespace is None or entry.key.namespace == namespace)
                and entry.refcount == 0
            ]
            for entry in targets:
                self._remove_locked(entry, invalidation=True)
            return len(targets)

    def prune(self) -> int:
        with self._lock:
            before = len(self._entries)
            self._prune_locked(self._now())
            return before - len(self._entries)

    def metrics(self) -> CacheMetrics:
        with self._lock:
            return CacheMetrics(
                self._hits,
                self._misses,
                self._denied,
                self._insertions,
                self._evictions,
                self._invalidations,
                self._cow_clones,
                self._saved_prefill_tokens,
                self._bytes_used,
                len(self._entries),
                sum(entry.refcount > 0 for entry in self._entries.values()),
            )

    def _validate_values(
        self,
        key: ExactTokenCacheKey,
        scope: CacheScope,
        payload: bytes,
    ) -> None:
        if type(key) is not ExactTokenCacheKey or type(scope) is not CacheScope:
            raise TypeError("key and scope must be validated cache contracts")
        if type(payload) is not bytes or not payload:
            raise ValueError("payload must be non-empty bytes")
        if len(payload) > self.byte_budget:
            raise CacheCapacityError("cache entry exceeds the byte budget")

    def _now(self) -> float:
        value = self._clock()
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise RuntimeError("cache clock must return a finite non-negative timestamp")
        return float(value)

    def _ensure_capacity_locked(self, byte_size: int, now: float) -> None:
        self._prune_locked(now)
        while (
            self._bytes_used + byte_size > self.byte_budget
            or len(self._entries) >= self.max_entries
        ):
            evictable = [entry for entry in self._entries.values() if entry.refcount == 0]
            if not evictable:
                raise CacheCapacityError("cache capacity is pinned by active leases")
            victim = min(evictable, key=lambda entry: (entry.last_access, entry.created_at))
            self._remove_locked(victim, invalidation=False)
            self._evictions += 1

    def _prune_locked(self, now: float) -> None:
        expired = [
            entry
            for entry in self._entries.values()
            if entry.refcount == 0 and entry.expires_at <= now
        ]
        for entry in expired:
            self._remove_locked(entry, invalidation=False)
            self._evictions += 1

    def _remove_locked(self, entry: _Entry, *, invalidation: bool) -> None:
        self._entries.pop(entry.entry_id, None)
        if not entry.ephemeral:
            self._index.pop((entry.key, entry.scope), None)
        self._bytes_used -= entry.byte_size
        if invalidation:
            self._invalidations += 1

    def _lease_entry_locked(self, lease: CacheLease) -> _Entry:
        entry = self._entries.get(lease.entry_id)
        if (
            entry is None
            or entry.generation != lease.generation
            or entry.key != lease.key
            or entry.refcount == 0
        ):
            raise RuntimeError("cache lease is stale")
        return entry
