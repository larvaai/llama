from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .contracts import BackendCapabilities


class BackendMode(str, Enum):
    STEPPABLE = "steppable"
    MANAGED = "managed"


class BackendLifecycle(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    DRAINING = "draining"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    name: str
    mode: BackendMode
    capabilities: BackendCapabilities
    factory: Callable[[], object]
    priority: int = 0
    keepalive_seconds: float = 300.0

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or len(self.name.encode()) > 128:
            raise ValueError("backend descriptor name is invalid")
        if type(self.mode) is not BackendMode:
            raise TypeError("mode must be BackendMode")
        if type(self.capabilities) is not BackendCapabilities:
            raise TypeError("capabilities must be BackendCapabilities")
        if not callable(self.factory):
            raise TypeError("factory must be callable")
        if type(self.priority) is not int:
            raise TypeError("priority must be an integer")
        if (
            type(self.keepalive_seconds) not in {int, float}
            or not math.isfinite(self.keepalive_seconds)
            or self.keepalive_seconds < 0
        ):
            raise ValueError("keepalive_seconds must be finite and non-negative")
        object.__setattr__(self, "keepalive_seconds", float(self.keepalive_seconds))
        if self.mode is BackendMode.STEPPABLE and not self.capabilities.supports_sequence_steps:
            raise ValueError("steppable descriptor lacks sequence-step capability")
        if self.mode is BackendMode.MANAGED and not self.capabilities.supports_full_request:
            raise ValueError("managed descriptor lacks full-request capability")


@dataclass(frozen=True, slots=True)
class BackendRegistrySnapshot:
    name: str
    mode: BackendMode
    priority: int
    lifecycle: BackendLifecycle
    generation: int
    active_leases: int
    last_used: float
    failure: str | None
    capabilities: BackendCapabilities


@dataclass(slots=True)
class _Entry:
    descriptor: BackendDescriptor
    lifecycle: BackendLifecycle = BackendLifecycle.UNLOADED
    generation: int = 0
    instance: object | None = None
    active_leases: int = 0
    last_used: float = 0.0
    failure: str | None = None


class BackendRegistryError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class BackendLease:
    def __init__(
        self,
        registry: BackendRegistry,
        name: str,
        generation: int,
        instance: object,
    ) -> None:
        self._registry = registry
        self.name = name
        self.generation = generation
        self.instance = instance
        self._released = False

    def release(self) -> bool:
        if self._released:
            return False
        self._released = True
        return self._registry.release(self)

    def __enter__(self) -> object:
        return self.instance

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class BackendRegistry:
    """Lazy capability registry with bounded load/unload and keepalive."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._condition = threading.Condition()
        self._entries: dict[str, _Entry] = {}

    def register(self, descriptor: BackendDescriptor) -> None:
        if type(descriptor) is not BackendDescriptor:
            raise TypeError("descriptor must be BackendDescriptor")
        with self._condition:
            if descriptor.name in self._entries:
                raise BackendRegistryError("duplicate_backend", descriptor.name)
            self._entries[descriptor.name] = _Entry(descriptor, last_used=self._now())

    def acquire(
        self,
        model: str,
        *,
        require_mode: BackendMode | None = None,
        require_streaming: bool = False,
        require_cancellation: bool = False,
    ) -> BackendLease:
        candidates = self._candidate_names(
            model,
            require_mode=require_mode,
            require_streaming=require_streaming,
            require_cancellation=require_cancellation,
        )
        failures = []
        for name in candidates:
            try:
                return self._acquire_name(name)
            except BackendRegistryError as exc:
                failures.append(f"{name}={exc.code}")
        if not candidates:
            raise BackendRegistryError(
                "no_route",
                f"no backend capability route for model {model!r}",
            )
        raise BackendRegistryError(
            "all_backends_failed",
            ", ".join(failures),
        )

    def _candidate_names(
        self,
        model: str,
        *,
        require_mode: BackendMode | None,
        require_streaming: bool,
        require_cancellation: bool,
    ) -> list[str]:
        with self._condition:
            entries = [
                entry
                for entry in self._entries.values()
                if model in entry.descriptor.capabilities.models
                and (require_mode is None or entry.descriptor.mode is require_mode)
                and (
                    not require_streaming
                    or entry.descriptor.capabilities.supports_streaming
                )
                and (
                    not require_cancellation
                    or entry.descriptor.capabilities.supports_cancellation
                )
            ]
            entries.sort(
                key=lambda entry: (
                    -entry.descriptor.priority,
                    entry.descriptor.name,
                )
            )
            return [entry.descriptor.name for entry in entries]

    def _acquire_name(self, name: str) -> BackendLease:
        while True:
            with self._condition:
                entry = self._entries[name]
                if entry.lifecycle is BackendLifecycle.LOADING:
                    self._condition.wait()
                    continue
                if entry.lifecycle is BackendLifecycle.DRAINING:
                    raise BackendRegistryError("backend_draining", name)
                if entry.lifecycle is BackendLifecycle.READY:
                    assert entry.instance is not None
                    entry.active_leases += 1
                    entry.last_used = self._now()
                    return BackendLease(self, name, entry.generation, entry.instance)
                entry.lifecycle = BackendLifecycle.LOADING
                entry.failure = None
                descriptor = entry.descriptor
                break
        try:
            instance = descriptor.factory()
            capabilities = getattr(instance, "capabilities", None)
            if capabilities is not None and capabilities != descriptor.capabilities:
                raise BackendRegistryError(
                    "capability_mismatch",
                    f"loaded backend {name!r} differs from its declaration",
                )
        except Exception as exc:
            with self._condition:
                entry = self._entries[name]
                entry.lifecycle = BackendLifecycle.FAILED
                entry.failure = f"{type(exc).__name__}: {exc}"[:512]
                entry.instance = None
                self._condition.notify_all()
            if isinstance(exc, BackendRegistryError):
                raise
            raise BackendRegistryError("backend_load_failed", name) from exc
        with self._condition:
            entry = self._entries[name]
            entry.instance = instance
            entry.generation += 1
            entry.active_leases = 1
            entry.last_used = self._now()
            entry.lifecycle = BackendLifecycle.READY
            self._condition.notify_all()
            return BackendLease(self, name, entry.generation, instance)

    def release(self, lease: BackendLease) -> bool:
        with self._condition:
            entry = self._entries.get(lease.name)
            if (
                entry is None
                or entry.generation != lease.generation
                or entry.active_leases == 0
            ):
                return False
            entry.active_leases -= 1
            entry.last_used = self._now()
            self._condition.notify_all()
            return True

    def unload(self, name: str) -> bool:
        with self._condition:
            entry = self._entries.get(name)
            if entry is None:
                raise BackendRegistryError("unknown_backend", name)
            if entry.lifecycle is BackendLifecycle.UNLOADED:
                return False
            if entry.active_leases:
                raise BackendRegistryError("backend_busy", name)
            if entry.lifecycle not in {BackendLifecycle.READY, BackendLifecycle.FAILED}:
                raise BackendRegistryError("backend_state_conflict", name)
            instance = entry.instance
            entry.lifecycle = BackendLifecycle.DRAINING
        stopped = True
        shutdown = getattr(instance, "shutdown", None)
        if callable(shutdown):
            try:
                outcome = shutdown()
                stopped = outcome is not False
            except Exception:
                stopped = False
        with self._condition:
            entry = self._entries[name]
            entry.instance = None
            entry.lifecycle = (
                BackendLifecycle.UNLOADED if stopped else BackendLifecycle.FAILED
            )
            entry.failure = None if stopped else "shutdown_failed"
            entry.last_used = self._now()
            self._condition.notify_all()
        return stopped

    def sweep(self) -> tuple[str, ...]:
        now = self._now()
        with self._condition:
            targets = [
                entry.descriptor.name
                for entry in self._entries.values()
                if entry.lifecycle is BackendLifecycle.READY
                and entry.active_leases == 0
                and now - entry.last_used >= entry.descriptor.keepalive_seconds
            ]
        unloaded = []
        for name in targets:
            try:
                if self.unload(name):
                    unloaded.append(name)
            except BackendRegistryError:
                continue
        return tuple(unloaded)

    def snapshots(self) -> tuple[BackendRegistrySnapshot, ...]:
        with self._condition:
            return tuple(
                BackendRegistrySnapshot(
                    entry.descriptor.name,
                    entry.descriptor.mode,
                    entry.descriptor.priority,
                    entry.lifecycle,
                    entry.generation,
                    entry.active_leases,
                    entry.last_used,
                    entry.failure,
                    entry.descriptor.capabilities,
                )
                for entry in sorted(
                    self._entries.values(),
                    key=lambda item: item.descriptor.name,
                )
            )

    def preferred_mode(self, model: str) -> BackendMode:
        """Return the highest-priority declared route without loading it."""

        candidates = self._candidate_names(
            model,
            require_mode=None,
            require_streaming=False,
            require_cancellation=False,
        )
        if not candidates:
            raise BackendRegistryError("no_route", model)
        with self._condition:
            return self._entries[candidates[0]].descriptor.mode

    def _now(self) -> float:
        value = self._clock()
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise RuntimeError("registry clock must be finite and non-negative")
        return float(value)
