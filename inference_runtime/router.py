from __future__ import annotations

import threading

from model_worker.preflight import PreflightedRequest

from .contracts import SchedulingMetadata
from .ports import InferencePort, ManagedBackend, SchedulerEventSink
from .registry import BackendLease, BackendMode, BackendRegistry, BackendRegistryError


class InferenceRouter:
    """High-level model/capability router; never exposes sequence APIs to harness."""

    def __init__(self, registry: BackendRegistry) -> None:
        if type(registry) is not BackendRegistry:
            raise TypeError("registry must be BackendRegistry")
        self.registry = registry
        self._lock = threading.RLock()
        self._active: dict[str, tuple[BackendLease, object]] = {}

    def infer(
        self,
        request: PreflightedRequest,
        *,
        scheduling: SchedulingMetadata,
        events: SchedulerEventSink,
    ):
        with self._lock:
            if scheduling.request_id in self._active:
                raise BackendRegistryError("duplicate_request", scheduling.request_id)
        lease = self.registry.acquire(request.request.model_id)
        target = lease.instance
        with self._lock:
            if scheduling.request_id in self._active:
                lease.release()
                raise BackendRegistryError("duplicate_request", scheduling.request_id)
            self._active[scheduling.request_id] = (lease, target)
        try:
            if isinstance(target, InferencePort):
                return target.infer(request, scheduling=scheduling, events=events)
            if isinstance(target, ManagedBackend):
                return target.generate(request, scheduling=scheduling, events=events)
            raise BackendRegistryError(
                "invalid_route",
                f"backend {lease.name!r} is not a high-level inference target",
            )
        finally:
            with self._lock:
                self._active.pop(scheduling.request_id, None)
            lease.release()

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            active = self._active.get(request_id)
        if active is None:
            return False
        target = active[1]
        cancel = getattr(target, "cancel", None)
        return bool(cancel(request_id)) if callable(cancel) else False

    def preferred_mode(self, model: str) -> BackendMode:
        return self.registry.preferred_mode(model)
