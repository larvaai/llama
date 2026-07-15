from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from inference_runtime import (
    BackendCapabilities,
    BackendDescriptor,
    BackendLifecycle,
    BackendMode,
    BackendRegistry,
    BackendRegistryError,
    InferenceRouter,
    SchedulingMetadata,
)


def managed_capabilities(name="managed", model="qwen35-9b-local"):
    return BackendCapabilities(
        backend=name,
        models=(model,),
        supports_full_request=True,
        supports_sequence_steps=False,
        supports_streaming=False,
        supports_cancellation=False,
        supports_chunked_prefill=False,
        supports_decode_batching=False,
        supports_continuous_batching=False,
        supports_prefix_cache=False,
        supports_session_cache=False,
        supports_explicit_release=False,
        max_context_tokens=1024,
        max_output_tokens=128,
        max_concurrent_requests=4,
        max_concurrent_sequences=None,
        max_prefill_tokens_per_step=None,
        max_decode_tokens_per_step=None,
        max_sequences_per_step=None,
    )


class FakeManaged:
    def __init__(self, capabilities, result="managed"):
        self.capabilities = capabilities
        self.result = result
        self.generate_calls = 0
        self.shutdown_calls = 0

    def generate(self, request, *, scheduling, events):
        self.generate_calls += 1
        return self.result

    def cancel(self, request_id):
        return False

    def shutdown(self):
        self.shutdown_calls += 1
        return True


class FakeInferencePort:
    def __init__(self, capabilities, result="steppable-control-plane"):
        self.capabilities = capabilities
        self.result = result

    def infer(self, request, *, scheduling, events):
        return self.result

    def cancel(self, request_id):
        return False

    def shutdown(self):
        return True


def scheduling(request_id="route-1"):
    return SchedulingMetadata(
        request_id,
        "workflow",
        "agent",
        "throughput",
        1,
        None,
    )


def routed_request(model="qwen35-9b-local"):
    return SimpleNamespace(request=SimpleNamespace(model_id=model))


def test_registry_is_lazy_routes_by_priority_and_keeps_leases():
    created = []
    low_caps = managed_capabilities("low")
    high_caps = managed_capabilities("high")
    registry = BackendRegistry()
    registry.register(
        BackendDescriptor(
            "low",
            BackendMode.MANAGED,
            low_caps,
            lambda: created.append("low") or FakeManaged(low_caps),
            priority=1,
        )
    )
    registry.register(
        BackendDescriptor(
            "high",
            BackendMode.MANAGED,
            high_caps,
            lambda: created.append("high") or FakeManaged(high_caps),
            priority=10,
        )
    )

    assert created == []
    assert registry.preferred_mode("qwen35-9b-local") is BackendMode.MANAGED
    first = registry.acquire("qwen35-9b-local")
    second = registry.acquire("qwen35-9b-local")
    assert created == ["high"]
    assert first.instance is second.instance
    assert registry.snapshots()[0].priority in {1, 10}
    with pytest.raises(BackendRegistryError, match="backend_busy"):
        registry.unload("high")
    assert first.release()
    assert second.release()
    assert not second.release()
    assert registry.unload("high")


def test_registry_falls_back_after_higher_priority_load_failure():
    low_caps = managed_capabilities("low")
    high_caps = managed_capabilities("high")
    low = FakeManaged(low_caps)
    registry = BackendRegistry()
    registry.register(
        BackendDescriptor(
            "high",
            BackendMode.MANAGED,
            high_caps,
            lambda: (_ for _ in ()).throw(RuntimeError("offline")),
            priority=10,
        )
    )
    registry.register(
        BackendDescriptor(
            "low",
            BackendMode.MANAGED,
            low_caps,
            lambda: low,
            priority=1,
        )
    )

    lease = registry.acquire("qwen35-9b-local")
    assert lease.instance is low
    assert lease.name == "low"
    assert lease.release()
    snapshots = {snapshot.name: snapshot for snapshot in registry.snapshots()}
    assert snapshots["high"].lifecycle is BackendLifecycle.FAILED
    assert snapshots["high"].failure.startswith("RuntimeError")


def test_keepalive_sweep_unloads_only_idle_backend():
    now = [0.0]
    caps = managed_capabilities()
    instance = FakeManaged(caps)
    registry = BackendRegistry(clock=lambda: now[0])
    registry.register(
        BackendDescriptor(
            "managed",
            BackendMode.MANAGED,
            caps,
            lambda: instance,
            keepalive_seconds=5,
        )
    )
    lease = registry.acquire("qwen35-9b-local")
    now[0] = 10
    assert registry.sweep() == ()
    assert lease.release()
    now[0] = 16
    assert registry.sweep() == ("managed",)
    assert instance.shutdown_calls == 1


def test_router_hides_backend_mode_and_releases_registry_lease():
    managed_caps = managed_capabilities("managed")
    sequence_caps = BackendCapabilities(
        backend="llama.cpp",
        models=("other-model",),
        supports_full_request=False,
        supports_sequence_steps=True,
        supports_streaming=True,
        supports_cancellation=True,
        supports_chunked_prefill=True,
        supports_decode_batching=True,
        supports_continuous_batching=True,
        supports_prefix_cache=False,
        supports_session_cache=False,
        supports_explicit_release=True,
        max_context_tokens=1024,
        max_output_tokens=128,
        max_concurrent_requests=4,
        max_concurrent_sequences=4,
        max_prefill_tokens_per_step=64,
        max_decode_tokens_per_step=8,
        max_sequences_per_step=4,
    )
    registry = BackendRegistry()
    registry.register(
        BackendDescriptor(
            "managed",
            BackendMode.MANAGED,
            managed_caps,
            lambda: FakeManaged(managed_caps),
        )
    )
    registry.register(
        BackendDescriptor(
            "llama",
            BackendMode.STEPPABLE,
            sequence_caps,
            lambda: FakeInferencePort(sequence_caps),
        )
    )
    router = InferenceRouter(registry)

    assert router.infer(
        routed_request(), scheduling=scheduling(), events=object()
    ) == "managed"
    snapshot = {item.name: item for item in registry.snapshots()}["managed"]
    assert snapshot.active_leases == 0
    assert router.preferred_mode("other-model") is BackendMode.STEPPABLE


def test_router_rejects_duplicate_active_request():
    caps = managed_capabilities()
    entered = threading.Event()
    release = threading.Event()

    class BlockingManaged(FakeManaged):
        def generate(self, request, *, scheduling, events):
            entered.set()
            release.wait(2)
            return "done"

    registry = BackendRegistry()
    registry.register(
        BackendDescriptor(
            "managed",
            BackendMode.MANAGED,
            caps,
            lambda: BlockingManaged(caps),
        )
    )
    router = InferenceRouter(registry)
    outcome = []
    thread = threading.Thread(
        target=lambda: outcome.append(
            router.infer(
                routed_request(), scheduling=scheduling("same"), events=object()
            )
        )
    )
    thread.start()
    assert entered.wait(1)
    with pytest.raises(BackendRegistryError, match="duplicate_request"):
        router.infer(
            routed_request(), scheduling=scheduling("same"), events=object()
        )
    release.set()
    thread.join(2)
    assert outcome == ["done"]
