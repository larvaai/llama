from __future__ import annotations

import pytest

from inference_runtime import (
    BackendCapabilities,
    BackendConformanceError,
    BackendMode,
    DecodeOutcome,
    PrefillOutcome,
    ReleaseOutcome,
    inspect_backend_conformance,
)
from inference_runtime.adapters import (
    MLXCompletion,
    MlxLMManagedBackend,
    SGLangManagedBackend,
    VLLMManagedBackend,
)


class FakeTransport:
    supports_cancellation = False

    def post_json(self, path, payload, *, timeout, request_id):
        raise AssertionError("static conformance must not dispatch")

    def cancel(self, request_id):
        return False


class FakeMLXRuntime:
    def complete(self, messages, *, max_tokens):
        return MLXCompletion("{}", 1, 1)

    def shutdown(self):
        return True


def managed(adapter):
    if adapter is MlxLMManagedBackend:
        return adapter(
            FakeMLXRuntime(),
            models=("model",),
            max_context_tokens=128,
            max_output_tokens=32,
            max_concurrent_requests=1,
        )
    return adapter(
        FakeTransport(),
        models=("model",),
        max_context_tokens=128,
        max_output_tokens=32,
        max_concurrent_requests=1,
    )


def steppable_capabilities():
    return BackendCapabilities(
        backend="fake-step",
        models=("model",),
        supports_full_request=False,
        supports_sequence_steps=True,
        supports_streaming=True,
        supports_cancellation=True,
        supports_chunked_prefill=True,
        supports_decode_batching=False,
        supports_continuous_batching=False,
        supports_prefix_cache=False,
        supports_session_cache=False,
        supports_explicit_release=True,
        max_context_tokens=128,
        max_output_tokens=32,
        max_concurrent_requests=1,
        max_concurrent_sequences=1,
        max_prefill_tokens_per_step=128,
        max_decode_tokens_per_step=1,
        max_sequences_per_step=1,
    )


class FakeSteppable:
    capabilities = steppable_capabilities()

    def open_sequence(self, request, *, scheduling, events):
        raise AssertionError

    def prefill(self, handle, *, token_budget, events) -> PrefillOutcome:
        raise AssertionError

    def decode(self, handle, *, token_budget, events) -> DecodeOutcome:
        raise AssertionError

    def release(self, handle, *, events) -> ReleaseOutcome:
        raise AssertionError


@pytest.mark.parametrize(
    "adapter",
    [VLLMManagedBackend, SGLangManagedBackend, MlxLMManagedBackend],
)
def test_common_conformance_accepts_managed_adapters_without_sequence_leak(adapter):
    report = inspect_backend_conformance(
        managed(adapter),
        mode=BackendMode.MANAGED,
    )
    assert report.passed
    assert "no_sequence_api_leak" in report.checks


def test_common_conformance_accepts_steppable_compute_boundary():
    report = inspect_backend_conformance(
        FakeSteppable(),
        mode=BackendMode.STEPPABLE,
    )
    assert report.backend == "fake-step"
    assert "explicit_release_boundary" in report.checks


def test_common_conformance_rejects_mode_mismatch():
    with pytest.raises(BackendConformanceError):
        inspect_backend_conformance(
            managed(VLLMManagedBackend),
            mode=BackendMode.STEPPABLE,
        )
