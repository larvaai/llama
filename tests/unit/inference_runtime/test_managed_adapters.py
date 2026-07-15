from __future__ import annotations

import threading

import pytest

from inference_runtime import SchedulerEventKind, SchedulingMetadata
from inference_runtime.adapters import (
    ManagedBackendError,
    SGLangManagedBackend,
    VLLMManagedBackend,
)
from model_worker.preflight import preflight


class Sink:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)


class FakeTransport:
    def __init__(self, response, *, cancellable=False):
        self.response = response
        self.supports_cancellation = cancellable
        self.calls = []
        self.cancelled = []

    def post_json(self, path, payload, *, timeout, request_id):
        self.calls.append((path, payload, timeout, request_id))
        return self.response

    def cancel(self, request_id):
        self.cancelled.append(request_id)
        return self.supports_cancellation


def scheduling(request_id="managed-1"):
    return SchedulingMetadata(
        request_id,
        "workflow",
        "agent",
        "throughput",
        1,
        None,
    )


def prepared(manifest, request_body):
    return preflight(request_body, manifest)


def backend(cls, transport):
    return cls(
        transport,
        models=("qwen35-9b-local",),
        max_context_tokens=1024,
        max_output_tokens=128,
        max_concurrent_requests=2,
    )


def completion(content='{"result":"ok"}'):
    return {
        "id": "provider-1",
        "model": "qwen35-9b-local",
        "system_fingerprint": "fingerprint",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5},
    }


@pytest.mark.parametrize("adapter", [VLLMManagedBackend, SGLangManagedBackend])
def test_managed_adapter_builds_json_schema_request_and_revalidates_output(
    adapter,
    manifest,
    request_body,
):
    transport = FakeTransport(completion())
    target = backend(adapter, transport)
    sink = Sink()

    result = target.generate(
        prepared(manifest, request_body),
        scheduling=scheduling(),
        events=sink,
    )

    assert result.output == {"result": "ok"}
    assert result.usage["prompt_tokens"] == 20
    assert result.model["backend"] in {"vllm", "sglang"}
    path, payload, timeout, request_id = transport.calls[0]
    assert path == "/v1/chat/completions"
    assert request_id == "managed-1"
    assert timeout > 0
    assert payload["stream"] is False
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["messages"][0]["role"] == "system"
    assert [event.kind for event in sink.events] == [
        SchedulerEventKind.ADMITTED,
        SchedulerEventKind.REQUEST_COMPLETED,
    ]


def test_managed_adapter_fails_closed_on_contract_violation(manifest, request_body):
    transport = FakeTransport(completion('{"wrong":"value"}'))
    target = backend(VLLMManagedBackend, transport)
    sink = Sink()

    with pytest.raises(ManagedBackendError) as error:
        target.generate(
            prepared(manifest, request_body),
            scheduling=scheduling(),
            events=sink,
        )

    assert error.value.code == "provider_protocol_error"
    assert sink.events[-1].kind is SchedulerEventKind.REQUEST_FAILED
    assert sink.events[-1].error_code == "provider_protocol_error"


def test_managed_adapter_capabilities_do_not_claim_sequence_control():
    target = backend(VLLMManagedBackend, FakeTransport(completion()))
    assert target.capabilities.supports_full_request
    assert not target.capabilities.supports_sequence_steps
    assert target.capabilities.max_concurrent_sequences is None
    assert not target.capabilities.supports_cancellation


def test_managed_adapter_cancel_uses_only_explicit_transport_capability(
    manifest,
    request_body,
):
    entered = threading.Event()
    release = threading.Event()

    class BlockingTransport(FakeTransport):
        def post_json(self, path, payload, *, timeout, request_id):
            entered.set()
            release.wait(2)
            return self.response

    transport = BlockingTransport(completion(), cancellable=True)
    target = backend(VLLMManagedBackend, transport)
    result = []
    thread = threading.Thread(
        target=lambda: result.append(
            target.generate(
                prepared(manifest, request_body),
                scheduling=scheduling("cancel-me"),
                events=Sink(),
            )
        )
    )
    thread.start()
    assert entered.wait(1)
    assert target.cancel("cancel-me")
    release.set()
    thread.join(2)
    assert transport.cancelled == ["cancel-me"]
    assert len(result) == 1
