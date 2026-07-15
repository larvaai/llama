from __future__ import annotations

import pytest

from inference_runtime import SchedulerEventKind, SchedulingMetadata
from inference_runtime.adapters import (
    MLXCompletion,
    MLXRuntimeUnavailable,
    ManagedBackendError,
    MlxLMManagedBackend,
    NativeMLXLMRuntime,
)
from model_worker.preflight import preflight


class Sink:
    def __init__(self):
        self.events = []

    def publish(self, event):
        self.events.append(event)


class FakeRuntime:
    def __init__(self, text='{"result":"ok"}'):
        self.text = text
        self.calls = []
        self.shutdown_calls = 0

    def complete(self, messages, *, max_tokens):
        self.calls.append((messages, max_tokens))
        return MLXCompletion(self.text, 20, 5, "mlx-test")

    def shutdown(self):
        self.shutdown_calls += 1
        return True


def scheduling(request_id="mlx-1"):
    return SchedulingMetadata(
        request_id,
        "workflow",
        "agent",
        "throughput",
        1,
        None,
    )


def backend(runtime):
    return MlxLMManagedBackend(
        runtime,
        models=("qwen35-9b-local",),
        max_context_tokens=1024,
        max_output_tokens=128,
        max_concurrent_requests=1,
    )


def test_mlx_managed_backend_revalidates_output_and_is_capability_honest(
    manifest,
    request_body,
):
    runtime = FakeRuntime()
    target = backend(runtime)
    sink = Sink()
    request = preflight(request_body, manifest)

    result = target.generate(
        request,
        scheduling=scheduling(),
        events=sink,
    )

    assert result.output == {"result": "ok"}
    assert result.model["backend"] == "mlx-lm"
    assert result.usage["prompt_tokens"] == 20
    assert runtime.calls[0][1] == min(request.limits.total_tokens, 128)
    assert runtime.calls[0][0][0]["role"] == "system"
    assert [event.kind for event in sink.events] == [
        SchedulerEventKind.ADMITTED,
        SchedulerEventKind.REQUEST_COMPLETED,
    ]
    capabilities = target.capabilities
    assert capabilities.supports_full_request
    assert not capabilities.supports_sequence_steps
    assert not capabilities.supports_streaming
    assert not capabilities.supports_cancellation
    assert capabilities.max_concurrent_sequences is None
    assert target.shutdown()
    assert runtime.shutdown_calls == 1


def test_mlx_managed_backend_fails_closed_on_invalid_structured_output(
    manifest,
    request_body,
):
    target = backend(FakeRuntime('{"wrong":"value"}'))
    sink = Sink()

    with pytest.raises(ManagedBackendError) as captured:
        target.generate(
            preflight(request_body, manifest),
            scheduling=scheduling(),
            events=sink,
        )

    assert captured.value.code == "provider_protocol_error"
    assert sink.events[-1].kind is SchedulerEventKind.REQUEST_FAILED


def test_native_mlx_runtime_uses_public_load_generate_and_chat_template_api():
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt):
            calls.append(("template", messages, add_generation_prompt))
            return "rendered prompt"

        def encode(self, text):
            return list(text.encode())

    model = object()
    tokenizer = Tokenizer()

    def load(model_path, **kwargs):
        calls.append(("load", model_path, kwargs))
        return model, tokenizer

    def generate(loaded_model, loaded_tokenizer, **kwargs):
        assert loaded_model is model and loaded_tokenizer is tokenizer
        calls.append(("generate", kwargs))
        return '{"result":"ok"}'

    runtime = NativeMLXLMRuntime(
        "mlx-community/test-model",
        tokenizer_config={"trust_remote_code": False},
        load_function=load,
        generate_function=generate,
    )
    completion = runtime.complete(
        [{"role": "user", "content": "hello"}],
        max_tokens=32,
    )

    assert completion.text == '{"result":"ok"}'
    assert completion.prompt_tokens == len("rendered prompt".encode())
    assert completion.completion_tokens == len(completion.text.encode())
    assert calls[0] == (
        "load",
        "mlx-community/test-model",
        {"tokenizer_config": {"trust_remote_code": False}},
    )
    assert calls[1][0] == "template"
    assert calls[2][1]["max_tokens"] == 32
    assert calls[2][1]["verbose"] is False


def test_native_mlx_runtime_reports_missing_optional_dependency(monkeypatch):
    def unavailable(name):
        assert name == "mlx_lm"
        raise ImportError(name)

    monkeypatch.setattr("inference_runtime.adapters.mlx_lm.importlib.import_module", unavailable)
    with pytest.raises(MLXRuntimeUnavailable, match="not installed"):
        NativeMLXLMRuntime("model")


@pytest.mark.parametrize(
    "factory",
    [
        lambda: MLXCompletion("ok", -1, 0),
        lambda: MLXCompletion("ok", 0, True),
        lambda: NativeMLXLMRuntime("", load_function=lambda _: None, generate_function=str),
        lambda: NativeMLXLMRuntime("model", load_function=str),
    ],
)
def test_mlx_contracts_fail_closed(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()
