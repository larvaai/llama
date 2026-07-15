from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .openai_compatible import (
    JSONTransport,
    ManagedBackendError,
    ManagedBackendProfile,
    OpenAICompatibleManagedBackend,
)


class MLXRuntimeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MLXCompletion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        if type(self.text) is not str:
            raise TypeError("text must be a string")
        for name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.fingerprint is not None and (
            type(self.fingerprint) is not str or not self.fingerprint
        ):
            raise ValueError("fingerprint must be a non-empty string or None")


@runtime_checkable
class MLXRuntime(Protocol):
    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
    ) -> MLXCompletion: ...

    def shutdown(self) -> bool: ...


class NativeMLXLMRuntime:
    """Lazy local `mlx-lm` runtime with no claimed token-step control.

    The public MLX LM generation API owns its cache and generation loop, so it
    is deliberately exposed as a managed full-request backend. A future
    steppable adapter must use a stable per-sequence API and pass the same
    conformance gates before advertising sequence capabilities.
    """

    def __init__(
        self,
        model_path: str,
        *,
        tokenizer_config: Mapping[str, Any] | None = None,
        load_function=None,
        generate_function=None,
    ) -> None:
        if type(model_path) is not str or not model_path or len(model_path.encode()) > 4096:
            raise ValueError("model_path must be a non-empty bounded string")
        if (load_function is None) != (generate_function is None):
            raise ValueError("load_function and generate_function must be supplied together")
        if tokenizer_config is not None and type(tokenizer_config) is not dict:
            raise TypeError("tokenizer_config must be a dict or None")
        if load_function is None:
            try:
                module = importlib.import_module("mlx_lm")
                load_function = module.load
                generate_function = module.generate
            except (ImportError, AttributeError) as exc:
                raise MLXRuntimeUnavailable(
                    "mlx-lm is not installed with a supported local compute backend"
                ) from exc
        if not callable(load_function) or not callable(generate_function):
            raise TypeError("mlx-lm load and generate functions must be callable")
        load_kwargs = {}
        if tokenizer_config is not None:
            load_kwargs["tokenizer_config"] = dict(tokenizer_config)
        try:
            model, tokenizer = load_function(model_path, **load_kwargs)
        except Exception as exc:
            raise MLXRuntimeUnavailable(
                f"mlx-lm could not load the configured model: {type(exc).__name__}"
            ) from exc
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise MLXRuntimeUnavailable("mlx-lm tokenizer has no chat template")
        if not callable(getattr(tokenizer, "encode", None)):
            raise MLXRuntimeUnavailable("mlx-lm tokenizer has no encode method")
        self.model_path = model_path
        self._model = model
        self._tokenizer = tokenizer
        self._generate = generate_function
        self._lock = threading.Lock()

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
    ) -> MLXCompletion:
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        normalized = []
        for index, message in enumerate(messages):
            if type(message) is not dict or set(message) != {"role", "content"}:
                raise ValueError(f"messages[{index}] must contain role and content")
            role = message["role"]
            content = message["content"]
            if type(role) is not str or not role or type(content) is not str:
                raise ValueError(f"messages[{index}] is invalid")
            normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError("messages must not be empty")
        with self._lock:
            prompt = self._tokenizer.apply_chat_template(
                normalized,
                add_generation_prompt=True,
            )
            text = self._generate(
                self._model,
                self._tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )
        if type(text) is not str:
            raise RuntimeError("mlx-lm generate returned a non-string result")
        prompt_tokens = (
            len(prompt)
            if isinstance(prompt, (list, tuple))
            else len(self._tokenizer.encode(prompt))
        )
        completion_tokens = len(self._tokenizer.encode(text))
        return MLXCompletion(
            text,
            prompt_tokens,
            completion_tokens,
            fingerprint="mlx-lm-local",
        )

    def shutdown(self) -> bool:
        return True


class MLXLocalTransport(JSONTransport):
    """In-process transport that maps MLX completion to the managed boundary."""

    def __init__(self, runtime: MLXRuntime) -> None:
        if not isinstance(runtime, MLXRuntime):
            raise TypeError("runtime must implement MLXRuntime")
        self.runtime = runtime

    @property
    def supports_cancellation(self) -> bool:
        return False

    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        timeout: float,
        request_id: str,
    ) -> Any:
        del timeout
        if path != "/v1/chat/completions":
            raise ManagedBackendError(
                "provider_protocol_error",
                "unsupported MLX endpoint",
                retryable=False,
            )
        try:
            if type(payload) is not dict:
                raise TypeError("payload")
            model = payload["model"]
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            if type(model) is not str or not model:
                raise TypeError("model")
            if type(messages) is not list or not messages:
                raise TypeError("messages")
            if type(max_tokens) is not int or max_tokens <= 0:
                raise TypeError("max_tokens")
            completion = self.runtime.complete(messages, max_tokens=max_tokens)
        except ManagedBackendError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ManagedBackendError(
                "provider_protocol_error",
                f"invalid MLX request: {exc}",
                retryable=False,
            ) from exc
        except Exception as exc:
            raise ManagedBackendError(
                "provider_execution_error",
                type(exc).__name__,
                retryable=True,
            ) from exc
        return {
            "id": f"mlx-{request_id}",
            "model": model,
            "system_fingerprint": completion.fingerprint,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": completion.text},
                }
            ],
            "usage": {
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
            },
        }

    def cancel(self, request_id: str) -> bool:
        del request_id
        return False

    def shutdown(self) -> bool:
        return self.runtime.shutdown()


class MlxLMManagedBackend(OpenAICompatibleManagedBackend):
    def __init__(self, runtime: MLXRuntime, **kwargs: Any) -> None:
        super().__init__(
            ManagedBackendProfile(
                "mlx-lm",
                response_format_json_schema=False,
            ),
            MLXLocalTransport(runtime),
            **kwargs,
        )

    @classmethod
    def from_model(
        cls,
        model_path: str,
        *,
        tokenizer_config: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> MlxLMManagedBackend:
        runtime = NativeMLXLMRuntime(
            model_path,
            tokenizer_config=tokenizer_config,
        )
        return cls(runtime, **kwargs)
