from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from model_worker.contracts import GenerateResult
from model_worker.output_contract import validate_output
from model_worker.preflight import PreflightedRequest
from model_worker.strict_json import loads

from ..contracts import (
    BackendCapabilities,
    SchedulerEvent,
    SchedulerEventKind,
    SchedulingMetadata,
)
from ..ports import SchedulerEventSink


class ManagedBackendError(RuntimeError):
    def __init__(self, code: str, detail: str, *, retryable: bool) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.retryable = retryable


@runtime_checkable
class JSONTransport(Protocol):
    @property
    def supports_cancellation(self) -> bool: ...

    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        *,
        timeout: float,
        request_id: str,
    ) -> Any: ...

    def cancel(self, request_id: str) -> bool: ...


class UrllibJSONTransport:
    """Dependency-free bounded JSON transport for OpenAI-compatible servers.

    urllib cannot safely interrupt an in-flight request, so this transport is
    deliberately honest and never advertises cancellation.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        response_byte_limit: int = 4 * 1024 * 1024,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain query or fragment components")
        if api_key is not None and (type(api_key) is not str or not api_key):
            raise ValueError("api_key must be a non-empty string or None")
        if type(response_byte_limit) is not int or response_byte_limit <= 0:
            raise ValueError("response_byte_limit must be a positive integer")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.response_byte_limit = response_byte_limit

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
        if type(path) is not str or not path.startswith("/"):
            raise ValueError("path must be absolute within the configured server")
        if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        body = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Request-ID": request_id,
        }
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                raw = response.read(self.response_byte_limit + 1)
        except urllib.error.HTTPError as exc:
            raise ManagedBackendError(
                "provider_http_error",
                f"provider returned HTTP {exc.code}",
                retryable=exc.code == 429 or exc.code >= 500,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ManagedBackendError(
                "provider_unavailable",
                type(exc).__name__,
                retryable=True,
            ) from exc
        if len(raw) > self.response_byte_limit:
            raise ManagedBackendError(
                "provider_response_too_large",
                "provider JSON exceeded the configured byte limit",
                retryable=False,
            )
        try:
            return loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ManagedBackendError(
                "provider_protocol_error",
                "provider did not return strict UTF-8 JSON",
                retryable=False,
            ) from exc

    def cancel(self, request_id: str) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class ManagedBackendProfile:
    backend_id: str
    endpoint: str = "/v1/chat/completions"
    response_format_json_schema: bool = True

    def __post_init__(self) -> None:
        if type(self.backend_id) is not str or not self.backend_id:
            raise ValueError("backend_id must be a non-empty string")
        if type(self.endpoint) is not str or not self.endpoint.startswith("/"):
            raise ValueError("endpoint must be an absolute server path")
        if type(self.response_format_json_schema) is not bool:
            raise TypeError("response_format_json_schema must be a bool")


class OpenAICompatibleManagedBackend:
    """Full-request adapter for servers that own their own scheduler and KV.

    It intentionally exposes no sequence handle or token-step API. Provider
    output is treated as untrusted and revalidated against the local contract.
    """

    def __init__(
        self,
        profile: ManagedBackendProfile,
        transport: JSONTransport,
        *,
        models: tuple[str, ...],
        max_context_tokens: int,
        max_output_tokens: int,
        max_concurrent_requests: int,
        request_timeout: float = 120.0,
        clock=time.monotonic,
    ) -> None:
        if type(profile) is not ManagedBackendProfile:
            raise TypeError("profile must be ManagedBackendProfile")
        if not isinstance(transport, JSONTransport):
            raise TypeError("transport must implement JSONTransport")
        if type(models) is not tuple or not models:
            raise ValueError("models must be a non-empty tuple")
        if any(type(model) is not str or not model for model in models):
            raise ValueError("models must contain non-empty strings")
        if len(set(models)) != len(models):
            raise ValueError("models must not contain duplicates")
        for name, value in (
            ("max_context_tokens", max_context_tokens),
            ("max_output_tokens", max_output_tokens),
            ("max_concurrent_requests", max_concurrent_requests),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if max_output_tokens > max_context_tokens:
            raise ValueError("max_output_tokens must not exceed max_context_tokens")
        if (
            type(request_timeout) not in {int, float}
            or not math.isfinite(request_timeout)
            or request_timeout <= 0
        ):
            raise ValueError("request_timeout must be finite and positive")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.profile = profile
        self.transport = transport
        self.request_timeout = float(request_timeout)
        self._clock = clock
        self._capacity = threading.BoundedSemaphore(max_concurrent_requests)
        self._lock = threading.RLock()
        self._active: set[str] = set()
        self._event_failures = 0
        self._capabilities = BackendCapabilities(
            backend=profile.backend_id,
            models=models,
            supports_full_request=True,
            supports_sequence_steps=False,
            supports_streaming=False,
            supports_cancellation=transport.supports_cancellation,
            supports_chunked_prefill=False,
            supports_decode_batching=False,
            supports_continuous_batching=False,
            supports_prefix_cache=False,
            supports_session_cache=False,
            supports_explicit_release=False,
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
            max_concurrent_requests=max_concurrent_requests,
            max_concurrent_sequences=None,
            max_prefill_tokens_per_step=None,
            max_decode_tokens_per_step=None,
            max_sequences_per_step=None,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def event_failures(self) -> int:
        with self._lock:
            return self._event_failures

    def generate(
        self,
        request: PreflightedRequest,
        *,
        scheduling: SchedulingMetadata,
        events: SchedulerEventSink,
    ) -> GenerateResult:
        if type(request) is not PreflightedRequest:
            raise TypeError("managed backend requires PreflightedRequest")
        if type(scheduling) is not SchedulingMetadata:
            raise TypeError("scheduling must be SchedulingMetadata")
        if not isinstance(events, SchedulerEventSink):
            raise TypeError("events must implement SchedulerEventSink")
        if request.request.model_id not in self.capabilities.models:
            raise ManagedBackendError(
                "model_not_supported",
                request.request.model_id,
                retryable=False,
            )
        admitted_at = self._now()
        queue_timeout = request.limits.queue_timeout_ms / 1000
        if scheduling.deadline_monotonic is not None:
            queue_timeout = min(
                queue_timeout,
                max(0.0, scheduling.deadline_monotonic - admitted_at),
            )
        if not self._capacity.acquire(timeout=queue_timeout):
            raise ManagedBackendError(
                "backend_capacity_exhausted",
                "managed backend concurrency limit reached",
                retryable=True,
            )
        started_at = self._now()
        with self._lock:
            if scheduling.request_id in self._active:
                self._capacity.release()
                raise ManagedBackendError(
                    "duplicate_request",
                    scheduling.request_id,
                    retryable=False,
                )
            self._active.add(scheduling.request_id)
        self._publish(events, SchedulerEventKind.ADMITTED, scheduling.request_id)
        try:
            timeout = min(
                self.request_timeout,
                request.limits.execution_timeout_ms / 1000,
            )
            if scheduling.deadline_monotonic is not None:
                timeout = min(
                    timeout,
                    scheduling.deadline_monotonic - self._now(),
                )
            if timeout <= 0:
                raise ManagedBackendError(
                    "deadline_exceeded",
                    "request deadline elapsed before provider dispatch",
                    retryable=False,
                )
            raw = self.transport.post_json(
                self.profile.endpoint,
                self._payload(request),
                timeout=timeout,
                request_id=scheduling.request_id,
            )
            result = self._result(
                raw,
                request=request,
                scheduling=scheduling,
                admitted_at=admitted_at,
                started_at=started_at,
            )
            self._publish(
                events,
                SchedulerEventKind.REQUEST_COMPLETED,
                scheduling.request_id,
            )
            return result
        except ManagedBackendError as exc:
            self._publish(
                events,
                SchedulerEventKind.REQUEST_FAILED,
                scheduling.request_id,
                error_code=exc.code,
            )
            raise
        except Exception as exc:
            error = ManagedBackendError(
                "provider_adapter_failed",
                type(exc).__name__,
                retryable=False,
            )
            self._publish(
                events,
                SchedulerEventKind.REQUEST_FAILED,
                scheduling.request_id,
                error_code=error.code,
            )
            raise error from exc
        finally:
            with self._lock:
                self._active.discard(scheduling.request_id)
            self._capacity.release()

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            active = request_id in self._active
        if not active or not self.capabilities.supports_cancellation:
            return False
        try:
            return bool(self.transport.cancel(request_id))
        except Exception:
            return False

    def shutdown(self) -> bool:
        shutdown = getattr(self.transport, "shutdown", None)
        if callable(shutdown):
            return shutdown() is not False
        return True

    def _payload(self, request: PreflightedRequest) -> dict[str, Any]:
        maximum = min(
            request.limits.total_tokens,
            self.capabilities.max_output_tokens,
        )
        payload: dict[str, Any] = {
            "model": request.request.model_id,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.model_messages
            ],
            "max_tokens": maximum,
            "stream": False,
        }
        if self.profile.response_format_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "model_worker_output",
                    "strict": True,
                    "schema": request.request.output_contract.schema,
                },
            }
        return payload

    def _result(
        self,
        raw: Any,
        *,
        request: PreflightedRequest,
        scheduling: SchedulingMetadata,
        admitted_at: float,
        started_at: float,
    ) -> GenerateResult:
        try:
            if type(raw) is not dict:
                raise TypeError("root")
            choices = raw["choices"]
            if type(choices) is not list or len(choices) != 1:
                raise TypeError("choices")
            choice = choices[0]
            if type(choice) is not dict or type(choice.get("message")) is not dict:
                raise TypeError("message")
            content = choice["message"].get("content")
            if type(content) is not str:
                raise TypeError("content")
            output = loads(content)
            errors = validate_output(output, request.contract)
            if errors:
                raise ValueError("contract")
            usage = raw.get("usage", {})
            if type(usage) is not dict:
                raise TypeError("usage")
            prompt_tokens = self._usage_count(usage, "prompt_tokens")
            completion_tokens = self._usage_count(usage, "completion_tokens")
        except (KeyError, TypeError, ValueError) as exc:
            raise ManagedBackendError(
                "provider_protocol_error",
                f"invalid completion response: {exc}",
                retryable=False,
            ) from exc
        ended_at = self._now()
        identity = raw.get("system_fingerprint")
        model = raw.get("model", request.request.model_id)
        if type(model) is not str or not model:
            model = request.request.model_id
        return GenerateResult(
            scheduling.request_id,
            str(raw.get("id") or f"managed-{scheduling.request_id}"),
            "completed",
            True,
            True,
            output,
            {
                "prompt_tokens": prompt_tokens,
                "reasoning_tokens": 0,
                "final_tokens": completion_tokens,
                "sampled_tokens": completion_tokens,
                "cached_prompt_tokens": 0,
                "cache_hit": False,
                "context_limit": self.capabilities.max_context_tokens,
                "context_headroom": max(
                    0,
                    self.capabilities.max_context_tokens
                    - prompt_tokens
                    - request.limits.total_tokens,
                ),
            },
            {
                "queue_ms": max(0.0, started_at - admitted_at) * 1000,
                "prompt_decode_ms": 0.0,
                "generation_ms": max(0.0, ended_at - started_at) * 1000,
                "total_ms": max(0.0, ended_at - admitted_at) * 1000,
            },
            {
                "id": model,
                "backend": self.profile.backend_id,
                "provider_fingerprint": identity if type(identity) is str else None,
            },
        )

    @staticmethod
    def _usage_count(usage: Mapping[str, Any], name: str) -> int:
        value = usage.get(name, 0)
        if type(value) is not int or value < 0:
            raise TypeError(name)
        return value

    def _publish(
        self,
        sink: SchedulerEventSink,
        kind: SchedulerEventKind,
        request_id: str,
        *,
        error_code: str | None = None,
    ) -> None:
        try:
            sink.publish(
                SchedulerEvent(
                    kind,
                    request_id,
                    self._now(),
                    error_code=error_code,
                )
            )
        except Exception:
            with self._lock:
                self._event_failures += 1

    def _now(self) -> float:
        value = self._clock()
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise RuntimeError("managed backend clock must be finite and non-negative")
        return float(value)


class VLLMManagedBackend(OpenAICompatibleManagedBackend):
    def __init__(self, transport: JSONTransport, **kwargs: Any) -> None:
        super().__init__(ManagedBackendProfile("vllm"), transport, **kwargs)


class SGLangManagedBackend(OpenAICompatibleManagedBackend):
    def __init__(self, transport: JSONTransport, **kwargs: Any) -> None:
        super().__init__(ManagedBackendProfile("sglang"), transport, **kwargs)
