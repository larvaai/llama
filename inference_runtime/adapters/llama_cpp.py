from __future__ import annotations

import hashlib
import json
import queue
import subprocess
import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from model_worker.preflight import PreflightedRequest
from model_worker.strict_json import loads

from ..contracts import (
    BackendCapabilities,
    DecodeOutcome,
    DecodeStatus,
    FinishReason,
    PrefillOutcome,
    PrefillStatus,
    ReleaseOutcome,
    ReleaseStatus,
    SchedulingMetadata,
    SequenceCompletion,
    SequenceHandle,
    SequenceStep,
    validate_sequence_batch,
)
from ..ports import SchedulerEventSink
from ..runtime_manifest import InferenceRuntimeManifest, load_inference_runtime_manifest


RUNTIME_IPC_VERSION = "inference-runtime-ipc.v1"


class BackendCommandError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class LlamaCppSteppableBackend:
    """Process adapter for the persistent multi-sequence llama.cpp runtime."""

    def __init__(
        self,
        executable: Path,
        runtime_manifest_path: Path,
        *,
        verify_model_files: bool = True,
        startup_timeout: float = 60.0,
        command_timeout: float = 60.0,
        shutdown_timeout: float = 5.0,
        released_handle_capacity: int = 4096,
    ) -> None:
        if min(startup_timeout, command_timeout, shutdown_timeout) <= 0:
            raise ValueError("backend timeouts must be positive")
        if type(released_handle_capacity) is not int or released_handle_capacity <= 0:
            raise ValueError("released_handle_capacity must be a positive integer")
        self.executable = executable.resolve()
        if not self.executable.is_file():
            raise FileNotFoundError(self.executable)
        self.runtime_manifest: InferenceRuntimeManifest = load_inference_runtime_manifest(
            runtime_manifest_path,
            verify_model_files=verify_model_files,
        )
        self.startup_timeout = startup_timeout
        self.command_timeout = command_timeout
        self.shutdown_timeout = shutdown_timeout
        self.released_handle_capacity = released_handle_capacity
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._command_id = 0
        self._process_generation = 0
        self._state = "STOPPED"
        self._active: dict[SequenceHandle, dict[str, Any]] = {}
        self._reserved_tokens: dict[SequenceHandle, int] = {}
        self._released: OrderedDict[SequenceHandle, dict[str, Any]] = OrderedDict()
        self._executable_digest = (
            "sha256:" + hashlib.sha256(self.executable.read_bytes()).hexdigest()
        )
        scheduler = self.runtime_manifest.scheduler
        model = self.runtime_manifest.model_manifest
        sequence_context = min(
            model.context["n_ctx"],
            scheduler.kv_tokens // scheduler.max_sequences,
        )
        self._capabilities = BackendCapabilities(
            backend=self.runtime_manifest.backend_id,
            models=(model.id,),
            supports_full_request=False,
            supports_sequence_steps=True,
            supports_streaming=True,
            supports_cancellation=True,
            supports_chunked_prefill=True,
            supports_decode_batching=True,
            supports_continuous_batching=True,
            supports_prefix_cache=self.runtime_manifest.cache.enabled,
            supports_session_cache=self.runtime_manifest.cache.enabled,
            supports_explicit_release=True,
            max_context_tokens=sequence_context,
            max_output_tokens=min(
                model.limits["max_total_tokens"],
                sequence_context,
            ),
            max_concurrent_requests=scheduler.max_sequences,
            max_concurrent_sequences=scheduler.max_sequences,
            max_prefill_tokens_per_step=scheduler.prefill_chunk_tokens,
            max_decode_tokens_per_step=scheduler.decode_quantum_tokens,
            max_sequences_per_step=scheduler.max_decode_batch,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def supervisor_state(self) -> str:
        with self._lock:
            if (
                self._state == "READY"
                and (self._process is None or self._process.poll() is not None)
            ):
                self._state = "DEGRADED"
            return self._state

    @property
    def process_generation(self) -> int:
        with self._lock:
            return self._process_generation

    @property
    def process_id(self) -> int | None:
        """Return the live native worker PID for bounded observability only."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return None
            return self._process.pid

    @property
    def runtime_identity(self) -> dict[str, Any]:
        return {
            "backend_id": self.runtime_manifest.backend_id,
            "runtime_manifest_digest": self.runtime_manifest.digest,
            "model_manifest_digest": self.runtime_manifest.model_manifest.digest,
            "native_executable_sha256": self._executable_digest,
            "process_generation": self.process_generation,
        }

    def start(self) -> None:
        with self._lock:
            if (
                self._process is not None
                and self._process.poll() is None
                and self._state == "READY"
            ):
                return
            self._state = "STARTING"
            process: subprocess.Popen[str] | None = None
            try:
                process = subprocess.Popen(
                    [str(self.executable), str(self.runtime_manifest.path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                )
                ready_line = self._readline_with_timeout(process, self.startup_timeout)
                ready = loads(ready_line)
                self._verify_ready(ready)
                self._process = process
                self._process_generation += 1
                self._command_id = 0
                self._active.clear()
                self._reserved_tokens.clear()
                self._released.clear()
                self._state = "READY"
            except BaseException:
                if process is not None:
                    self._terminate_process(process)
                self._process = None
                self._state = "DEGRADED"
                raise

    def _verify_ready(self, ready: Any) -> None:
        expected = {
            "protocol_version",
            "type",
            "sequence",
            "backend_id",
            "model_id",
            "model_manifest_digest",
            "max_sequences",
            "cpu_threads",
            "kv_tokens",
            "sequence_context_tokens",
            "prefill_chunk_tokens",
            "max_decode_batch",
            "decode_quantum_tokens",
            "tick_token_budget",
            "cache",
        }
        scheduler = self.runtime_manifest.scheduler
        if (
            type(ready) is not dict
            or set(ready) != expected
            or ready.get("protocol_version") != RUNTIME_IPC_VERSION
            or ready.get("type") != "ready"
            or ready.get("sequence") != 0
            or ready.get("backend_id") != self.runtime_manifest.backend_id
            or ready.get("model_id") != self.runtime_manifest.model_manifest.id
            or ready.get("model_manifest_digest")
            != self.runtime_manifest.model_manifest.digest
            or ready.get("max_sequences") != scheduler.max_sequences
            or ready.get("cpu_threads") != scheduler.cpu_threads
            or ready.get("kv_tokens") != scheduler.kv_tokens
            or ready.get("sequence_context_tokens")
            != self._capabilities.max_context_tokens
            or ready.get("prefill_chunk_tokens") != scheduler.prefill_chunk_tokens
            or ready.get("max_decode_batch") != scheduler.max_decode_batch
            or ready.get("decode_quantum_tokens") != scheduler.decode_quantum_tokens
            or ready.get("tick_token_budget") != scheduler.tick_token_budget
            or ready.get("cache")
            != {
                "enabled": self.runtime_manifest.cache.enabled,
                "byte_budget": self.runtime_manifest.cache.byte_budget,
                "max_entries": self.runtime_manifest.cache.max_entries,
                "ttl_seconds": self.runtime_manifest.cache.ttl_seconds,
            }
        ):
            raise BackendCommandError("worker_not_ready", "invalid runtime ready frame")

    @staticmethod
    def _readline_with_timeout(
        process: subprocess.Popen[str],
        timeout: float,
    ) -> str:
        if process.stdout is None:
            raise BackendCommandError("worker_crashed", "runtime stdout is unavailable")
        result: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

        def read() -> None:
            try:
                result.put(process.stdout.readline())
            except BaseException as exc:
                result.put(exc)

        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        try:
            value = result.get(timeout=timeout)
        except queue.Empty as exc:
            raise BackendCommandError("worker_timeout", "runtime response timed out") from exc
        if isinstance(value, BaseException):
            raise BackendCommandError("worker_crashed", "runtime stdout failed") from value
        if not value:
            raise BackendCommandError("worker_crashed", "runtime exited before response")
        return value

    def _command(self, command_type: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self.start()
            process = self._process
            assert process is not None and process.stdin is not None
            self._command_id += 1
            command_id = self._command_id
            command = {
                "protocol_version": RUNTIME_IPC_VERSION,
                "type": command_type,
                "command_id": command_id,
                **payload,
            }
            try:
                process.stdin.write(
                    json.dumps(
                        command,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                process.stdin.flush()
                response = loads(
                    self._readline_with_timeout(process, self.command_timeout)
                )
                if (
                    type(response) is not dict
                    or response.get("protocol_version") != RUNTIME_IPC_VERSION
                    or response.get("command_id") != command_id
                    or type(response.get("type")) is not str
                ):
                    raise BackendCommandError(
                        "protocol_violation",
                        "invalid or uncorrelated runtime response",
                    )
                if response["type"] == "command_error":
                    if set(response) != {
                        "protocol_version",
                        "type",
                        "command_id",
                        "error_code",
                        "detail",
                    }:
                        raise BackendCommandError(
                            "protocol_violation",
                            "malformed runtime error response",
                        )
                    raise BackendCommandError(
                        str(response["error_code"]),
                        str(response["detail"]),
                    )
                return response
            except BackendCommandError as exc:
                if exc.code in {
                    "worker_crashed",
                    "worker_timeout",
                    "protocol_violation",
                }:
                    self._fail_process_locked(process)
                raise
            except (BrokenPipeError, OSError, ValueError) as exc:
                self._fail_process_locked(process)
                raise BackendCommandError(
                    "worker_crashed",
                    "runtime command transport failed",
                ) from exc

    def _fail_process_locked(self, process: subprocess.Popen[str]) -> None:
        self._terminate_process(process)
        if self._process is process:
            self._process = None
        self._active.clear()
        self._reserved_tokens.clear()
        self._released.clear()
        self._state = "DEGRADED"

    def _terminate_process(self, process: subprocess.Popen[str]) -> bool:
        if process.poll() is not None:
            return True
        process.terminate()
        try:
            process.wait(self.shutdown_timeout)
            return True
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(self.shutdown_timeout)
                return True
            except subprocess.TimeoutExpired:
                return False

    def open_sequence(
        self,
        request: PreflightedRequest,
        *,
        scheduling: SchedulingMetadata,
        events: SchedulerEventSink,
    ) -> SequenceHandle:
        del events
        if type(request) is not PreflightedRequest:
            raise TypeError("llama.cpp sequence backend requires PreflightedRequest")
        if type(scheduling) is not SchedulingMetadata:
            raise TypeError("scheduling must be validated SchedulingMetadata")
        envelope = {
            "request": asdict(request.request),
            "grammar": request.grammar,
            "model_messages": [asdict(message) for message in request.model_messages],
            "prompt_hash": request.prompt_hash,
            "prompt_version": request.prompt_version,
            "cache_scope": (
                asdict(scheduling.cache_scope)
                if scheduling.cache_scope is not None
                else None
            ),
            "session_cache": (
                asdict(scheduling.session_cache)
                if scheduling.session_cache is not None
                else None
            ),
            "cache_namespace": {
                "model_digest": self.runtime_manifest.model_manifest.digest,
                "template_digest": "sha256:"
                + hashlib.sha256(request.prompt_version.encode("utf-8")).hexdigest(),
                "tokenizer_digest": self.runtime_manifest.model_manifest.raw[
                    "gguf_sha256"
                ],
                "adapter_digest": self._executable_digest,
                "context_digest": self.runtime_manifest.digest,
            },
        }
        response = self._command(
            "open_sequence",
            request_id=scheduling.request_id,
            attempt_id=uuid.uuid4().hex,
            request=envelope,
        )
        if set(response) != {
            "protocol_version",
            "type",
            "command_id",
            "handle",
            "prompt_tokens",
            "reserved_tokens",
            "cache_hit",
            "cached_prompt_tokens",
        } or response["type"] != "sequence_opened":
            self._protocol_failure("invalid sequence_opened response")
        native = self._parse_native_handle(response["handle"])
        with self._lock:
            generation = (self._process_generation << 32) | native.generation
            public = SequenceHandle(
                native.backend,
                native.model,
                native.sequence,
                generation,
            )
            if public in self._active:
                self._protocol_failure("runtime reused an active sequence handle")
            self._active[public] = response["handle"]
            self._reserved_tokens[public] = int(response["reserved_tokens"])
            return public

    def reservation_tokens(self, handle: SequenceHandle) -> int:
        with self._lock:
            try:
                return self._reserved_tokens[handle]
            except KeyError as exc:
                raise BackendCommandError("stale_handle", "sequence is not active") from exc

    def cache_stats(self) -> dict[str, int | bool]:
        response = self._command("cache_stats")
        if response.get("type") != "cache_stats" or set(response) != {
            "protocol_version",
            "type",
            "command_id",
            "cache",
        }:
            self._protocol_failure("invalid cache_stats response")
        cache = response["cache"]
        expected = {
            "enabled",
            "entries",
            "bytes_used",
            "byte_budget",
            "hits",
            "exact_hits",
            "prefix_hits",
            "session_hits",
            "misses",
            "session_misses",
            "insertions",
            "session_insertions",
            "session_entries",
            "cow_clones",
            "evictions",
            "restore_failures",
            "store_failures",
            "saved_prefill_tokens",
        }
        if type(cache) is not dict or set(cache) != expected:
            self._protocol_failure("malformed cache_stats response")
        if type(cache["enabled"]) is not bool or any(
            type(cache[name]) is not int or cache[name] < 0
            for name in expected - {"enabled"}
        ):
            self._protocol_failure("invalid cache_stats values")
        return dict(cache)

    def clear_cache(self) -> int:
        response = self._command("cache_clear")
        if response.get("type") != "cache_cleared" or set(response) != {
            "protocol_version",
            "type",
            "command_id",
            "removed_entries",
        }:
            self._protocol_failure("invalid cache_cleared response")
        removed = response["removed_entries"]
        if type(removed) is not int or removed < 0:
            self._protocol_failure("invalid cache clear count")
        return removed

    def _parse_native_handle(self, value: Any) -> SequenceHandle:
        if type(value) is not dict or set(value) != {
            "backend",
            "model",
            "sequence",
            "generation",
        }:
            self._protocol_failure("invalid native sequence handle")
        try:
            handle = SequenceHandle(**value)
        except (TypeError, ValueError) as exc:
            self._protocol_failure("invalid native sequence handle", cause=exc)
        if (
            handle.backend != self.runtime_manifest.backend_id
            or handle.model != self.runtime_manifest.model_manifest.id
        ):
            self._protocol_failure("native sequence identity mismatch")
        return handle

    def _protocol_failure(
        self,
        detail: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        with self._lock:
            process = self._process
            if process is not None:
                self._fail_process_locked(process)
        error = BackendCommandError("protocol_violation", detail)
        if cause is not None:
            raise error from cause
        raise error

    def _native_steps(self, steps: tuple[SequenceStep, ...]) -> list[dict[str, Any]]:
        result = []
        with self._lock:
            for step in steps:
                native = self._active.get(step.handle)
                if native is None:
                    raise BackendCommandError("stale_handle", "sequence is not active")
                result.append({"handle": native, "token_budget": step.token_budget})
        return result

    def prefill(
        self,
        handle: SequenceHandle,
        *,
        token_budget: int,
        events: SchedulerEventSink,
    ) -> PrefillOutcome:
        return self.prefill_batch(
            (SequenceStep(handle, token_budget),),
            events=events,
        )[0]

    def prefill_batch(
        self,
        steps: tuple[SequenceStep, ...],
        *,
        events: SchedulerEventSink,
    ) -> tuple[PrefillOutcome, ...]:
        del events
        capabilities = self.capabilities
        assert capabilities.max_sequences_per_step is not None
        assert capabilities.max_prefill_tokens_per_step is not None
        validate_sequence_batch(
            steps,
            max_sequences=capabilities.max_sequences_per_step,
            max_tokens_per_step=capabilities.max_prefill_tokens_per_step,
        )
        if sum(step.token_budget for step in steps) > self.runtime_manifest.scheduler.tick_token_budget:
            raise ValueError("prefill batch exceeds tick token budget")
        response = self._command("prefill_batch", steps=self._native_steps(steps))
        if response.get("type") != "prefill_completed" or set(response) != {
            "protocol_version",
            "type",
            "command_id",
            "outcomes",
        }:
            self._protocol_failure("invalid prefill batch response")
        outcomes = response["outcomes"]
        if type(outcomes) is not list or len(outcomes) != len(steps):
            self._protocol_failure("prefill outcome cardinality mismatch")
        parsed = []
        for step, outcome in zip(steps, outcomes, strict=True):
            if type(outcome) is not dict or set(outcome) != {
                "handle",
                "status",
                "processed_tokens",
                "remaining_tokens",
            }:
                self._protocol_failure("malformed prefill outcome")
            self._verify_native_correlation(step.handle, outcome["handle"])
            try:
                parsed.append(
                    PrefillOutcome(
                        step.handle,
                        PrefillStatus(outcome["status"]),
                        outcome["processed_tokens"],
                        outcome["remaining_tokens"],
                    )
                )
            except (TypeError, ValueError) as exc:
                self._protocol_failure("invalid prefill outcome", cause=exc)
        return tuple(parsed)

    def decode(
        self,
        handle: SequenceHandle,
        *,
        token_budget: int,
        events: SchedulerEventSink,
    ) -> DecodeOutcome:
        return self.decode_batch(
            (SequenceStep(handle, token_budget),),
            events=events,
        )[0]

    def decode_batch(
        self,
        steps: tuple[SequenceStep, ...],
        *,
        events: SchedulerEventSink,
    ) -> tuple[DecodeOutcome, ...]:
        del events
        capabilities = self.capabilities
        assert capabilities.max_sequences_per_step is not None
        assert capabilities.max_decode_tokens_per_step is not None
        validate_sequence_batch(
            steps,
            max_sequences=capabilities.max_sequences_per_step,
            max_tokens_per_step=capabilities.max_decode_tokens_per_step,
        )
        if sum(step.token_budget for step in steps) > self.runtime_manifest.scheduler.tick_token_budget:
            raise ValueError("decode batch exceeds tick token budget")
        response = self._command("decode_batch", steps=self._native_steps(steps))
        if response.get("type") != "decode_completed" or set(response) != {
            "protocol_version",
            "type",
            "command_id",
            "outcomes",
        }:
            self._protocol_failure("invalid decode batch response")
        outcomes = response["outcomes"]
        if type(outcomes) is not list or len(outcomes) != len(steps):
            self._protocol_failure("decode outcome cardinality mismatch")
        return tuple(
            self._decode_outcome(step.handle, outcome)
            for step, outcome in zip(steps, outcomes, strict=True)
        )

    def _decode_outcome(
        self,
        public: SequenceHandle,
        outcome: Any,
    ) -> DecodeOutcome:
        if type(outcome) is not dict:
            self._protocol_failure("malformed decode outcome")
        common = {"handle", "status", "token_ids", "text_delta"}
        status = outcome.get("status")
        expected = {
            "progressed": common,
            "finished": common | {"finish_reason", "completion"},
            "failed": common | {"error_code"},
        }.get(status)
        if expected is None or set(outcome) != expected:
            self._protocol_failure("malformed decode outcome")
        self._verify_native_correlation(public, outcome["handle"])
        try:
            token_ids = tuple(outcome["token_ids"])
            if status == "progressed":
                return DecodeOutcome(
                    public,
                    DecodeStatus.PROGRESSED,
                    token_ids,
                    outcome["text_delta"],
                )
            if status == "failed":
                return DecodeOutcome(
                    public,
                    DecodeStatus.FAILED,
                    token_ids,
                    outcome["text_delta"],
                    error_code=outcome["error_code"],
                )
            completion_payload = dict(outcome["completion"])
            for name in ("sample_itl_ms", "final_itl_ms"):
                if name in completion_payload:
                    completion_payload[name] = tuple(completion_payload[name])
            completion = SequenceCompletion(**completion_payload)
            return DecodeOutcome(
                public,
                DecodeStatus.FINISHED,
                token_ids,
                outcome["text_delta"],
                FinishReason(outcome["finish_reason"]),
                completion,
            )
        except (TypeError, ValueError, KeyError) as exc:
            self._protocol_failure("invalid decode outcome", cause=exc)

    def _verify_native_correlation(
        self,
        public: SequenceHandle,
        native: Any,
    ) -> None:
        with self._lock:
            expected = self._active.get(public)
        if native != expected:
            self._protocol_failure("native outcome handle mismatch")

    def release(
        self,
        handle: SequenceHandle,
        *,
        events: SchedulerEventSink,
    ) -> ReleaseOutcome:
        del events
        with self._lock:
            native = self._active.get(handle)
            if native is None:
                if handle in self._released:
                    return ReleaseOutcome(handle, ReleaseStatus.ALREADY_RELEASED, 0)
                return ReleaseOutcome(handle, ReleaseStatus.STALE_HANDLE, 0)
        response = self._command("release_sequence", handle=native)
        if response.get("type") != "sequence_released" or set(response) != {
            "protocol_version",
            "type",
            "command_id",
            "handle",
            "status",
            "released_bytes",
        }:
            self._protocol_failure("invalid release response")
        self._verify_native_correlation(handle, response["handle"])
        try:
            status = ReleaseStatus(response["status"])
            outcome = ReleaseOutcome(handle, status, response["released_bytes"])
        except (TypeError, ValueError) as exc:
            self._protocol_failure("invalid release outcome", cause=exc)
        if status in {
            ReleaseStatus.RELEASED,
            ReleaseStatus.ALREADY_RELEASED,
            ReleaseStatus.STALE_HANDLE,
        }:
            with self._lock:
                removed = self._active.pop(handle, None)
                self._reserved_tokens.pop(handle, None)
                if status in {ReleaseStatus.RELEASED, ReleaseStatus.ALREADY_RELEASED}:
                    self._released[handle] = removed or native
                    self._released.move_to_end(handle)
                    while len(self._released) > self.released_handle_capacity:
                        self._released.popitem(last=False)
        return outcome

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                self._state = "STOPPED"
                self._active.clear()
                self._reserved_tokens.clear()
                self._released.clear()
                return
            self._state = "DRAINING"
            try:
                self._command_id += 1
                command_id = self._command_id
                if process.stdin is not None:
                    process.stdin.write(
                        json.dumps(
                            {
                                "protocol_version": RUNTIME_IPC_VERSION,
                                "type": "shutdown",
                                "command_id": command_id,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    process.stdin.flush()
                    response = loads(
                        self._readline_with_timeout(process, self.shutdown_timeout)
                    )
                    if (
                        response.get("type") != "shutdown_complete"
                        or response.get("command_id") != command_id
                    ):
                        raise BackendCommandError(
                            "protocol_violation",
                            "invalid shutdown response",
                        )
            except (BackendCommandError, BrokenPipeError, OSError):
                pass
            stopped = self._terminate_process(process)
            self._process = None
            self._active.clear()
            self._reserved_tokens.clear()
            self._released.clear()
            self._state = "STOPPED" if stopped else "DEGRADED"
