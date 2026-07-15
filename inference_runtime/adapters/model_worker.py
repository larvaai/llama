from __future__ import annotations

import threading
import time
from typing import Any, Callable

from model_worker.dispatcher import Dispatcher
from model_worker.errors import WorkerError
from model_worker.manifest import ModelManifest
from model_worker.preflight import PreflightedRequest
from model_worker.request_registry import Lifecycle, RequestRecord

from ..contracts import (
    BackendCapabilities,
    SchedulerEvent,
    SchedulerEventKind,
    SchedulingMetadata,
)
from ..ports import SchedulerEventSink


class SerialModelWorkerAdapter:
    """Expose the Model Worker v1 dispatcher as an honest managed backend.

    Model Worker v1 owns its FIFO queue and executes a single native request at
    a time.  This adapter deliberately does not emulate sequence stepping.
    M3 replaces it with a native ``SteppableBackend`` implementation.
    """

    def __init__(
        self,
        dispatcher: Dispatcher,
        manifest: ModelManifest,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._dispatcher = dispatcher
        self._clock = clock
        self._active_lock = threading.RLock()
        self._active: dict[str, RequestRecord] = {}
        self._capabilities = BackendCapabilities(
            backend="model-worker-v1-serial",
            models=(manifest.id,),
            supports_full_request=True,
            supports_sequence_steps=False,
            supports_streaming=False,
            supports_cancellation=True,
            supports_chunked_prefill=False,
            supports_decode_batching=False,
            supports_continuous_batching=False,
            supports_prefix_cache=False,
            supports_session_cache=False,
            supports_explicit_release=False,
            max_context_tokens=manifest.context["n_ctx"],
            max_output_tokens=manifest.limits["max_total_tokens"],
            max_concurrent_requests=manifest.limits["max_queue"] + 1,
            max_concurrent_sequences=None,
            max_prefill_tokens_per_step=None,
            max_decode_tokens_per_step=None,
            max_sequences_per_step=None,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def generate(
        self,
        request: PreflightedRequest,
        *,
        scheduling: SchedulingMetadata,
        events: SchedulerEventSink,
    ) -> Any:
        if type(request) is not PreflightedRequest:
            raise TypeError("serial Model Worker adapter requires PreflightedRequest")
        now = self._clock()
        if (
            scheduling.deadline_monotonic is not None
            and scheduling.deadline_monotonic <= now
        ):
            self._publish_failed(events, scheduling.request_id, "deadline_exceeded")
            raise WorkerError("deadline_exceeded", "inference deadline already expired")

        with self._active_lock:
            if scheduling.request_id in self._active:
                raise WorkerError("invalid_request", "duplicate scheduling request_id")

        record: RequestRecord | None = None
        try:
            record = self._dispatcher.submit(request)
            with self._active_lock:
                if scheduling.request_id in self._active:
                    self._dispatcher.cancel(record.request_id)
                    raise WorkerError("invalid_request", "duplicate scheduling request_id")
                self._active[scheduling.request_id] = record
            events.publish(
                SchedulerEvent(
                    kind=SchedulerEventKind.ADMITTED,
                    request_id=scheduling.request_id,
                    at_monotonic=self._clock(),
                )
            )
            self._dispatcher.wait(record)
            if record.lifecycle is Lifecycle.COMPLETED:
                events.publish(
                    SchedulerEvent(
                        kind=SchedulerEventKind.REQUEST_COMPLETED,
                        request_id=scheduling.request_id,
                        at_monotonic=self._clock(),
                    )
                )
                return record.result

            error = self._terminal_error(record)
            self._publish_failed(events, scheduling.request_id, error.code)
            raise error
        except WorkerError as exc:
            if record is None:
                self._publish_failed(events, scheduling.request_id, exc.code)
            raise
        finally:
            with self._active_lock:
                current = self._active.get(scheduling.request_id)
                if current is record:
                    self._active.pop(scheduling.request_id, None)

    def cancel(self, request_id: str) -> bool:
        with self._active_lock:
            record = self._active.get(request_id)
        if record is None:
            return False
        return self._dispatcher.cancel(record.request_id)

    def _publish_failed(
        self,
        events: SchedulerEventSink,
        request_id: str,
        error_code: str,
    ) -> None:
        events.publish(
            SchedulerEvent(
                kind=SchedulerEventKind.REQUEST_FAILED,
                request_id=request_id,
                at_monotonic=self._clock(),
                error_code=error_code,
            )
        )

    @staticmethod
    def _terminal_error(record: RequestRecord) -> WorkerError:
        if isinstance(record.error, WorkerError):
            return record.error
        code = record.error
        if code == "queue_timeout":
            return WorkerError("queue_timeout", "queue deadline exceeded")
        if code == "deadline_exceeded":
            return WorkerError("deadline_exceeded", "execution deadline exceeded")
        if record.lifecycle is Lifecycle.CANCELLED or code == "cancelled":
            return WorkerError("cancelled", "request cancelled")
        return WorkerError("worker_crashed", "managed backend request failed")
