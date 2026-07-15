from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Protocol

from .errors import WorkerError
from .events import BoundedRequestEventBuffer
from .request_registry import Lifecycle, RequestRecord, RequestRegistry, TERMINAL


class WorkerBackend(Protocol):
    def execute(self, record: RequestRecord) -> Any: ...
    def cancel(self, record: RequestRecord) -> None: ...
    def kill_and_restart(self) -> bool: ...
    def shutdown(self) -> None: ...


class Dispatcher:
    def __init__(
        self,
        worker: WorkerBackend,
        *,
        capacity: int,
        watchdog_grace_ms: int = 500,
        event_buffer_max_events: int = 1024,
        event_buffer_max_bytes: int = 1024 * 1024,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if type(event_buffer_max_events) is not int or event_buffer_max_events <= 0:
            raise ValueError("event_buffer_max_events must be a positive integer")
        if type(event_buffer_max_bytes) is not int or event_buffer_max_bytes <= 0:
            raise ValueError("event_buffer_max_bytes must be a positive integer")
        self.registry = RequestRegistry()
        self.worker = worker
        self.watchdog_grace = watchdog_grace_ms / 1000
        self._event_buffer_max_events = event_buffer_max_events
        self._event_buffer_max_bytes = event_buffer_max_bytes

        # Only live queued records are stored. Removal on cancellation/expiry
        # keeps the physical structure bounded while preserving FIFO order.
        self._queue_capacity = capacity
        self._queue_condition = threading.Condition()
        self._pending: deque[RequestRecord] = deque()
        self._dispatch_stopping = False
        self._expiry_stopping = False

        self._accepting = True
        self._admission_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._shutdown_call_lock = threading.Lock()
        self._shutdown_started = False
        self._shutdown_complete = False
        self._backend_shutdown_thread: threading.Thread | None = None
        self._restart_thread: threading.Thread | None = None
        self._restart_succeeded: bool | None = None
        self._backend_lifecycle_lock = threading.Lock()
        self._request_threads_lock = threading.Lock()
        self._request_threads: set[threading.Thread] = set()

        self._expiry_thread = threading.Thread(
            target=self._expire_queued,
            name="model-worker-queue-expiry",
            daemon=True,
        )
        self._thread = threading.Thread(
            target=self._run,
            name="model-worker-dispatcher",
            daemon=True,
        )
        self._expiry_thread.start()
        self._thread.start()

    @property
    def supervisor_state(self) -> str:
        state = getattr(self.worker, "supervisor_state", None)
        if state is None:
            return "READY"
        return state.value if hasattr(state, "value") else str(state)

    @property
    def process_generation(self) -> int | None:
        value = getattr(self.worker, "process_generation", None)
        return value if type(value) is int else None

    @property
    def runtime_identity(self) -> dict[str, Any]:
        value = getattr(self.worker, "runtime_identity", {})
        return dict(value) if isinstance(value, dict) else {}

    @property
    def queued_count(self) -> int:
        with self._queue_condition:
            return len(self._pending)

    @property
    def shutdown_complete(self) -> bool:
        with self._state_lock:
            return self._shutdown_complete

    def submit(self, request: Any) -> RequestRecord:
        with self._admission_lock:
            if not self._accepting:
                raise WorkerError("shutdown", "service is draining")
            # Queue-full is an admission rejection, not a lifecycle terminal.
            # Check it before creating a registry record so rejected requests
            # are accounted once by the API boundary rather than once here and
            # again by the terminal observer.
            with self._queue_condition:
                if len(self._pending) >= self._queue_capacity:
                    raise WorkerError("queue_full", "dispatcher queue is full")
                record = self.registry.create(
                    request,
                    request.limits.queue_timeout_ms,
                    request.limits.execution_timeout_ms,
                )
                if request.request.stream.enabled:
                    record.event_sink = BoundedRequestEventBuffer(
                        record.request_id,
                        record.attempt_id,
                        max_events=self._event_buffer_max_events,
                        max_bytes=self._event_buffer_max_bytes,
                    )
                self.registry.transition(record, Lifecycle.PREFLIGHTED)
                self.registry.transition(record, Lifecycle.QUEUED)
                self._pending.append(record)
                self._queue_condition.notify_all()
            return record

    def cancel(self, request_id: str) -> bool:
        record = self.registry.get(request_id)
        if record is None:
            return False
        with self._queue_condition:
            changed = self.registry.cancel(request_id)
            cancelled_while_queued = record.lifecycle == Lifecycle.CANCELLED
            running = record.lifecycle == Lifecycle.RUNNING
            if cancelled_while_queued and self._remove_pending_locked(record):
                self._queue_condition.notify_all()
        if not changed:
            return changed
        if cancelled_while_queued:
            self._close_event_buffer(record)
        if running:
            self.worker.cancel(record)
        return True

    def wait(self, record: RequestRecord, timeout: float | None = None) -> RequestRecord:
        deadline = None if timeout is None else time.monotonic() + timeout
        with record.condition:
            while record.lifecycle not in TERMINAL:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                record.condition.wait(remaining)
        return record

    def _remove_pending(self, record: RequestRecord) -> bool:
        with self._queue_condition:
            removed = self._remove_pending_locked(record)
            if removed:
                self._queue_condition.notify_all()
            return removed

    def _remove_pending_locked(self, record: RequestRecord) -> bool:
        for index, candidate in enumerate(self._pending):
            if candidate is record:
                del self._pending[index]
                return True
        return False

    @staticmethod
    def _close_event_buffer(record: RequestRecord) -> None:
        if isinstance(record.event_sink, BoundedRequestEventBuffer):
            record.event_sink.close()

    def _next_pending(self) -> RequestRecord | None:
        with self._queue_condition:
            while not self._pending and not self._dispatch_stopping:
                self._queue_condition.wait()
            if self._dispatch_stopping:
                return None
            return self._pending.popleft()

    def _expire_queued(self) -> None:
        while True:
            with self._queue_condition:
                while not self._pending and not self._expiry_stopping:
                    self._queue_condition.wait()
                if self._expiry_stopping:
                    return
                now = time.monotonic()
                earliest = min(record.queue_deadline for record in self._pending)
                if earliest > now:
                    self._queue_condition.wait(earliest - now)
                    continue
                expired = tuple(
                    record for record in self._pending if record.queue_deadline <= now
                )
                for record in expired:
                    self.registry.compare_and_transition(
                        record,
                        Lifecycle.QUEUED,
                        Lifecycle.TIMED_OUT,
                        error="queue_timeout",
                    )
                    # Removal happens before releasing the queue lock. A waiter
                    # notified by the terminal transition therefore cannot race
                    # a new submit into a transient queue_full rejection.
                    self._remove_pending_locked(record)
                self._queue_condition.notify_all()
            for record in expired:
                if record.lifecycle in TERMINAL:
                    self._close_event_buffer(record)

    def _start_tracked_thread(
        self,
        target: Callable[[], None],
        *,
        name: str,
    ) -> threading.Thread:
        def invoke() -> None:
            target()

        thread = threading.Thread(target=invoke, name=name, daemon=True)
        with self._request_threads_lock:
            self._request_threads = {
                candidate for candidate in self._request_threads if candidate.is_alive()
            }
            self._request_threads.add(thread)
            # Start while holding the tracking lock so shutdown cannot observe
            # and discard a not-yet-started thread that will become live later.
            thread.start()
        return thread

    def _schedule_restart(self) -> None:
        with self._state_lock:
            if self._shutdown_started:
                return
            if self._restart_thread is not None and self._restart_thread.is_alive():
                return
            self._restart_succeeded = None

            def restart() -> None:
                succeeded = False
                try:
                    with self._backend_lifecycle_lock:
                        succeeded = bool(self.worker.kill_and_restart())
                except BaseException:
                    succeeded = False
                finally:
                    with self._state_lock:
                        self._restart_succeeded = succeeded
                    with self._queue_condition:
                        self._queue_condition.notify_all()

            thread = threading.Thread(
                target=restart,
                name="model-worker-backend-restart",
                daemon=True,
            )
            self._restart_thread = thread
            thread.start()

    def _wait_for_restart(self) -> bool:
        while True:
            with self._queue_condition:
                if self._dispatch_stopping:
                    return False
            with self._state_lock:
                restart = self._restart_thread
            if restart is None or not restart.is_alive():
                return True
            restart.join(.05)

    def _run(self) -> None:
        while self._wait_for_restart():
            record = self._next_pending()
            if record is None:
                return
            started = self.registry.compare_and_transition(
                record,
                Lifecycle.QUEUED,
                Lifecycle.RUNNING,
                predicate=lambda: time.monotonic() < record.queue_deadline,
            )
            if not started:
                self.registry.compare_and_transition(
                    record,
                    Lifecycle.QUEUED,
                    Lifecycle.TIMED_OUT,
                    error="queue_timeout",
                )
                self._close_event_buffer(record)
                continue

            outcome: dict[str, Any] = {}

            def execute() -> None:
                try:
                    outcome["result"] = self.worker.execute(record)
                except BaseException as exc:  # contained at process boundary
                    outcome["error"] = exc

            task = self._start_tracked_thread(
                execute,
                name=f"request-{record.request_id}",
            )
            execution_deadline = (
                record.timestamps[Lifecycle.RUNNING.value] + record.execution_timeout
            )
            while task.is_alive():
                remaining = execution_deadline - time.monotonic()
                if remaining <= 0:
                    break
                task.join(min(.02, remaining))
                if record.lifecycle in TERMINAL:
                    break
            if record.lifecycle in TERMINAL:
                self._close_event_buffer(record)
                continue
            if task.is_alive():
                record.cancel_event.set()
                cancel_task = self._start_tracked_thread(
                    lambda: self.worker.cancel(record),
                    name=f"request-cancel-{record.request_id}",
                )
                grace_deadline = execution_deadline + self.watchdog_grace
                task.join(max(0, grace_deadline - time.monotonic()))
                restart_required = task.is_alive() or cancel_task.is_alive()
                self.registry.compare_and_transition(
                    record,
                    Lifecycle.RUNNING,
                    Lifecycle.TIMED_OUT,
                    error="deadline_exceeded",
                )
                if restart_required:
                    self._schedule_restart()
            elif (
                "error" in outcome
                and isinstance(outcome["error"], WorkerError)
                and outcome["error"].code == "slow_consumer"
            ):
                self.registry.compare_and_transition(
                    record,
                    Lifecycle.RUNNING,
                    Lifecycle.FAILED,
                    error=outcome["error"],
                )
            elif record.cancel_event.is_set():
                self.registry.compare_and_transition(
                    record,
                    Lifecycle.RUNNING,
                    Lifecycle.CANCELLED,
                    error="cancelled",
                )
            elif "error" in outcome:
                error = outcome["error"]
                stored = error if isinstance(error, WorkerError) else "worker_crashed"
                self.registry.compare_and_transition(
                    record,
                    Lifecycle.RUNNING,
                    Lifecycle.FAILED,
                    error=stored,
                )
            else:
                self.registry.compare_and_transition(
                    record,
                    Lifecycle.RUNNING,
                    Lifecycle.COMPLETED,
                    result=outcome.get("result"),
                )
            self._close_event_buffer(record)

    def _begin_shutdown(self) -> None:
        with self._admission_lock:
            self._accepting = False
        with self._state_lock:
            self._shutdown_started = True
        with self._queue_condition:
            self._dispatch_stopping = True
            self._expiry_stopping = True
            self._queue_condition.notify_all()

        for record in self.registry.snapshot():
            if record.lifecycle in TERMINAL:
                continue
            changed = self.registry.cancel(record.request_id)
            if not changed:
                continue
            if record.lifecycle == Lifecycle.CANCELLED:
                self._remove_pending(record)
            elif self.registry.compare_and_transition(
                record,
                Lifecycle.RUNNING,
                Lifecycle.CANCELLED,
                error="shutdown",
            ):
                self._start_tracked_thread(
                    lambda current=record: self.worker.cancel(current),
                    name=f"shutdown-cancel-{record.request_id}",
                )
            if record.lifecycle in TERMINAL:
                self._close_event_buffer(record)

        def stop_backend() -> None:
            with self._backend_lifecycle_lock:
                self.worker.shutdown()

        backend_shutdown = threading.Thread(
            target=stop_backend,
            name="model-worker-backend-shutdown",
            daemon=True,
        )
        with self._state_lock:
            self._backend_shutdown_thread = backend_shutdown
            backend_shutdown.start()

    def _live_shutdown_threads(self) -> tuple[threading.Thread, ...]:
        current = threading.current_thread()
        with self._state_lock:
            candidates = [
                self._thread,
                self._expiry_thread,
                self._restart_thread,
                self._backend_shutdown_thread,
            ]
        with self._request_threads_lock:
            self._request_threads = {
                thread for thread in self._request_threads if thread.is_alive()
            }
            candidates.extend(self._request_threads)
        unique: dict[int, threading.Thread] = {}
        for thread in candidates:
            if thread is not None and thread is not current and thread.is_alive():
                unique[id(thread)] = thread
        return tuple(unique.values())

    def shutdown(self, hard_timeout: float = 5.0) -> None:
        if hard_timeout < 0:
            raise ValueError("hard_timeout must be non-negative")
        deadline = time.monotonic() + hard_timeout
        lock_timeout = max(0, deadline - time.monotonic())
        if lock_timeout == 0:
            acquired = self._shutdown_call_lock.acquire(blocking=False)
        else:
            acquired = self._shutdown_call_lock.acquire(timeout=lock_timeout)
        if not acquired:
            return
        try:
            with self._state_lock:
                if self._shutdown_complete:
                    return
                first_call = not self._shutdown_started
            if first_call:
                self._begin_shutdown()

            while True:
                threads = self._live_shutdown_threads()
                remaining = deadline - time.monotonic()
                if not threads or remaining <= 0:
                    break
                slice_seconds = min(.02, remaining)
                for thread in threads:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    thread.join(min(slice_seconds, remaining))

            complete = not self._live_shutdown_threads()
            with self._state_lock:
                self._shutdown_complete = complete
        finally:
            self._shutdown_call_lock.release()
