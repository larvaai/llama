from __future__ import annotations

import queue
import threading
import time
from typing import Any, Protocol

from .errors import WorkerError
from .request_registry import Lifecycle, RequestRecord, RequestRegistry, TERMINAL


class WorkerBackend(Protocol):
    def execute(self, record: RequestRecord) -> Any: ...
    def cancel(self, record: RequestRecord) -> None: ...
    def kill_and_restart(self) -> bool: ...
    def shutdown(self) -> None: ...


class Dispatcher:
    def __init__(self, worker: WorkerBackend, *, capacity: int, watchdog_grace_ms: int = 500) -> None:
        self.registry = RequestRegistry()
        self.worker = worker
        self.queue: queue.Queue[RequestRecord | None] = queue.Queue(maxsize=capacity)
        self.watchdog_grace = watchdog_grace_ms / 1000
        self._accepting = True
        self._thread = threading.Thread(target=self._run, name="model-worker-dispatcher", daemon=True)
        self._thread.start()

    def submit(self, request: Any) -> RequestRecord:
        if not self._accepting:
            raise WorkerError("shutdown", "service is draining")
        record = self.registry.create(request, request.limits.queue_timeout_ms, request.limits.execution_timeout_ms)
        self.registry.transition(record, Lifecycle.PREFLIGHTED)
        self.registry.transition(record, Lifecycle.QUEUED)
        try:
            self.queue.put_nowait(record)
        except queue.Full as exc:
            self.registry.transition(record, Lifecycle.FAILED, error="queue_full")
            raise WorkerError("queue_full", "dispatcher queue is full") from exc
        return record

    def cancel(self, request_id: str) -> bool:
        record = self.registry.get(request_id)
        changed = self.registry.cancel(request_id)
        if changed and record is not None and record.lifecycle == Lifecycle.RUNNING:
            self.worker.cancel(record)
        return changed

    def wait(self, record: RequestRecord, timeout: float | None = None) -> RequestRecord:
        deadline = None if timeout is None else time.monotonic() + timeout
        with record.condition:
            while record.lifecycle not in TERMINAL:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                record.condition.wait(remaining)
        return record

    def _run(self) -> None:
        while True:
            record = self.queue.get()
            if record is None:
                self.queue.task_done()
                return
            try:
                if record.lifecycle in TERMINAL:
                    continue
                if time.monotonic() >= record.queue_deadline:
                    self.registry.transition(record, Lifecycle.TIMED_OUT, error="queue_timeout")
                    continue
                self.registry.transition(record, Lifecycle.RUNNING)
                outcome: dict[str, Any] = {}

                def invoke() -> None:
                    try:
                        outcome["result"] = self.worker.execute(record)
                    except BaseException as exc:  # contained at process boundary
                        outcome["error"] = exc

                task = threading.Thread(target=invoke, name=f"request-{record.request_id}", daemon=True)
                task.start()
                task.join(record.execution_timeout)
                if task.is_alive():
                    record.cancel_event.set()
                    self.worker.cancel(record)
                    task.join(self.watchdog_grace)
                    if task.is_alive():
                        self.worker.kill_and_restart()
                    self.registry.transition(record, Lifecycle.TIMED_OUT, error="deadline_exceeded")
                elif record.cancel_event.is_set():
                    self.registry.transition(record, Lifecycle.CANCELLED, error="cancelled")
                elif "error" in outcome:
                    error = outcome["error"]
                    stored = error if isinstance(error, WorkerError) else "worker_crashed"
                    self.registry.transition(record, Lifecycle.FAILED, error=stored)
                else:
                    self.registry.transition(record, Lifecycle.COMPLETED, result=outcome.get("result"))
            finally:
                self.queue.task_done()

    def shutdown(self, hard_timeout: float = 5.0) -> None:
        self._accepting = False
        for record in self.registry.snapshot():
            if record.lifecycle not in TERMINAL:
                self.cancel(record.request_id)
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            while True:
                try:
                    dropped = self.queue.get_nowait()
                    self.queue.task_done()
                    if dropped is not None and dropped.lifecycle not in TERMINAL:
                        self.registry.transition(dropped, Lifecycle.CANCELLED, error="shutdown")
                except queue.Empty:
                    break
            self.queue.put_nowait(None)
        self._thread.join(hard_timeout)
        self.worker.shutdown()
