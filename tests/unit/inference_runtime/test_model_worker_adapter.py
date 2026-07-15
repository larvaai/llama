from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import pytest

from inference_runtime import (
    SchedulerEvent,
    SchedulerEventKind,
    SchedulingMetadata,
    require_managed_backend,
)
from inference_runtime.adapters import SerialModelWorkerAdapter
from model_worker.errors import WorkerError
from model_worker.preflight import preflight
from model_worker.request_registry import Lifecycle, RequestRegistry


@dataclass
class _Result:
    value: str


class _Dispatcher:
    def __init__(self, result: Any = _Result("ok")) -> None:
        self.registry = RequestRegistry()
        self.result = result
        self.submitted = threading.Event()
        self.release = threading.Event()
        self.cancelled: list[str] = []

    def submit(self, request):
        record = self.registry.create(request, 1000, 1000)
        self.registry.transition(record, Lifecycle.PREFLIGHTED)
        self.registry.transition(record, Lifecycle.QUEUED)
        self.registry.transition(record, Lifecycle.RUNNING)
        self.submitted.set()
        return record

    def wait(self, record, timeout=None):
        self.release.wait(1)
        if record.cancel_event.is_set():
            self.registry.transition(record, Lifecycle.CANCELLED, error="cancelled")
        elif isinstance(self.result, BaseException):
            self.registry.transition(record, Lifecycle.FAILED, error=self.result)
        else:
            self.registry.transition(record, Lifecycle.COMPLETED, result=self.result)
        return record

    def cancel(self, request_id):
        record = self.registry.get(request_id)
        if record is None:
            return False
        record.cancel_event.set()
        self.cancelled.append(request_id)
        self.release.set()
        return True


class _Sink:
    def __init__(self) -> None:
        self.events: list[SchedulerEvent] = []

    def publish(self, event: SchedulerEvent) -> None:
        self.events.append(event)


def _metadata(request_id: str = "external-request") -> SchedulingMetadata:
    return SchedulingMetadata(
        request_id=request_id,
        workflow_id="workflow",
        agent_id="agent",
        service_class="throughput",
        weight=1,
        deadline_monotonic=None,
    )


def test_serial_adapter_advertises_only_managed_capabilities(
    manifest,
    request_body,
):
    adapter = SerialModelWorkerAdapter(_Dispatcher(), manifest)
    assert require_managed_backend(adapter) is adapter
    assert adapter.capabilities.supports_full_request
    assert not adapter.capabilities.supports_sequence_steps
    assert not adapter.capabilities.supports_streaming
    assert adapter.capabilities.max_concurrent_sequences is None


def test_serial_adapter_preserves_completion_and_emits_terminal_once(
    manifest,
    request_body,
):
    dispatcher = _Dispatcher()
    dispatcher.release.set()
    adapter = SerialModelWorkerAdapter(dispatcher, manifest, clock=lambda: 5.0)
    sink = _Sink()

    result = adapter.generate(
        preflight(request_body, manifest),
        scheduling=_metadata(),
        events=sink,
    )

    assert result == _Result("ok")
    assert [event.kind for event in sink.events] == [
        SchedulerEventKind.ADMITTED,
        SchedulerEventKind.REQUEST_COMPLETED,
    ]
    assert not adapter.cancel("external-request")


def test_serial_adapter_maps_external_cancel_to_internal_record(
    manifest,
    request_body,
):
    dispatcher = _Dispatcher()
    adapter = SerialModelWorkerAdapter(dispatcher, manifest)
    sink = _Sink()
    outcome: list[BaseException] = []

    def invoke() -> None:
        try:
            adapter.generate(
                preflight(request_body, manifest),
                scheduling=_metadata(),
                events=sink,
            )
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    assert dispatcher.submitted.wait(1)
    assert adapter.cancel("external-request")
    thread.join(1)

    assert not thread.is_alive()
    assert len(dispatcher.cancelled) == 1
    assert isinstance(outcome[0], WorkerError)
    assert outcome[0].code == "cancelled"
    assert [event.kind for event in sink.events].count(
        SchedulerEventKind.REQUEST_FAILED
    ) == 1


def test_serial_adapter_rejects_expired_deadline_before_submit(
    manifest,
    request_body,
):
    dispatcher = _Dispatcher()
    adapter = SerialModelWorkerAdapter(dispatcher, manifest, clock=lambda: 10.0)
    sink = _Sink()
    metadata = SchedulingMetadata(
        request_id="expired",
        workflow_id="workflow",
        agent_id="agent",
        service_class="interactive",
        weight=10,
        deadline_monotonic=9.0,
    )

    with pytest.raises(WorkerError, match="already expired") as raised:
        adapter.generate(
            preflight(request_body, manifest),
            scheduling=metadata,
            events=sink,
        )

    assert raised.value.code == "deadline_exceeded"
    assert not dispatcher.submitted.is_set()
    assert [event.kind for event in sink.events] == [
        SchedulerEventKind.REQUEST_FAILED
    ]


def test_serial_adapter_preserves_typed_worker_error(
    manifest,
    request_body,
):
    dispatcher = _Dispatcher(WorkerError("output_invalid", "bad output"))
    dispatcher.release.set()
    adapter = SerialModelWorkerAdapter(dispatcher, manifest)
    sink = _Sink()

    with pytest.raises(WorkerError) as raised:
        adapter.generate(
            preflight(request_body, manifest),
            scheduling=_metadata(),
            events=sink,
        )

    assert raised.value.code == "output_invalid"
    assert sink.events[-1].error_code == "output_invalid"
