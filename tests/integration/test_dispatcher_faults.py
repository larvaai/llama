from __future__ import annotations

import time

from model_worker.dispatcher import Dispatcher
from model_worker.preflight import preflight
from model_worker.request_registry import Lifecycle


class FakeWorker:
    def __init__(self): self.mode="ok"; self.received=[]; self.kills=0; self.cancelled=0
    def execute(self, record):
        self.received.append(record.request_id)
        if self.mode == "hang":
            while not record.cancel_event.wait(.01): pass
            time.sleep(.2)
        if self.mode == "crash": raise RuntimeError("boom")
        return {"termination":"completed","protocol_valid":True,"output_valid":True,"output":{"result":"x"}}
    def cancel(self, record): self.cancelled += 1; record.cancel_event.set()
    def kill_and_restart(self): self.kills += 1; return True
    def shutdown(self): pass


def test_queued_cancel_never_reaches_model(manifest, request_body):
    worker = FakeWorker(); dispatcher = Dispatcher(worker, capacity=2)
    first = preflight(request_body, manifest); first.request.limits.__class__
    worker.mode = "hang"
    r1 = dispatcher.submit(first)
    time.sleep(.02)
    r2 = dispatcher.submit(first)
    assert dispatcher.cancel(r2.request_id)
    assert dispatcher.wait(r2, 1).lifecycle == Lifecycle.CANCELLED
    dispatcher.cancel(r1.request_id); dispatcher.wait(r1, 1)
    assert r2.request_id not in worker.received
    dispatcher.shutdown()


def test_watchdog_and_crash_do_not_block_next_request(manifest, request_body):
    worker = FakeWorker(); dispatcher = Dispatcher(worker, capacity=3, watchdog_grace_ms=10)
    prepared = preflight(request_body, manifest)
    worker.mode="hang"; timed = dispatcher.submit(prepared)
    assert dispatcher.wait(timed, 1).lifecycle == Lifecycle.TIMED_OUT
    assert worker.kills == 1
    worker.mode="crash"; crashed=dispatcher.submit(prepared)
    assert dispatcher.wait(crashed, 1).lifecycle == Lifecycle.FAILED
    worker.mode="ok"; good=dispatcher.submit(prepared)
    assert dispatcher.wait(good, 1).lifecycle == Lifecycle.COMPLETED
    dispatcher.shutdown()
