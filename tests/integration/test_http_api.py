from __future__ import annotations

import http.client
import json
import threading

from model_worker.dispatcher import Dispatcher
from model_worker.http_api import ModelWorkerHTTPServer
from model_worker.security import ExposurePolicy


class Worker:
    def execute(self, record): return {"protocol_version":"model-worker.v1","request_id":record.request_id,"attempt_id":record.attempt_id,"termination":"completed","protocol_valid":True,"output_valid":True,"output":{"result":"ok"},"error":None}
    def cancel(self, record): record.cancel_event.set()
    def kill_and_restart(self): return True
    def shutdown(self): pass


def test_http_preflight_health_and_oversize(manifest, request_body):
    dispatcher = Dispatcher(Worker(), capacity=2)
    server = ModelWorkerHTTPServer(("127.0.0.1", 0), dispatcher, manifest, ExposurePolicy(), read_timeout=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    connection = http.client.HTTPConnection(*server.server_address, timeout=2)
    connection.request("GET", "/ready"); ready = connection.getresponse()
    assert ready.status == 200 and json.loads(ready.read())["manifest_digest"].startswith("sha256:")
    body = json.dumps(request_body).encode()
    connection.request("POST", "/v1/model/generate", body=body, headers={"Content-Type":"application/json","Content-Length":str(len(body))})
    response = connection.getresponse(); payload=json.loads(response.read())
    assert response.status == 200 and payload["output_valid"] is True and "accepted" not in payload
    assert server.metrics.samples["queue_wait_ms"]
    connection.request("POST", "/v1/model/generate", body=b"{}", headers={"Content-Length":str(manifest.limits["input_bytes"] + manifest.limits["schema_bytes"] + 65537)})
    oversized=connection.getresponse(); assert oversized.status == 413; oversized.read()
    connection.close(); server.shutdown(); server.server_close(); thread.join(1)


def test_preflight_failure_never_enqueues(manifest, request_body):
    worker=Worker(); dispatcher=Dispatcher(worker, capacity=1)
    server=ModelWorkerHTTPServer(("127.0.0.1",0),dispatcher,manifest,ExposurePolicy())
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    bad=dict(request_body); bad["stream"]={"enabled":"false"}; raw=json.dumps(bad).encode()
    connection=http.client.HTTPConnection(*server.server_address,timeout=2)
    connection.request("POST","/v1/model/generate",body=raw,headers={"Content-Length":str(len(raw))})
    response=connection.getresponse(); assert response.status==400; response.read()
    assert dispatcher.registry.snapshot() == ()
    connection.close(); server.shutdown(); server.server_close(); thread.join(1)
