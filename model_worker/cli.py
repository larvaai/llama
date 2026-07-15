from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .dispatcher import Dispatcher
from .errors import WorkerError
from .http_api import ModelWorkerHTTPServer
from .manifest import load_manifest
from .security import ExposurePolicy
from .worker_process import NativeWorkerProcess


def validate_manifest_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model-worker-validate-manifest")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except (WorkerError, OSError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)})); return 1
    print(json.dumps({"valid": True, "model_id": manifest.id, "manifest_digest": manifest.digest, "runtime_build": manifest.raw["runtime_build"]})); return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model-worker")
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--native-executable", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8090, type=int)
    parser.add_argument("--tls-terminated", action="store_true")
    parser.add_argument("--trusted-reverse-proxy", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.model_manifest)
    policy = ExposurePolicy(args.host, os.environ.get("MODEL_WORKER_BEARER_TOKEN"), args.tls_terminated, args.trusted_reverse_proxy)
    policy.validate()
    worker = NativeWorkerProcess(args.native_executable.resolve(), manifest)
    worker.start()
    dispatcher = Dispatcher(worker, capacity=manifest.limits["max_queue"])
    server = ModelWorkerHTTPServer((args.host, args.port), dispatcher, manifest, policy)
    print(json.dumps({"status": "ready", "host": args.host, "port": args.port, "model_id": manifest.id, "manifest_digest": manifest.digest}), flush=True)
    try: server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0


if __name__ == "__main__": raise SystemExit(main())
