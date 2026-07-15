import argparse
import json
import queue
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from compile_schema_subset import compile_schema
from prepare_phase_d_prompt import prepare_prompts


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts" / "service"
PERSISTENT_EXE = ROOT / "build" / "persistent_worker.exe"
MODEL = Path(r"C:\Users\namso\.lmstudio\models\LuffyTheFox\Qwen3.5-9B-Claude-4.6-Opus-Uncensored-Distilled-GGUF\Qwen3.5-9B.Q4_K_M.gguf")
DEFAULT_SCHEMA = json.loads((ROOT / "phase_d_schema.json").read_text(encoding="utf-8"))


def validate_value(value, declared):
    types = [declared] if isinstance(declared, str) else declared
    checks = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: type(item) is int,
        "boolean": lambda item: type(item) is bool,
        "null": lambda item: item is None,
    }
    return any(item in checks and checks[item](value) for item in types)


def validate_schema_subset(value, schema):
    properties = schema["properties"]
    return (
        isinstance(value, dict)
        and list(value) == schema["required"] == list(properties)
        and all(validate_value(value[name], spec["type"]) for name, spec in properties.items())
        and all("enum" not in spec or value[name] in spec["enum"] for name, spec in properties.items())
    )


class PersistentWorker:
    """One isolated C++ process; model persists, request contexts do not."""

    def __init__(self):
        self.lock = threading.Lock()
        self.process = None
        self.runtime = None
        self.process_generation = 0
        self.model_loads_total = 0

    def start(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return True
            self._close_locked()
            runtime_path = ARTIFACTS / "persistent-worker-runtime.log"
            self.runtime = runtime_path.open("ab")
            self.process = subprocess.Popen(
                [str(PERSISTENT_EXE), str(MODEL), "99"],
                cwd=ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.runtime,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            process = self.process

        ready_box = queue.Queue(maxsize=1)

        def read_ready():
            try:
                ready_box.put(process.stdout.readline())
            except Exception as exc:
                ready_box.put(exc)

        reader = threading.Thread(target=read_ready, daemon=True)
        reader.start()
        try:
            item = ready_box.get(timeout=120)
        except queue.Empty:
            self.stop(force=True)
            return False
        if isinstance(item, Exception) or not item:
            self.stop(force=True)
            return False
        try:
            ready = json.loads(item)
        except json.JSONDecodeError:
            self.stop(force=True)
            return False
        if ready.get("type") != "worker_ready":
            self.stop(force=True)
            return False
        with self.lock:
            if self.process is process and process.poll() is None:
                self.process_generation += 1
                self.model_loads_total += 1
                return True
        return False

    def is_ready(self):
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def stop(self, force=False):
        with self.lock:
            process = self.process
            if process is not None and process.poll() is None:
                try:
                    if not force:
                        process.stdin.write("shutdown\n")
                        process.stdin.flush()
                        process.wait(timeout=5)
                    else:
                        process.kill()
                        process.wait(timeout=5)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait()
            self._close_locked()

    def _close_locked(self):
        if self.process is not None:
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        pass
        if self.runtime is not None:
            self.runtime.close()
        self.process = None
        self.runtime = None

    def crash_for_test(self):
        self.stop(force=True)

    def request(self, fields, poll_callback):
        if not self.start():
            return None, {"error": "worker_start_failed"}
        with self.lock:
            process = self.process
            generation = self.process_generation
            loads = self.model_loads_total
            try:
                process.stdin.write("\t".join(fields) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                return None, {
                    "error": "worker_pipe_failed",
                    "worker_exit": process.poll(),
                    "process_generation": generation,
                    "model_loads_total": loads,
                }

        response_box = queue.Queue(maxsize=1)

        def read_response():
            try:
                response_box.put(process.stdout.readline())
            except Exception as exc:
                response_box.put(exc)

        reader = threading.Thread(target=read_response, daemon=True)
        reader.start()
        while reader.is_alive():
            poll_callback()
            if process.poll() is not None:
                reader.join(timeout=1)
                return None, {
                    "error": "persistent_worker_crashed",
                    "worker_exit": process.returncode,
                    "process_generation": generation,
                    "model_loads_total": loads,
                }
            time.sleep(0.015)
        item = response_box.get()
        if isinstance(item, Exception) or not item:
            return None, {
                "error": "worker_response_missing",
                "worker_exit": process.poll(),
                "process_generation": generation,
                "model_loads_total": loads,
            }
        try:
            response = json.loads(item)
        except json.JSONDecodeError:
            return None, {
                "error": "worker_response_invalid",
                "process_generation": generation,
                "model_loads_total": loads,
            }
        response["process_generation"] = generation
        response["model_loads_total"] = loads
        poll_callback()
        return response, None


class ServiceState:
    def __init__(self, testing=False):
        self.lock = threading.Lock()
        self.worker_lock = threading.Lock()
        self.active = False
        self.waiting = 0
        self.testing = testing
        self.cancel_files = {}
        self.metrics = {
            "requests_total": 0,
            "completed_total": 0,
            "failed_total": 0,
            "cancelled_total": 0,
            "queue_rejected_total": 0,
            "worker_crashes_total": 0,
            "duration_seconds_sum": 0.0,
            "duration_seconds_count": 0,
        }

    def reserve(self):
        with self.lock:
            self.metrics["requests_total"] += 1
            if not self.active:
                self.active = True
                queued = False
            elif self.waiting < 1:
                self.waiting += 1
                queued = True
            else:
                self.metrics["queue_rejected_total"] += 1
                return None
        self.worker_lock.acquire()
        if queued:
            with self.lock:
                self.waiting -= 1
                self.active = True
        return queued

    def release(self):
        with self.lock:
            if self.waiting == 0:
                self.active = False
        self.worker_lock.release()

    def snapshot(self):
        with self.lock:
            return {
                "active": self.active,
                "queue_depth": self.waiting,
                "metrics": dict(self.metrics),
            }


class ControlledHandler(BaseHTTPRequestHandler):
    server_version = "ControlledInference/0.2"

    @property
    def state(self):
        return self.server.state

    def log_message(self, fmt, *args):
        return

    def send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_sse(self, event, payload):
        data = json.dumps(payload, ensure_ascii=False)
        self.wfile.write(f"event: {event}\ndata: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/health":
            snapshot = self.state.snapshot()
            ready = PERSISTENT_EXE.exists() and MODEL.exists() and self.server.worker.is_ready()
            self.send_json(200 if ready else 503, {
                "status": "ok" if ready else "not_ready",
                "decoder_ready": ready,
                "model_resident": ready,
                "process_generation": self.server.worker.process_generation,
                "model_loads_total": self.server.worker.model_loads_total,
                "worker_busy": snapshot["active"],
                "queue_depth": snapshot["queue_depth"],
            })
            return
        if self.path == "/metrics":
            snapshot = self.state.snapshot()
            lines = [
                "# TYPE controlled_requests_total counter",
                f"controlled_requests_total {snapshot['metrics']['requests_total']}",
                f"controlled_completed_total {snapshot['metrics']['completed_total']}",
                f"controlled_failed_total {snapshot['metrics']['failed_total']}",
                f"controlled_cancelled_total {snapshot['metrics']['cancelled_total']}",
                f"controlled_queue_rejected_total {snapshot['metrics']['queue_rejected_total']}",
                f"controlled_worker_crashes_total {snapshot['metrics']['worker_crashes_total']}",
                f"controlled_model_loads_total {self.server.worker.model_loads_total}",
                f"controlled_worker_process_generation {self.server.worker.process_generation}",
                f"controlled_duration_seconds_sum {snapshot['metrics']['duration_seconds_sum']:.6f}",
                f"controlled_duration_seconds_count {snapshot['metrics']['duration_seconds_count']}",
                "# TYPE controlled_worker_active gauge",
                f"controlled_worker_active {1 if snapshot['active'] else 0}",
                f"controlled_queue_depth {snapshot['queue_depth']}",
                "",
            ]
            body = "\n".join(lines).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_json(404, {"error": "not_found"})

    def do_POST(self):
        if self.path.startswith("/v1/controlled/cancel/"):
            request_id = self.path.rsplit("/", 1)[-1]
            with self.state.lock:
                cancel_file = self.state.cancel_files.get(request_id)
            if not cancel_file:
                self.send_json(404, {"error": "request_not_active", "request_id": request_id})
            else:
                cancel_file.write_text("cancel", encoding="ascii")
                self.send_json(202, {"status": "cancellation_requested", "request_id": request_id})
            return
        if self.path != "/v1/controlled/generate":
            self.send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            self.handle_generate(body)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": "invalid_request", "detail": str(exc)})

    def handle_generate(self, body):
        task = body.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        language = body.get("response_language", "auto")
        if language not in ("auto", "vi", "en"):
            raise ValueError("response_language must be auto, vi, or en")
        compile_schema(body.get("schema") or DEFAULT_SCHEMA)  # Fail before queueing or sending SSE headers.
        for name in ("reasoning_budget", "final_budget", "total_budget"):
            if name in body and (type(body[name]) is not int or body[name] <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if body.get("simulate_worker_crash") and not self.state.testing:
            raise ValueError("simulate_worker_crash is only available in testing mode")
        stream = bool(body.get("stream", False))
        request_id = body.get("request_id") or uuid.uuid4().hex
        if not request_id.replace("-", "").isalnum():
            raise ValueError("request_id must be alphanumeric or hyphen")

        snapshot = self.state.snapshot()
        would_queue = snapshot["active"]
        if would_queue and snapshot["queue_depth"] >= 1:
            with self.state.lock:
                self.state.metrics["requests_total"] += 1
                self.state.metrics["queue_rejected_total"] += 1
            self.send_json(429, {"error": "queue_full", "capacity": 1})
            return
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.send_sse("queued", {"request_id": request_id, "queued": would_queue})

        queued = self.state.reserve()
        if queued is None:
            if stream:
                self.send_sse("error", {"error": "queue_full"})
            else:
                self.send_json(429, {"error": "queue_full", "capacity": 1})
            return
        started = time.monotonic()
        try:
            result, http_status = self.run_worker(body, request_id, stream)
            duration = time.monotonic() - started
            with self.state.lock:
                self.state.metrics["duration_seconds_sum"] += duration
                self.state.metrics["duration_seconds_count"] += 1
                termination = result.get("termination")
                if result.get("worker_crash"):
                    self.state.metrics["worker_crashes_total"] += 1
                    self.state.metrics["failed_total"] += 1
                elif termination == "completed" and result.get("accepted"):
                    self.state.metrics["completed_total"] += 1
                elif termination == "cancelled":
                    self.state.metrics["cancelled_total"] += 1
                else:
                    self.state.metrics["failed_total"] += 1
            result["duration_seconds"] = round(duration, 4)
            if stream:
                self.send_sse("result", result)
            else:
                self.send_json(http_status, result)
        except (BrokenPipeError, ConnectionResetError):
            with self.state.lock:
                cancel_file = self.state.cancel_files.get(request_id)
            if cancel_file:
                cancel_file.write_text("cancel", encoding="ascii")
        finally:
            with self.state.lock:
                self.state.cancel_files.pop(request_id, None)
            self.state.release()

    def run_worker(self, body, request_id, stream):
        request_dir = ARTIFACTS / "requests" / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        schema = body.get("schema") or DEFAULT_SCHEMA
        grammar_text = compile_schema(schema)
        grammar_path = request_dir / "schema.gbnf"
        system_path = request_dir / "system.txt"
        user_path = request_dir / "user.txt"
        tokens_path = request_dir / "tokens.jsonl"
        cancel_path = request_dir / "cancel.flag"
        for path in (tokens_path, cancel_path):
            path.unlink(missing_ok=True)
        grammar_path.write_text(grammar_text, encoding="utf-8")
        system_prompt, user_prompt = prepare_prompts({
            "task": body["task"],
            "response_language": body.get("response_language", "auto"),
        })
        system_path.write_text(system_prompt, encoding="utf-8")
        user_path.write_text(user_prompt, encoding="utf-8")
        with self.state.lock:
            self.state.cancel_files[request_id] = cancel_path

        if body.get("simulate_worker_crash") and self.state.testing:
            generation = self.server.worker.process_generation
            loads = self.server.worker.model_loads_total
            self.server.worker.crash_for_test()
            recovered = self.server.worker.start()
            return ({
                "request_id": request_id,
                "worker_crash": True,
                "worker_exit": -9,
                "error": "simulated_persistent_worker_crash",
                "process_generation": generation,
                "model_loads_total": loads,
                "worker_recovered": recovered,
            }, 502)

        emitted = 0
        disconnected = False

        def emit_available():
            nonlocal emitted, disconnected
            if not stream or not tokens_path.exists() or disconnected:
                return
            try:
                rows = tokens_path.read_text(encoding="utf-8").splitlines()
                for line in rows[emitted:]:
                    event = json.loads(line)
                    emitted += 1
                    if event.get("type") == "token" and event.get("text_delta"):
                        self.send_sse(event["stream_channel"], {
                            "request_id": request_id,
                            "delta": event["text_delta"],
                        })
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            except (BrokenPipeError, ConnectionResetError):
                cancel_path.write_text("cancel", encoding="ascii")
                disconnected = True

        fields = [
            request_id,
            str(tokens_path.resolve()),
            str(grammar_path.resolve()),
            str(system_path.resolve()),
            str(user_path.resolve()),
            str(int(body.get("reasoning_budget", 768))),
            str(int(body.get("final_budget", 256))),
            str(int(body.get("total_budget", 1024))),
            str(cancel_path.resolve()),
            "248069",
        ]
        worker_response, worker_error = self.server.worker.request(fields, emit_available)
        emit_available()
        if worker_error:
            recovered = self.server.worker.start()
            return ({
                "request_id": request_id,
                "worker_crash": True,
                **worker_error,
                "worker_recovered": recovered,
            }, 502)
        exit_code = worker_response["exit_code"]

        if not tokens_path.exists():
            return ({
                "request_id": request_id,
                "worker_crash": True,
                "worker_exit": exit_code,
                "error": "worker_exited_without_artifact",
            }, 502)
        rows = [json.loads(line) for line in tokens_path.read_text(encoding="utf-8").splitlines()]
        summary = next((row for row in rows if row.get("type") == "summary"), None)
        if summary is None:
            return ({
                "request_id": request_id,
                "worker_crash": True,
                "worker_exit": exit_code,
                "error": "worker_missing_summary",
            }, 502)
        final = None
        schema_valid = False
        if summary["termination"] == "completed" and summary["final_text"] is not None:
            final = json.loads(summary["final_text"])
            schema_valid = validate_schema_subset(final, schema)
        expected_result = body.get("expected_result")
        acceptance_valid = expected_result is None or (final is not None and final.get("result") == expected_result)
        accepted = summary["termination"] == "completed" and schema_valid and acceptance_valid
        result = {
            "request_id": request_id,
            "termination": summary["termination"],
            "accepted": accepted,
            "schema_valid": schema_valid,
            "acceptance_valid": acceptance_valid,
            "final": final,
            "usage": {
                "sampled_tokens": summary["sampled_tokens"],
                "thinking_tokens": summary["thinking_tokens"],
                "final_tokens": summary["final_tokens"],
            },
            "worker_exit": exit_code,
            "worker": {
                "persistent": True,
                "process_generation": worker_response["process_generation"],
                "request_ordinal": worker_response["request_ordinal"],
                "model_load_count": worker_response["model_load_count"],
                "model_loads_total": worker_response["model_loads_total"],
                "context_fresh": worker_response["context_fresh"],
            },
        }
        return result, 200 if accepted or summary["termination"] != "completed" else 422


class ControlledHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, testing=False):
        super().__init__(address, ControlledHandler)
        self.state = ServiceState(testing=testing)
        self.worker = PersistentWorker()
        if not self.worker.start():
            self.server_close()
            raise RuntimeError("persistent worker failed to start")

    def server_close(self):
        if hasattr(self, "worker"):
            self.worker.stop()
        super().server_close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--testing", action="store_true")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    server = ControlledHTTPServer((args.host, args.port), testing=args.testing)
    print(json.dumps({"status": "listening", "host": args.host, "port": args.port}), flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
