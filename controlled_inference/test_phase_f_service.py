import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PORT = 18090
BASE = f"http://127.0.0.1:{PORT}"


def get_json(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as response:
        return response.status, json.load(response)


def post_json(path, payload, timeout=180):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def stream_request(payload):
    request = urllib.request.Request(
        BASE + "/v1/controlled/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    events = []
    with urllib.request.urlopen(request, timeout=180) as response:
        event_name = None
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                events.append((event_name, json.loads(line[6:])))
    return events


def wait_for(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except Exception:
            pass
        time.sleep(0.05)
    raise TimeoutError("condition not reached")


def main():
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "build_phase_b.ps1")],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    service_log = ARTIFACTS / "phase-f-service.log"
    with service_log.open("wb") as log:
        service = subprocess.Popen(
            [sys.executable, str(ROOT / "controlled_service.py"), "--port", str(PORT), "--testing"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        )
    checks = {}
    try:
        health = wait_for(lambda: get_json("/health")[1])
        checks["health_ready"] = health["status"] == "ok" and health["decoder_ready"]

        stream_box = {}
        stream_payload = {
            "request_id": "stream-vi",
            "task": "Đếm nhãn bắt đầu bằng A, không phân biệt hoa thường: Alpha, beta, atlas, Gamma. Kết quả là 2.",
            "response_language": "vi",
            "expected_result": 2,
            "stream": True,
        }
        stream_thread = threading.Thread(target=lambda: stream_box.setdefault("events", stream_request(stream_payload)))
        stream_thread.start()
        wait_for(lambda: get_json("/health")[1]["worker_busy"])

        queued_box = {}
        queued_payload = {
            "request_id": "queued-en",
            "task": "Count labels starting with A ignoring case: Alpha, beta, atlas, Gamma. The result is 2.",
            "response_language": "en",
            "expected_result": 2,
        }
        queued_thread = threading.Thread(
            target=lambda: queued_box.setdefault("response", post_json("/v1/controlled/generate", queued_payload))
        )
        queued_thread.start()
        wait_for(lambda: get_json("/health")[1]["queue_depth"] == 1)

        rejected_status, rejected = post_json("/v1/controlled/generate", {
            "request_id": "rejected-third",
            "task": "Count A labels: Alpha, atlas. Result 2.",
            "response_language": "en",
        })
        checks["queue_capacity_one"] = rejected_status == 429 and rejected["error"] == "queue_full"

        stream_thread.join(timeout=180)
        queued_thread.join(timeout=180)
        events = stream_box["events"]
        names = [name for name, _ in events]
        result_event = next(payload for name, payload in events if name == "result")
        thinking_text = "".join(payload["delta"] for name, payload in events if name == "thinking")
        final_text = "".join(payload["delta"] for name, payload in events if name == "final")
        checks["streaming_channels"] = bool(thinking_text) and bool(final_text) and {"thinking", "final", "result"} <= set(names)
        checks["streamed_final_is_complete_json"] = json.loads(final_text) == result_event["final"]
        checks["queued_request_completed"] = queued_box["response"][0] == 200 and queued_box["response"][1]["accepted"]

        crash_status, crash = post_json("/v1/controlled/generate", {
            "request_id": "forced-crash",
            "task": "test worker crash",
            "response_language": "en",
            "simulate_worker_crash": True,
        })
        checks["worker_crash_isolated"] = crash_status == 502 and crash["worker_crash"]
        checks["service_alive_after_crash"] = get_json("/health")[1]["status"] == "ok"

        recovery_status, recovery = post_json("/v1/controlled/generate", {
            "request_id": "after-crash",
            "task": "Count labels starting with A ignoring case: Alpha, beta, atlas, Gamma. The result is 2.",
            "response_language": "en",
            "expected_result": 2,
        })
        checks["request_recovers_after_worker_crash"] = recovery_status == 200 and recovery["accepted"]

        cancel_box = {}
        cancel_id = "cancel-through-api"
        cancel_tokens = ARTIFACTS / "service" / "requests" / cancel_id / "tokens.jsonl"
        cancel_tokens.unlink(missing_ok=True)
        cancel_thread = threading.Thread(target=lambda: cancel_box.setdefault("response", post_json(
            "/v1/controlled/generate",
            {
                "request_id": cancel_id,
                "task": "Count labels starting with A ignoring case: Alpha, beta, atlas, Gamma. The result is 2.",
                "response_language": "en",
            },
        )))
        cancel_thread.start()
        wait_for(lambda: cancel_tokens.exists() and cancel_tokens.read_text(encoding="utf-8", errors="ignore").count('"type":"token"') >= 10, timeout=120)
        cancel_status, cancel_ack = post_json(f"/v1/controlled/cancel/{cancel_id}", {})
        cancel_thread.join(timeout=180)
        checks["cancellation_api"] = (
            cancel_status == 202
            and cancel_ack["status"] == "cancellation_requested"
            and cancel_box["response"][1]["termination"] == "cancelled"
        )

        with urllib.request.urlopen(BASE + "/metrics", timeout=10) as response:
            metrics = response.read().decode("utf-8")
        checks["metrics_exposed"] = (
            "controlled_requests_total" in metrics
            and "controlled_worker_crashes_total 1" in metrics
            and "controlled_queue_rejected_total 1" in metrics
            and "controlled_cancelled_total 1" in metrics
        )
    finally:
        service.terminate()
        try:
            service.wait(timeout=5)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait()

    result = {
        "phase": "F-service-prototype",
        "passed": all(checks.values()),
        "checks": checks,
    }
    (ARTIFACTS / "phase-f-service-results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
