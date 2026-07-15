import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PORT = 18091
BASE = f"http://127.0.0.1:{PORT}"


def get_json(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as response:
        return response.status, json.load(response)


def post_json(payload, timeout=180):
    request = urllib.request.Request(
        BASE + "/v1/controlled/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def wait_for_health(timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, health = get_json("/health")
            if status == 200 and health["model_resident"]:
                return health
        except Exception:
            pass
        time.sleep(0.1)
    raise TimeoutError("persistent service did not become healthy")


def main():
    subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "build_phase_b.ps1")],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    service_log = ARTIFACTS / "phase-g-service.log"
    with service_log.open("wb") as log:
        service = subprocess.Popen(
            [sys.executable, str(ROOT / "controlled_service.py"), "--port", str(PORT), "--testing"],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    cases = [
        {
            "request_id": "persistent-one-en",
            "task": "Count labels starting with A ignoring case: Alpha, beta, atlas, Gamma. The result is 2.",
            "response_language": "en",
            "expected_result": 2,
        },
        {
            "request_id": "persistent-two-vi",
            "task": "Đếm các số chẵn trong danh sách 1, 2, 4, 7, 8. Kết quả là 3.",
            "response_language": "vi",
            "expected_result": 3,
        },
        {
            "request_id": "persistent-three-en",
            "task": "Count the word red in: red, blue, green. The result is 1.",
            "response_language": "en",
            "expected_result": 1,
        },
    ]
    checks = {}
    responses = []
    try:
        initial_health = wait_for_health()
        checks["model_resident_before_requests"] = (
            initial_health["model_resident"]
            and initial_health["model_loads_total"] == 1
            and initial_health["process_generation"] == 1
        )
        for case in cases:
            status, response = post_json(case)
            responses.append(response)
            if status != 200:
                raise RuntimeError(f"request {case['request_id']} failed: {status} {response}")

        workers = [response["worker"] for response in responses]
        checks["three_sequential_requests_accepted"] = all(response["accepted"] for response in responses)
        checks["results_do_not_leak_between_requests"] = [response["final"]["result"] for response in responses] == [2, 3, 1]
        checks["same_worker_process"] = [worker["process_generation"] for worker in workers] == [1, 1, 1]
        checks["model_loaded_once"] = [worker["model_loads_total"] for worker in workers] == [1, 1, 1]
        checks["request_ordinals_increment"] = [worker["request_ordinal"] for worker in workers] == [1, 2, 3]
        checks["fresh_context_per_request"] = all(worker["context_fresh"] for worker in workers)

        metas = []
        for case in cases:
            token_path = ARTIFACTS / "service" / "requests" / case["request_id"] / "tokens.jsonl"
            rows = [json.loads(line) for line in token_path.read_text(encoding="utf-8").splitlines()]
            metas.append(rows[0])
        checks["artifacts_confirm_single_load_fresh_contexts"] = (
            [meta["model_load_count"] for meta in metas] == [1, 1, 1]
            and [meta["request_ordinal"] for meta in metas] == [1, 2, 3]
            and all(meta["context_fresh"] for meta in metas)
        )
        checks["vietnamese_round_trip"] = (
            responses[1]["final"]["result"] == 3
            and isinstance(responses[1]["final"]["evidence"], str)
            and "�" not in responses[1]["final"]["evidence"]
        )
        final_health = get_json("/health")[1]
        checks["model_still_resident_after_requests"] = (
            final_health["model_resident"]
            and final_health["model_loads_total"] == 1
            and final_health["process_generation"] == 1
        )
    finally:
        service.terminate()
        try:
            service.wait(timeout=10)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait()

    result = {
        "phase": "G-persistent-sequential-worker",
        "passed": all(checks.values()),
        "checks": checks,
        "requests": [
            {
                "request_id": response.get("request_id"),
                "result": (response.get("final") or {}).get("result"),
                "worker": response.get("worker"),
                "duration_seconds": response.get("duration_seconds"),
            }
            for response in responses
        ],
    }
    output = ARTIFACTS / "phase-g-persistent-results.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
