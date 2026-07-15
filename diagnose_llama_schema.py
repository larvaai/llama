import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = Path(r"C:\Users\namso\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.24.0")
VENDOR = Path(r"C:\Users\namso\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2")
SERVER = RUNTIME / "llama-server.exe"
MODEL = Path(r"C:\Users\namso\.lmstudio\models\LuffyTheFox\Qwen3.5-9B-Claude-4.6-Opus-Uncensored-Distilled-GGUF\Qwen3.5-9B.Q4_K_M.gguf")
URL = "http://127.0.0.1:8080"

BASE_PROPERTIES = {
    "status": {"type": "string"},
    "result": {"type": "integer"},
    "evidence": {"type": "string"},
    "reason": {"type": "null"},
}

CASES = [
    ("one_boolean", {"ok": {"type": "boolean"}}, ["ok"], True),
    ("status_only", {"status": {"type": "string"}}, ["status"], True),
    ("status_enum", {"status": {"type": "string", "enum": ["completed", "blocked"]}}, ["status"], True),
    ("status_result", {k: BASE_PROPERTIES[k] for k in ("status", "result")}, ["status", "result"], True),
    ("three_fields", {k: BASE_PROPERTIES[k] for k in ("status", "result", "evidence")}, ["status", "result", "evidence"], True),
    ("four_fields", BASE_PROPERTIES, list(BASE_PROPERTIES), True),
    ("result_nullable", {**BASE_PROPERTIES, "result": {"type": ["integer", "null"]}}, list(BASE_PROPERTIES), True),
    ("reason_nullable", {**BASE_PROPERTIES, "reason": {"type": ["string", "null"]}}, list(BASE_PROPERTIES), True),
    ("both_nullable", {**BASE_PROPERTIES, "result": {"type": ["integer", "null"]}, "reason": {"type": ["string", "null"]}}, list(BASE_PROPERTIES), True),
    ("with_enum", {**BASE_PROPERTIES, "status": {"type": "string", "enum": ["completed", "blocked"]}}, list(BASE_PROPERTIES), True),
    ("with_max_length", {**BASE_PROPERTIES, "evidence": {"type": "string", "maxLength": 80}}, list(BASE_PROPERTIES), True),
    ("allow_extra", BASE_PROPERTIES, list(BASE_PROPERTIES), False),
]


def wait_ready(timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=1) as response:
                if json.load(response).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def start_server():
    env = os.environ.copy()
    env["PATH"] = str(VENDOR) + os.pathsep + env.get("PATH", "")
    process = subprocess.Popen(
        [str(SERVER), "-m", str(MODEL), "--port", "8080", "--host", "127.0.0.1",
         "--jinja", "--reasoning", "on", "--reasoning-format", "deepseek",
         "-c", "8192", "-np", "1", "-ngl", "99"],
        cwd=RUNTIME,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if not wait_ready():
        process.kill()
        raise RuntimeError("llama-server did not become ready")
    return process


def stop_server(process):
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def run_case(name, properties, required, additional_properties):
    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": additional_properties,
    }
    payload = {
        "model": MODEL.name,
        "temperature": 0,
        "max_tokens": 256,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema},
        },
        "messages": [
            {"role": "system", "content": "Think first. Return a final JSON object matching the schema."},
            {"role": "user", "content": "Count labels starting with A ignoring case: Alpha, beta, atlas, Gamma. Use completed status, result 2, brief evidence, and null reason when those fields exist."},
        ],
    }
    request = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    reasoning_parts, content_parts, events = [], [], 0
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                delta = event.get("choices", [{}])[0].get("delta", {})
                reasoning_parts.append(delta.get("reasoning_content") or "")
                content_parts.append(delta.get("content") or "")
                events += 1
        content = "".join(content_parts).strip()
        reasoning = "".join(reasoning_parts).strip()
        parsed = json.loads(content)
        return {
            "case": name,
            "status": "passed",
            "seconds": round(time.monotonic() - started, 3),
            "events": events,
            "reasoning_chars": len(reasoning),
            "content": content,
            "parsed_keys": sorted(parsed),
        }
    except Exception as exc:
        return {
            "case": name,
            "status": "failed",
            "seconds": round(time.monotonic() - started, 3),
            "events": events,
            "reasoning_chars": len("".join(reasoning_parts)),
            "content_so_far": "".join(content_parts),
            "error": f"{type(exc).__name__}: {exc}",
        }


results = []
for case in CASES:
    server = start_server()
    try:
        result = run_case(*case)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        stop_server(server)

summary = {
    "thinking": True,
    "reasoning_format": "deepseek",
    "runtime": "llama.cpp 0eca4d4 (LM Studio runtime 2.24.0)",
    "passed": sum(item["status"] == "passed" for item in results),
    "failed": sum(item["status"] != "passed" for item in results),
    "results": results,
}
(ROOT / "llama-schema-diagnostic.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps({k: summary[k] for k in ("passed", "failed")}, indent=2))
