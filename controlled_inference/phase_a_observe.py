"""Phase A: observe reasoning/final boundaries without applying a grammar."""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "controlled_inference" / "artifacts" / "phase-a-observation.json"
RUNTIME = Path(r"C:\Users\namso\.lmstudio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.24.0")
VENDOR = Path(r"C:\Users\namso\.lmstudio\extensions\backends\vendor\win-llama-cuda12-vendor-v2")
SERVER = RUNTIME / "llama-server.exe"
MODEL = Path(r"C:\Users\namso\.lmstudio\models\LuffyTheFox\Qwen3.5-9B-Claude-4.6-Opus-Uncensored-Distilled-GGUF\Qwen3.5-9B.Q4_K_M.gguf")
BASE_URL = "http://127.0.0.1:8080"


def post(path, payload, timeout=30):
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=timeout)


def wait_ready(timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(BASE_URL + "/health", timeout=1) as response:
                if json.load(response).get("status") == "ok":
                    return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("llama-server did not become ready")


def start_server():
    env = os.environ.copy()
    env["PATH"] = str(VENDOR) + os.pathsep + str(RUNTIME) + os.pathsep + env.get("PATH", "")
    log_path = ROOT / "controlled_inference" / "artifacts" / "phase-a-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [str(SERVER), "-m", str(MODEL), "--host", "127.0.0.1", "--port", "8080",
         "--jinja", "--reasoning", "on", "--reasoning-format", "deepseek",
         "--reasoning-budget", "-1", "-c", "8192", "-np", "1", "-ngl", "99",
         "--log-verbosity", "2", "--log-colors", "off"],
        cwd=RUNTIME, env=env, stdout=log, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        wait_ready()
    except Exception:
        process.kill()
        log.close()
        raise
    return process, log


def tokenize(text):
    if not text:
        return []
    with post("/tokenize", {"content": text, "add_special": False}, timeout=10) as response:
        data = json.load(response)
    return data.get("tokens", [])


def observe():
    payload = {
        "model": MODEL.name,
        "temperature": 0,
        "max_tokens": 1024,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
        "messages": [
            {"role": "system", "content": "Think first. Then answer with one short JSON object."},
            {"role": "user", "content": "Count labels starting with A ignoring case: Alpha, beta, atlas, Gamma."},
        ],
    }
    started = time.monotonic()
    events = []
    reasoning_parts, content_parts = [], []
    with post("/v1/chat/completions", payload, timeout=120) as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            delta = event.get("choices", [{}])[0].get("delta", {})
            reasoning = delta.get("reasoning_content") or ""
            content = delta.get("content") or ""
            if reasoning:
                reasoning_parts.append(reasoning)
                phase = "thinking"
                fragment = reasoning
            elif content:
                content_parts.append(content)
                phase = "final"
                fragment = content
            else:
                continue
            events.append({
                "index": len(events),
                "ms": round((time.monotonic() - started) * 1000, 1),
                "phase": phase,
                "text": fragment,
                "token_ids": tokenize(fragment),
            })

    reasoning = "".join(reasoning_parts)
    content = "".join(content_parts)
    first_final = next((i for i, event in enumerate(events) if event["phase"] == "final"), None)
    transitions = [
        {"from": events[i - 1]["phase"], "to": events[i]["phase"], "event_index": i}
        for i in range(1, len(events)) if events[i]["phase"] != events[i - 1]["phase"]
    ]
    return {
        "phase": "A-token-observation",
        "grammar_enabled": False,
        "thinking_enabled": True,
        "reasoning_format": "deepseek",
        "model": str(MODEL),
        "boundary_observation": {
            "first_final_event": first_final,
            "transitions": transitions,
            "note": "The HTTP parser removes the internal end-of-reasoning marker; exact control-token proof requires the C API in Phase B.",
        },
        "reasoning": {"text": reasoning, "token_ids": tokenize(reasoning)},
        "final": {"text": content, "token_ids": tokenize(content)},
        "candidate_markers": {
            "<think>": tokenize("<think>"),
            "</think>": tokenize("</think>"),
        },
        "events": events,
    }


if __name__ == "__main__":
    process, log = start_server()
    try:
        result = observe()
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "artifact": str(OUT),
            "reasoning_tokens": len(result["reasoning"]["token_ids"]),
            "final_tokens": len(result["final"]["token_ids"]),
            "transitions": result["boundary_observation"]["transitions"],
            "final": result["final"]["text"],
        }, ensure_ascii=False, indent=2))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
