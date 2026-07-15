import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "qwen3.5-9b-claude-4.6-opus-uncensored-distilled"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

skill = (ROOT / "skills" / "atomic-worker" / "SKILL.md").read_text(encoding="utf-8")
task = json.loads((ROOT / "experiment-task.json").read_text(encoding="utf-8"))

payload = {
    "model": MODEL,
    "temperature": 0,
    "max_tokens": 2048,
    "messages": [
        {"role": "system", "content": skill},
        {"role": "user", "content": json.dumps(task, ensure_ascii=False)},
    ],
}

request = urllib.request.Request(
    "http://localhost:1234/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    raw_response = json.load(response)

message = raw_response["choices"][0]["message"]
content = (message.get("content") or "").strip()
if not content:
    (ROOT / "experiment-raw-response.json").write_text(
        json.dumps(raw_response, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "valid": False,
        "error": "message.content is empty; reasoning_content is never accepted",
        "content": content,
    }, ensure_ascii=False, indent=2))
    sys.exit(1)
try:
    output = json.loads(content)
except json.JSONDecodeError as exc:
    (ROOT / "experiment-raw-response.json").write_text(
        json.dumps(raw_response, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"valid": False, "error": f"invalid JSON: {exc}", "raw": content}, ensure_ascii=False, indent=2))
    sys.exit(1)

errors = []
if set(output) != {"status", "result", "evidence", "reason"}:
    errors.append("output keys do not match the required schema")
if output.get("status") != "completed":
    errors.append("status is not completed")
if output.get("result") != 2:
    errors.append("result is not 2")
if not isinstance(output.get("evidence"), str) or not output["evidence"].strip():
    errors.append("evidence is empty")
elif len(output["evidence"]) > 80:
    errors.append("evidence exceeds 80 characters")
if output.get("reason") is not None:
    errors.append("reason must be null for completed")

record = {
    "model": MODEL,
    "task": task,
    "output": output,
    "validation": {"passed": not errors, "errors": errors},
    "usage": raw_response.get("usage"),
}
(ROOT / "experiment-result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(record, ensure_ascii=False, indent=2))
sys.exit(0 if not errors else 2)
