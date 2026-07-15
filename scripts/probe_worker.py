from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1/model/generate")
    parser.add_argument("--model-id", default="qwen35-9b-local")
    args = parser.parse_args()
    schema = {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"], "additionalProperties": False}
    body = {
        "protocol_version": "model-worker.v1", "model_id": args.model_id,
        "messages": [
            {"role": "system", "content": "Think briefly and privately, then obey the output contract."},
            {"role": "user", "content": "Return the word ok in the result field."},
        ],
        "output_contract": {"version": "structured-output.v1", "schema": schema},
        "limits": {"reasoning_tokens": 256, "final_tokens": 64, "total_tokens": 300, "queue_timeout_ms": 5000, "execution_timeout_ms": 180000},
        "stream": {"enabled": False, "include_reasoning": False},
    }
    request = urllib.request.Request(args.url, json.dumps(body).encode(), {"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=190) as response:
            payload = json.load(response)
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
        payload = json.load(exc)
    print(json.dumps({"http_status": status, "response": payload}, ensure_ascii=False))
    return 0 if status == 200 and payload.get("protocol_valid") and payload.get("output_valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
