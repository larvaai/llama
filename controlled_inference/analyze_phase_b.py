import json
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
gpu = "--gpu" in sys.argv
source = root / "artifacts" / ("phase-b-gpu-tokens.jsonl" if gpu else "phase-b-tokens.jsonl")
target = root / "artifacts" / ("phase-b-gpu-observation.json" if gpu else "phase-b-observation.json")
runtime_log = root / "artifacts" / "phase-b-gpu-runtime.log"
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
tokens = [row for row in rows if row.get("type") == "token"]
boundaries = [row for row in tokens if row.get("is_boundary")]
runtime_text = runtime_log.read_text(encoding="utf-16", errors="ignore") if gpu else ""
if gpu and "offloaded 33/33 layers to GPU" not in runtime_text:
    runtime_text = runtime_log.read_text(encoding="utf-8", errors="ignore")
full_gpu = "offloaded 33/33 layers to GPU" in runtime_text
gpu_buffers = {
    "model_mib": float(m.group(1)) if (m := re.search(r"CUDA0 model buffer size\s*=\s*([0-9.]+) MiB", runtime_text)) else None,
    "kv_mib": float(m.group(1)) if (m := re.search(r"CUDA0 KV buffer size\s*=\s*([0-9.]+) MiB", runtime_text)) else None,
    "compute_mib": float(m.group(1)) if (m := re.search(r"CUDA0 compute buffer size\s*=\s*([0-9.]+) MiB", runtime_text)) else None,
}

result = {
    "phase": "B-direct-C-API-boundary",
    "passed": (not gpu or full_gpu) and len(boundaries) == 1
    and tokens[0]["token_id"] == 248068
    and boundaries[0]["token_id"] == 248069
    and boundaries[0]["phase_before"] == "THINKING"
    and boundaries[0]["phase_after"] == "FINAL",
    "acceptance": {
        "thinking_start_seen": tokens[0]["token_id"] == 248068,
        "thinking_end_seen": any(row["token_id"] == 248069 for row in tokens),
        "exactly_one_boundary": len(boundaries) == 1,
        "state_switched_on_boundary": len(boundaries) == 1
        and boundaries[0]["phase_before"] == "THINKING"
        and boundaries[0]["phase_after"] == "FINAL",
        "final_tokens_exist_after_boundary": bool(boundaries)
        and any(row["index"] > boundaries[0]["index"] for row in tokens),
        "full_gpu_offload": full_gpu if gpu else None,
    },
    "boundary": boundaries[0] if len(boundaries) == 1 else boundaries,
    "tokens_around_boundary": tokens[max(0, boundaries[0]["index"] - 4):boundaries[0]["index"] + 5] if boundaries else [],
    "summary": next((row for row in rows if row.get("type") == "summary"), None),
    "gpu_buffers_mib": gpu_buffers if gpu else None,
    "execution_note": "Official llama.cpp b10012, n_gpu_layers=99, native exit code 0." if gpu else "Legacy CPU diagnostic run.",
}
target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
