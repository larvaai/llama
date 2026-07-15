import json
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
tokens_path = root / "artifacts" / "phase-c-gpu-tokens.jsonl"
runtime_path = root / "artifacts" / "phase-c-gpu-runtime.log"
output_path = root / "artifacts" / "phase-c-observation.json"

rows = [json.loads(line) for line in tokens_path.read_text(encoding="utf-8").splitlines()]
tokens = [row for row in rows if row.get("type") == "token"]
boundaries = [row for row in tokens if row.get("is_boundary")]
switches = [row for row in rows if row.get("type") == "sampler_switch"]
summary = next(row for row in rows if row.get("type") == "summary")
final = json.loads(summary["final_text"])
runtime = runtime_path.read_text(encoding="utf-16", errors="ignore")
if "offloaded 33/33 layers to GPU" not in runtime:
    runtime = runtime_path.read_text(encoding="utf-8", errors="ignore")

boundary_index = boundaries[0]["index"] if len(boundaries) == 1 else None
acceptance = {
    "full_gpu_offload": "offloaded 33/33 layers to GPU" in runtime,
    "exactly_one_thinking_boundary": len(boundaries) == 1,
    "exactly_one_sampler_switch": len(switches) == 1,
    "grammar_inactive_during_thinking": boundary_index is not None and all(
        not row["grammar_active"] for row in tokens if row["index"] <= boundary_index
    ),
    "grammar_active_for_every_final_token": boundary_index is not None and all(
        row["grammar_active"] for row in tokens if row["index"] > boundary_index
    ),
    "final_is_json_object": isinstance(final, dict),
    "final_has_exact_keys": list(final) == ["result"],
    "result_is_integer": type(final.get("result")) is int,
    "acceptance_result_is_2": final.get("result") == 2,
    "no_markdown_or_extra_text": summary["final_text"] == json.dumps(final, separators=(",", ":")),
    "native_generation_completed": summary["final_phase"] == "FINAL" and summary["grammar_activated"],
}

result = {
    "phase": "C-stateful-fixed-grammar",
    "passed": all(acceptance.values()),
    "acceptance": acceptance,
    "boundary": boundaries[0] if len(boundaries) == 1 else boundaries,
    "sampler_switch": switches,
    "final_text": summary["final_text"],
    "sampled_tokens": summary["sampled_tokens"],
    "gpu": {
        "layers": "33/33",
        "model_buffer_mib": float(m.group(1)) if (m := re.search(r"CUDA0 model buffer size\s*=\s*([0-9.]+) MiB", runtime)) else None,
    },
}
output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
sys.exit(0 if result["passed"] else 1)
