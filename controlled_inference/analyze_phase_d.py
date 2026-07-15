import json
import re
import sys
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("case", nargs="?", default="vi")
args = parser.parse_args()
case_id = args.case
root = Path(__file__).resolve().parent
artifacts = root / "artifacts"
schema = json.loads((root / "phase_d_schema.json").read_text(encoding="utf-8"))
case = json.loads((root / "phase_d_cases" / f"{case_id}.json").read_text(encoding="utf-8"))
prefix = f"phase-d-{case_id}"
rows = [json.loads(line) for line in (artifacts / f"{prefix}-tokens.jsonl").read_text(encoding="utf-8").splitlines()]
runtime = (artifacts / f"{prefix}-runtime.log").read_text(encoding="utf-16", errors="ignore")
if "offloaded 33/33 layers to GPU" not in runtime:
    runtime = (artifacts / f"{prefix}-runtime.log").read_text(encoding="utf-8", errors="ignore")

tokens = [row for row in rows if row.get("type") == "token"]
boundaries = [row for row in tokens if row.get("is_boundary")]
switches = [row for row in rows if row.get("type") == "sampler_switch"]
summary = next(row for row in rows if row.get("type") == "summary")
final = json.loads(summary["final_text"])
boundary_index = boundaries[0]["index"] if len(boundaries) == 1 else None


def matches_type(value, declared):
    types = [declared] if isinstance(declared, str) else declared
    checks = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: type(item) is int,
        "boolean": lambda item: type(item) is bool,
        "null": lambda item: item is None,
    }
    return any(checks[item](value) for item in types)


properties = schema["properties"]
schema_valid = (
    isinstance(final, dict)
    and list(final) == schema["required"] == list(properties)
    and all(matches_type(final[name], spec["type"]) for name, spec in properties.items())
    and all("enum" not in spec or final[name] in spec["enum"] for name, spec in properties.items())
)

vietnamese_letters = set("ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
expected_language = case["task_language"] if case["response_language"] == "auto" else case["response_language"]
has_vietnamese = any(character in vietnamese_letters for character in final.get("evidence", ""))
language_matches = has_vietnamese if expected_language == "vi" else not has_vietnamese

acceptance = {
    "full_gpu_offload": "offloaded 33/33 layers to GPU" in runtime,
    "exactly_one_boundary": len(boundaries) == 1,
    "exactly_one_sampler_switch": len(switches) == 1,
    "grammar_inactive_through_boundary": boundary_index is not None and all(
        not token["grammar_active"] for token in tokens if token["index"] <= boundary_index
    ),
    "grammar_active_for_all_final_tokens": boundary_index is not None and all(
        token["grammar_active"] for token in tokens if token["index"] > boundary_index
    ),
    "token_id_is_canonical_log_field": all(type(token.get("token_id")) is int for token in tokens),
    "piece_display_is_debug_only": all("piece_display" in token for token in tokens),
    "final_utf8_valid_after_join": summary.get("final_utf8_valid") is True
    and summary["final_text"].encode("utf-8").decode("utf-8") == summary["final_text"],
    "final_matches_schema_subset": schema_valid,
    "no_additional_properties": set(final) == set(properties),
    "status_enum_enforced": final.get("status") in ["completed", "blocked"],
    "task_acceptance_result": final.get("result") == case["expected_result"],
    "reason_is_null": final.get("reason") is None,
    "response_language_matches": language_matches,
    "native_generation_completed": summary["final_phase"] == "FINAL" and summary["grammar_activated"],
}

result = {
    "phase": "D-json-schema-subset-to-gbnf",
    "case": case,
    "expected_response_language": expected_language,
    "passed": all(acceptance.values()),
    "acceptance": acceptance,
    "boundary": boundaries[0] if len(boundaries) == 1 else boundaries,
    "sampler_switch": switches,
    "final": final,
    "final_text": summary["final_text"],
    "sampled_tokens": summary["sampled_tokens"],
    "gpu": {
        "layers": "33/33",
        "model_buffer_mib": float(m.group(1)) if (m := re.search(r"CUDA0 model buffer size\s*=\s*([0-9.]+) MiB", runtime)) else None,
    },
    "compiler_scope": {
        "root": "flat object",
        "all_properties_required": True,
        "property_order_fixed": True,
        "types": ["string", "integer", "boolean", "null"],
        "enum": True,
        "nullable_union": True,
        "additionalProperties": False,
    },
}
(artifacts / f"{prefix}-observation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
sys.exit(0 if result["passed"] else 1)
