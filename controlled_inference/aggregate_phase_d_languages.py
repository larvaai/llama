import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
artifacts = root / "artifacts"
case_ids = ["en", "vi", "auto_en", "auto_vi"]
cases = [
    json.loads((artifacts / f"phase-d-{case_id}-observation.json").read_text(encoding="utf-8"))
    for case_id in case_ids
]
result = {
    "phase": "D-language-matrix",
    "passed": all(case["passed"] for case in cases),
    "cases": [
        {
            "id": case["case"]["id"],
            "response_language": case["case"]["response_language"],
            "task_language": case["case"]["task_language"],
            "expected_response_language": case["expected_response_language"],
            "passed": case["passed"],
            "evidence": case["final"]["evidence"],
        }
        for case in cases
    ],
}
(artifacts / "phase-d-language-matrix.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
sys.exit(0 if result["passed"] else 1)
