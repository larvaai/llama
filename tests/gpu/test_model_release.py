from __future__ import annotations

from pathlib import Path
import json
import urllib.error
import urllib.request

import pytest

from model_worker.manifest import load_manifest


@pytest.mark.gpu
def test_release_manifest_hashes_and_sampling_profile(request):
    path=request.config.getoption("--model-manifest")
    if not path: pytest.skip("pass --model-manifest config/model.local.json for the GPU release gate")
    manifest=load_manifest(Path(path))
    assert manifest.raw["sampling"] == {"profile":"greedy-v1"}
    assert manifest.digest.startswith("sha256:")


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Count the labels that start with A, ignoring case: Alpha, beta, atlas, Gamma.", 2),
        ("Đếm các số chẵn trong danh sách: 1, 2, 4, 7, 8.", 3),
    ],
)
def test_real_model_inference_is_protocol_schema_and_semantically_valid(request, prompt, expected):
    url = request.config.getoption("--worker-url")
    if not url:
        if request.config.getoption("--require-gpu"):
            pytest.fail("--worker-url is required for the GPU release gate")
        pytest.skip("pass --worker-url for real-model inference evidence")
    manifest_path = request.config.getoption("--model-manifest")
    manifest = load_manifest(Path(manifest_path))
    schema = {"type":"object","properties":{"result":{"type":"integer"}},"required":["result"],"additionalProperties":False}
    body = {"protocol_version":"model-worker.v1","model_id":manifest.id,"messages":[{"role":"user","content":prompt}],"output_contract":{"version":"structured-output.v1","schema":schema,"instructions":"The result field is the integer count requested by the user."},"limits":{"reasoning_tokens":256,"final_tokens":64,"total_tokens":300,"queue_timeout_ms":5000,"execution_timeout_ms":180000},"stream":{"enabled":False,"include_reasoning":False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/model/generate", json.dumps(body, ensure_ascii=False).encode(), {"Content-Type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=190) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        pytest.fail(f"worker returned HTTP {exc.code}: {json.load(exc)}")
    assert payload["termination"] == "completed"
    assert payload["protocol_valid"] is True and payload["output_valid"] is True
    assert payload["output"] == {"result": expected}
    assert payload["error"] is None
    assert "accepted" not in payload
