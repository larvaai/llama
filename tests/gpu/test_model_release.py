from __future__ import annotations

from pathlib import Path
import json
import urllib.request

import pytest

from model_worker.manifest import load_manifest


@pytest.mark.gpu
def test_release_manifest_hashes_and_capabilities(request):
    path=request.config.getoption("--model-manifest")
    if not path: pytest.skip("pass --model-manifest config/model.local.json for the GPU release gate")
    manifest=load_manifest(Path(path))
    assert manifest.raw["sampling"] == {"profile":"greedy-v1"}
    assert manifest.digest.startswith("sha256:")


@pytest.mark.gpu
def test_real_model_inference_is_valid(request):
    url = request.config.getoption("--worker-url")
    if not url:
        if request.config.getoption("--require-gpu"):
            pytest.fail("--worker-url is required for the GPU release gate")
        pytest.skip("pass --worker-url for real-model inference evidence")
    manifest_path = request.config.getoption("--model-manifest")
    manifest = load_manifest(Path(manifest_path))
    schema = {"type":"object","properties":{"result":{"type":"string"}},"required":["result"],"additionalProperties":False}
    body = {"protocol_version":"model-worker.v1","model_id":manifest.id,"messages":[{"role":"user","content":"Return the word ok in result."}],"output_contract":{"version":"structured-output.v1","schema":schema},"limits":{"reasoning_tokens":256,"final_tokens":64,"total_tokens":300,"queue_timeout_ms":5000,"execution_timeout_ms":180000},"stream":{"enabled":False,"include_reasoning":False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/model/generate", json.dumps(body).encode(), {"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=190) as response:
        payload = json.load(response)
    assert payload["protocol_valid"] is True and payload["output_valid"] is True
    assert "accepted" not in payload
