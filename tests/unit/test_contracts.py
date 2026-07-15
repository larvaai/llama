from __future__ import annotations

import copy
from dataclasses import asdict

import pytest

from model_worker.contracts import GenerateRequest, GenerateResult
from model_worker.errors import WorkerError


def test_strict_request_and_no_semantic_acceptance(request_body):
    request = GenerateRequest.parse(request_body)
    assert not hasattr(request, "expected_result")
    result = GenerateResult("r", "a", "completed", True, True, {"status": "blocked"}, {}, {}, {})
    payload = result.as_dict()
    assert "accepted" not in payload and payload["output"]["status"] == "blocked"


@pytest.mark.parametrize("mutation", [
    lambda body: "string",
    lambda body: [],
    lambda body: None,
    lambda body: {**body, "unknown": 1},
    lambda body: {**body, "stream": {"enabled": "false"}},
    lambda body: {**body, "messages": [{"role": "user", "content": "x", "extra": 1}]},
])
def test_invalid_shapes_are_deterministic(request_body, mutation):
    with pytest.raises(WorkerError) as error:
        GenerateRequest.parse(mutation(copy.deepcopy(request_body)))
    assert error.value.code == "invalid_request"


def test_completed_invariant():
    with pytest.raises(ValueError): GenerateResult("r", "a", "completed", False, True, {}, {}, {}, {}).as_dict()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda body: {**body, "protocol_version": "wrong"},
        lambda body: {**body, "model_id": ""},
        lambda body: {**body, "messages": []},
        lambda body: {**body, "messages": [{"role": "tool", "content": "x"}]},
        lambda body: {**body, "messages": [{"role": "user", "content": ""}]},
        lambda body: {**body, "output_contract": {"version": "wrong", "schema": {}}},
        lambda body: {**body, "output_contract": {"version": "structured-output.v1", "schema": {}, "instructions": 1}},
        lambda body: {**body, "limits": {**body["limits"], "final_tokens": True}},
        lambda body: {**body, "limits": {**body["limits"], "total_tokens": 100}},
        lambda body: {**body, "metadata": {"client_request_id": 1}},
    ],
)
def test_each_request_contract_invariant_is_enforced(request_body, mutation):
    with pytest.raises(WorkerError) as captured:
        GenerateRequest.parse(mutation(copy.deepcopy(request_body)))
    assert captured.value.code == "invalid_request"


def test_request_defaults_and_result_serialization(request_body):
    body = copy.deepcopy(request_body)
    body.pop("stream")
    body.pop("metadata")
    request = GenerateRequest.parse(body)
    assert request.stream.enabled is False
    assert request.stream.include_reasoning is False
    assert request.client_request_id is None
    assert asdict(request.messages[0]) == {"role": "user", "content": "Return a structured answer."}

    payload = GenerateResult("r", "a", "failed", False, False, None, {}, {}, {}, {"code": "x"}).as_dict()
    assert payload["protocol_version"] == "model-worker.v1"
    assert payload["termination"] == "failed"
    assert payload["error"] == {"code": "x"}
