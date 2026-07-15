from __future__ import annotations

import copy

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
