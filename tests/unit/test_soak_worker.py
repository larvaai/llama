from __future__ import annotations

import pytest

from scripts.soak_worker import (
    IDENTITY_FIELDS,
    extract_runtime_identity,
    resolve_generation_limits,
)


def test_soak_generation_limits_use_safe_manifest_reasoning_budget():
    limits = {
        "max_reasoning_tokens": 1024,
        "max_final_tokens": 512,
        "max_total_tokens": 1536,
    }
    assert resolve_generation_limits(limits) == {
        "reasoning_tokens": 1024,
        "final_tokens": 64,
        "total_tokens": 1088,
    }
    assert resolve_generation_limits(limits, 512) == {
        "reasoning_tokens": 512,
        "final_tokens": 64,
        "total_tokens": 576,
    }


@pytest.mark.parametrize(
    ("limits", "reasoning_tokens", "message"),
    [
        ({}, None, "max_reasoning_tokens"),
        (
            {
                "max_reasoning_tokens": True,
                "max_final_tokens": 64,
                "max_total_tokens": 128,
            },
            None,
            "max_reasoning_tokens",
        ),
        (
            {
                "max_reasoning_tokens": 1024,
                "max_final_tokens": 64,
                "max_total_tokens": 1,
            },
            None,
            "reserve reasoning and final output",
        ),
        (
            {
                "max_reasoning_tokens": 1024,
                "max_final_tokens": 64,
                "max_total_tokens": 1000,
            },
            937,
            "exceeds the manifest envelope",
        ),
        (
            {
                "max_reasoning_tokens": 1024,
                "max_final_tokens": 64,
                "max_total_tokens": 1536,
            },
            True,
            "positive integer",
        ),
    ],
)
def test_soak_generation_limits_reject_invalid_or_unsafe_budget(
    limits,
    reasoning_tokens,
    message,
):
    with pytest.raises(ValueError, match=message):
        resolve_generation_limits(limits, reasoning_tokens)


def test_failed_payload_does_not_create_null_runtime_identity():
    complete = {name: f"value-{name}" for name in IDENTITY_FIELDS}
    assert extract_runtime_identity({"model": complete}) == tuple(
        complete[name] for name in IDENTITY_FIELDS
    )
    assert extract_runtime_identity({"error": {"code": "decode_failed"}}) is None
    assert extract_runtime_identity({"model": {name: None for name in IDENTITY_FIELDS}}) is None
    assert extract_runtime_identity({"model": {"revision": "only-one-field"}}) is None
    assert extract_runtime_identity([]) is None
