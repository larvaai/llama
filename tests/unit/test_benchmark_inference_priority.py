from scripts.benchmark_inference_priority import _body


def test_priority_benchmark_prompt_padding_changes_only_prompt_context():
    plain = _body("model", "priority-result")
    padded = _body(
        "model",
        "priority-result",
        prompt_padding="Context item. " * 16,
    )

    assert plain["output_contract"] == padded["output_contract"]
    assert plain["limits"] == padded["limits"]
    assert "Additional context:" not in plain["messages"][0]["content"]
    assert "Additional context:" in padded["messages"][0]["content"]
    assert padded["messages"][0]["content"].endswith("Context item. " * 16)
