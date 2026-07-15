from __future__ import annotations

import copy

import pytest

from model_worker.context import decode_prompt, preflight_context, prompt_chunks
from model_worker.errors import WorkerError
from model_worker.manifest import enforce_request_envelope, verify_capabilities
from model_worker.preflight import preflight


class Capabilities:
    def __init__(self, mapping=None, context=4096, template=True): self.mapping, self._context, self.template = mapping or {}, context, template
    def tokenize(self, text): return self.mapping.get(text, [ord(char) for char in text])
    def has_chat_template(self): return self.template
    def model_context(self): return self._context


def test_multi_token_markers_and_capabilities(manifest):
    start, end = verify_capabilities(manifest, Capabilities({"<think>": [1, 2], "</think>": [3, 4]}))
    assert start == (1, 2) and end == (3, 4)


def test_bad_capabilities_fail_readiness(manifest):
    with pytest.raises(WorkerError): verify_capabilities(manifest, Capabilities(template=False))
    with pytest.raises(WorkerError): verify_capabilities(manifest, Capabilities({"<think>": [1], "</think>": [1]}))


def test_envelope_and_model_identity(manifest, request_body):
    prepared = preflight(request_body, manifest)
    enforce_request_envelope(prepared.request, manifest)
    bad = copy.deepcopy(request_body); bad["model_id"] = "other"
    with pytest.raises(WorkerError): preflight(bad, manifest)


def test_context_chunking_and_cancellation():
    assert list(prompt_chunks(list(range(9)), 4)) == [[0,1,2,3],[4,5,6,7],[8]]
    assert preflight_context(100, 20, 128, 8) == 0
    with pytest.raises(WorkerError) as error: preflight_context(101, 20, 128, 8)
    assert error.value.code == "context_overflow"
    decoded = []
    with pytest.raises(WorkerError) as cancelled:
        decode_prompt(list(range(7)), 3, decoded.append, lambda: len(decoded) == 1)
    assert cancelled.value.code == "cancelled" and decoded == [[0, 1, 2]]
