from __future__ import annotations

import json

import pytest

from inference_runtime import (
    CacheScope,
    CacheVisibility,
    DecodeStatus,
    PrefillStatus,
    ReleaseStatus,
    SchedulingMetadata,
    SessionCacheControl,
    SequenceStep,
    require_batch_steppable_backend,
)
from inference_runtime.adapters import BackendCommandError, LlamaCppSteppableBackend
from model_worker.preflight import preflight


class NullSink:
    def publish(self, event) -> None:
        pass


class ScriptedBackend(LlamaCppSteppableBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.commands = []
        self.responses = []

    def _command(self, command_type, **payload):
        self.commands.append((command_type, payload))
        return self.responses.pop(0)


def make_backend(manifest, tmp_path):
    executable = tmp_path / "runtime.exe"
    executable.write_bytes(b"test-runtime")
    data = {
        "runtime_manifest_version": "inference-runtime.v1",
        "backend_id": "llama-cpp-test",
        "model_manifest": str(manifest.path),
        "model_manifest_digest": manifest.digest,
        "scheduler": {
            "max_sequences": 4,
            "cpu_threads": 4,
            "kv_tokens": 4096,
            "prefill_chunk_tokens": 128,
            "max_decode_batch": 4,
            "decode_quantum_tokens": 8,
            "tick_token_budget": 256,
        },
        "cache": {
            "enabled": True,
            "byte_budget": 1024,
            "max_entries": 4,
            "ttl_seconds": 60,
        },
    }
    runtime_manifest = tmp_path / "runtime.json"
    runtime_manifest.write_text(json.dumps(data), encoding="utf-8")
    return ScriptedBackend(
        executable,
        runtime_manifest,
        verify_model_files=False,
    )


def metadata(request_id):
    return SchedulingMetadata(
        request_id=request_id,
        workflow_id="workflow",
        agent_id="agent",
        service_class="throughput",
        weight=1,
        deadline_monotonic=None,
    )


def frame(frame_type, command_id, **payload):
    return {
        "protocol_version": "inference-runtime-ipc.v1",
        "type": frame_type,
        "command_id": command_id,
        **payload,
    }


def native_handle(sequence, generation=1):
    return {
        "backend": "llama-cpp-test",
        "model": "qwen35-9b-local",
        "sequence": sequence,
        "generation": generation,
    }


def ready_frame(backend):
    scheduler = backend.runtime_manifest.scheduler
    return {
        "protocol_version": "inference-runtime-ipc.v1",
        "type": "ready",
        "sequence": 0,
        "backend_id": backend.runtime_manifest.backend_id,
        "model_id": backend.runtime_manifest.model_manifest.id,
        "model_manifest_digest": backend.runtime_manifest.model_manifest.digest,
        "max_sequences": scheduler.max_sequences,
        "cpu_threads": scheduler.cpu_threads,
        "kv_tokens": scheduler.kv_tokens,
        "sequence_context_tokens": backend.capabilities.max_context_tokens,
        "prefill_chunk_tokens": scheduler.prefill_chunk_tokens,
        "max_decode_batch": scheduler.max_decode_batch,
        "decode_quantum_tokens": scheduler.decode_quantum_tokens,
        "tick_token_budget": scheduler.tick_token_budget,
        "cache": {
            "enabled": backend.runtime_manifest.cache.enabled,
            "byte_budget": backend.runtime_manifest.cache.byte_budget,
            "max_entries": backend.runtime_manifest.cache.max_entries,
            "ttl_seconds": backend.runtime_manifest.cache.ttl_seconds,
        },
    }


def test_llama_adapter_verifies_actual_per_sequence_context(
    manifest,
    tmp_path,
):
    backend = make_backend(manifest, tmp_path)
    assert backend.capabilities.max_context_tokens == 1024
    backend._verify_ready(ready_frame(backend))

    invalid = ready_frame(backend)
    invalid["sequence_context_tokens"] = 4096
    with pytest.raises(BackendCommandError, match="invalid runtime ready frame"):
        backend._verify_ready(invalid)


def test_llama_adapter_batches_prefill_decode_and_releases_exact_handles(
    manifest,
    request_body,
    tmp_path,
):
    backend = make_backend(manifest, tmp_path)
    assert require_batch_steppable_backend(backend) is backend
    sink = NullSink()
    prepared = preflight(request_body, manifest)
    handles = []
    for index in range(2):
        raw = native_handle(f"slot-{index}")
        backend.responses.append(
            frame(
                "sequence_opened",
                index + 1,
                handle=raw,
                prompt_tokens=10,
                reserved_tokens=30,
                cache_hit=False,
                cached_prompt_tokens=0,
            )
        )
        handles.append(
            backend.open_sequence(
                prepared,
                scheduling=metadata(f"request-{index}"),
                events=sink,
            )
        )
        assert backend.reservation_tokens(handles[-1]) == 30

    backend.responses.append(
        frame(
            "prefill_completed",
            3,
            outcomes=[
                {
                    "handle": native_handle(f"slot-{index}"),
                    "status": "ready",
                    "processed_tokens": 10,
                    "remaining_tokens": 0,
                }
                for index in range(2)
            ],
        )
    )
    prefilled = backend.prefill_batch(
        tuple(SequenceStep(handle, 10) for handle in handles),
        events=sink,
    )
    assert all(outcome.status is PrefillStatus.READY for outcome in prefilled)

    backend.responses.append(
        frame(
            "decode_completed",
            4,
            outcomes=[
                {
                    "handle": native_handle(f"slot-{index}"),
                    "status": "finished" if index == 0 else "failed",
                    "token_ids": [7],
                    "text_delta": "{}" if index == 0 else "",
                    **(
                        {
                            "finish_reason": "stop",
                            "completion": {
                                "final_text": "{}",
                                "prompt_tokens": 10,
                                "reasoning_tokens": 1,
                                "final_tokens": 1,
                                "sampled_tokens": 3,
                                "prompt_decode_ms": 2.0,
                                "generation_ms": 3.0,
                                "first_sample_ms": 1.5,
                                "first_final_ms": 2.5,
                                "sample_itl_ms": [1.0, 1.25],
                                "final_itl_ms": [],
                            },
                        }
                        if index == 0
                        else {"error_code": "protocol_violation:test"}
                    ),
                }
                for index in range(2)
            ],
        )
    )
    decoded = backend.decode_batch(
        tuple(SequenceStep(handle, 1) for handle in handles),
        events=sink,
    )
    assert decoded[0].status is DecodeStatus.FINISHED
    assert decoded[0].completion.final_text == "{}"
    assert decoded[0].completion.sample_itl_ms == (1.0, 1.25)
    assert decoded[1].status is DecodeStatus.FAILED
    assert decoded[1].error_code == "protocol_violation:test"

    for index, handle in enumerate(handles):
        backend.responses.append(
            frame(
                "sequence_released",
                5 + index,
                handle=native_handle(f"slot-{index}"),
                status="released",
                released_bytes=4096,
            )
        )
        released = backend.release(handle, events=sink)
        assert released.status is ReleaseStatus.RELEASED
        assert released.released_bytes == 4096
        assert backend.release(handle, events=sink).status is ReleaseStatus.ALREADY_RELEASED

    assert [name for name, _ in backend.commands] == [
        "open_sequence",
        "open_sequence",
        "prefill_batch",
        "decode_batch",
        "release_sequence",
        "release_sequence",
    ]


def test_llama_adapter_serializes_private_session_lineage(
    manifest,
    request_body,
    tmp_path,
):
    backend = make_backend(manifest, tmp_path)
    prepared = preflight(request_body, manifest)
    scope = CacheScope(
        "tenant-a",
        "workflow-a",
        "agent-a",
        CacheVisibility.PRIVATE,
    )
    scheduling = SchedulingMetadata(
        "session-request",
        "workflow-a",
        "agent-a",
        "throughput",
        1,
        None,
        scope,
        SessionCacheControl("session-a", parent_generation=4, commit=True),
    )
    backend.responses.append(
        frame(
            "sequence_opened",
            1,
            handle=native_handle("slot-0"),
            prompt_tokens=10,
            reserved_tokens=30,
            cache_hit=True,
            cached_prompt_tokens=8,
        )
    )

    backend.open_sequence(prepared, scheduling=scheduling, events=NullSink())

    assert backend.capabilities.supports_session_cache is True
    command, payload = backend.commands[0]
    assert command == "open_sequence"
    assert payload["request"]["session_cache"] == {
        "session_id": "session-a",
        "parent_generation": 4,
        "commit": True,
    }
    assert payload["request"]["cache_scope"]["visibility"] == "private"


def test_llama_adapter_validates_cache_control_frames(manifest, tmp_path):
    backend = make_backend(manifest, tmp_path)
    backend.responses.append(
        frame(
            "cache_stats",
            1,
            cache={
                "enabled": True,
                "entries": 1,
                "bytes_used": 256,
                "byte_budget": 1024,
                "hits": 2,
                "exact_hits": 1,
                "prefix_hits": 1,
                "session_hits": 0,
                "misses": 1,
                "session_misses": 0,
                "insertions": 1,
                "session_insertions": 0,
                "session_entries": 0,
                "cow_clones": 0,
                "evictions": 0,
                "restore_failures": 0,
                "store_failures": 0,
                "saved_prefill_tokens": 42,
            },
        )
    )
    assert backend.cache_stats()["saved_prefill_tokens"] == 42

    backend.responses.append(frame("cache_cleared", 2, removed_entries=1))
    assert backend.clear_cache() == 1
    assert [name for name, _ in backend.commands] == [
        "cache_stats",
        "cache_clear",
    ]
