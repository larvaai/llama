from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from inference_runtime import (
    CacheScope,
    CacheVisibility,
    ContinuousBatchScheduler,
    InferenceRuntimeError,
    SchedulingMetadata,
    SessionCacheControl,
)
from inference_runtime.adapters import LlamaCppSteppableBackend
from model_worker.manifest import load_manifest
from model_worker.preflight import preflight


class RecordingSink:
    def __init__(self) -> None:
        self.events = []
        self.lock = threading.Lock()

    def publish(self, event) -> None:
        with self.lock:
            self.events.append(event)


def _runtime_options(request):
    model_manifest = request.config.getoption("--model-manifest")
    runtime_manifest = request.config.getoption("--runtime-manifest")
    executable = request.config.getoption("--runtime-executable")
    if not model_manifest or not runtime_manifest or not executable:
        message = (
            "multi-sequence GPU gate requires --model-manifest, "
            "--runtime-manifest and --runtime-executable"
        )
        if request.config.getoption("--require-gpu"):
            pytest.fail(message)
        pytest.skip(message)
    return Path(model_manifest), Path(runtime_manifest), Path(executable)


def _body(model_id: str, expected: str) -> dict:
    return {
        "protocol_version": "model-worker.v1",
        "model_id": model_id,
        "messages": [
            {
                "role": "user",
                "content": f"Return a JSON object whose result field is exactly {expected}.",
            }
        ],
        "output_contract": {
            "version": "structured-output.v1",
            "schema": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
                "additionalProperties": False,
            },
            "instructions": f"The result field must be exactly {expected}.",
        },
        "limits": {
            "reasoning_tokens": 256,
            "final_tokens": 64,
            "total_tokens": 320,
            "queue_timeout_ms": 30000,
            "execution_timeout_ms": 90000,
        },
        "stream": {"enabled": False, "include_reasoning": False},
    }


def _conversation_body(model_id: str, messages: list[dict[str, str]]) -> dict:
    body = _body(model_id, "unused")
    body["messages"] = messages
    body["output_contract"]["instructions"] = (
        "Set result to the exact value requested in the latest user message."
    )
    return body


def _metadata(
    request_id: str,
    *,
    cache_scope=None,
    session_cache=None,
) -> SchedulingMetadata:
    workflow_id = cache_scope.workflow_id if cache_scope is not None else "gpu-workflow"
    agent_id = cache_scope.agent_id if cache_scope is not None else f"agent-{request_id}"
    return SchedulingMetadata(
        request_id,
        workflow_id,
        agent_id,
        "throughput",
        1,
        None,
        cache_scope,
        session_cache,
    )


def _components(request):
    model_path, runtime_path, executable = _runtime_options(request)
    manifest = load_manifest(model_path)
    backend = LlamaCppSteppableBackend(
        executable,
        runtime_path,
        startup_timeout=180,
        command_timeout=180,
    )
    scheduler = ContinuousBatchScheduler(backend, tick_token_budget=512)
    return manifest, backend, scheduler


@pytest.mark.gpu
def test_real_runtime_interleaves_2_4_8_sequences_without_state_leak(request):
    manifest, backend, scheduler = _components(request)
    try:
        for concurrency in (2, 4, 8):
            results = {}
            failures = {}

            def invoke(index):
                expected = f"ok-{concurrency}-{index}"
                try:
                    results[index] = scheduler.infer(
                        preflight(_body(manifest.id, expected), manifest),
                        scheduling=_metadata(f"batch-{concurrency}-{index}"),
                        events=RecordingSink(),
                    )
                except BaseException as exc:
                    failures[index] = exc

            threads = [
                threading.Thread(target=invoke, args=(index,))
                for index in range(concurrency)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(120)

            assert all(not thread.is_alive() for thread in threads)
            assert failures == {}
            assert {
                index: result.output
                for index, result in results.items()
            } == {
                index: {"result": f"ok-{concurrency}-{index}"}
                for index in range(concurrency)
            }
            assert scheduler.active_requests == 0
            assert backend.supervisor_state == "READY"
    finally:
        assert scheduler.shutdown(timeout=20)


@pytest.mark.gpu
def test_real_runtime_cancel_isolated_sequence_and_reuses_capacity(request):
    manifest, backend, scheduler = _components(request)
    results = {}
    failures = {}
    opened = threading.Event()
    decoded = threading.Event()

    class ProgressSink(RecordingSink):
        def publish(self, event):
            super().publish(event)
            if event.kind.value == "sequence_opened":
                opened.set()
            elif event.kind.value == "decode_completed":
                decoded.set()

    def invoke(name, expected, sink):
        try:
            results[name] = scheduler.infer(
                preflight(_body(manifest.id, expected), manifest),
                scheduling=_metadata(name),
                events=sink,
            )
        except BaseException as exc:
            failures[name] = exc

    cancel_sink = ProgressSink()
    cancelled = threading.Thread(
        target=invoke,
        args=("cancelled", "cancelled-result", cancel_sink),
    )
    survivor = threading.Thread(
        target=invoke,
        args=("survivor", "survivor-result", RecordingSink()),
    )
    try:
        cancelled.start()
        survivor.start()
        assert opened.wait(60)
        assert decoded.wait(60)
        cancel_started = time.monotonic()
        assert scheduler.cancel("cancelled")
        cancel_accepted = time.monotonic()
        assert cancel_accepted - cancel_started <= 1
        cancelled.join(120)
        survivor.join(120)
        assert not cancelled.is_alive() and not survivor.is_alive()
        assert isinstance(failures.get("cancelled"), InferenceRuntimeError)
        assert failures["cancelled"].code == "cancelled"
        assert results["survivor"].output == {"result": "survivor-result"}
        after_cancel_decode = [
            event
            for event in cancel_sink.events
            if event.kind.value == "decode_completed"
            and event.at_monotonic > cancel_accepted
        ]
        assert len(after_cancel_decode) <= 1
        release_index = next(
            index
            for index, event in enumerate(cancel_sink.events)
            if event.kind.value == "sequence_released"
        )
        terminal_index = next(
            index
            for index, event in enumerate(cancel_sink.events)
            if event.kind.value == "request_failed"
        )
        assert release_index < terminal_index
        assert (
            cancel_sink.events[terminal_index].at_monotonic - cancel_accepted
        ) <= 5

        reuse = scheduler.infer(
            preflight(_body(manifest.id, "reused-slot"), manifest),
            scheduling=_metadata("reuse"),
            events=RecordingSink(),
        )
        assert reuse.output == {"result": "reused-slot"}
        assert scheduler.active_requests == 0
        assert backend.supervisor_state == "READY"
    finally:
        assert scheduler.shutdown(timeout=20)


@pytest.mark.gpu
def test_real_runtime_cache_hit_is_exact_scoped_and_clearable(request):
    manifest, backend, scheduler = _components(request)
    prepared = preflight(_body(manifest.id, "ok-cache"), manifest)
    private = CacheScope(
        "tenant-a",
        "workflow-a",
        "agent-a",
        CacheVisibility.PRIVATE,
    )
    other_tenant = CacheScope(
        "tenant-b",
        "workflow-a",
        "agent-a",
        CacheVisibility.PRIVATE,
    )
    try:
        backend.clear_cache()
        first = scheduler.infer(
            prepared,
            scheduling=_metadata("cache-first", cache_scope=private),
            events=RecordingSink(),
        )
        second = scheduler.infer(
            prepared,
            scheduling=_metadata("cache-second", cache_scope=private),
            events=RecordingSink(),
        )
        isolated = scheduler.infer(
            prepared,
            scheduling=_metadata("cache-isolated", cache_scope=other_tenant),
            events=RecordingSink(),
        )
        stats = backend.cache_stats()

        assert first.output == second.output == isolated.output == {"result": "ok-cache"}
        assert first.usage["cache_hit"] is False
        assert second.usage["cache_hit"] is True, {
            "first": first.usage,
            "second": second.usage,
            "stats": stats,
        }
        assert second.usage["cached_prompt_tokens"] > 0
        assert isolated.usage["cache_hit"] is False
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["saved_prefill_tokens"] == second.usage["cached_prompt_tokens"]
        assert backend.clear_cache() >= 2
        assert backend.cache_stats()["bytes_used"] == 0
    finally:
        assert scheduler.shutdown(timeout=20)


@pytest.mark.gpu
def test_real_runtime_reuses_longest_exact_token_prefix(request):
    manifest, backend, scheduler = _components(request)
    scope = CacheScope(
        "tenant-prefix",
        "workflow-prefix",
        "agent-prefix",
        CacheVisibility.PRIVATE,
    )
    shared = "Stable shared context: " + ("alpha beta gamma " * 32)
    first_body = _conversation_body(
        manifest.id,
        [{
            "role": "user",
            "content": shared + "The exact result value for this turn is prefix-a.",
        }],
    )
    second_body = _conversation_body(
        manifest.id,
        [{
            "role": "user",
            "content": shared + "The exact result value for this turn is prefix-b.",
        }],
    )
    try:
        backend.clear_cache()
        first = scheduler.infer(
            preflight(first_body, manifest),
            scheduling=_metadata("prefix-first", cache_scope=scope),
            events=RecordingSink(),
        )
        second = scheduler.infer(
            preflight(second_body, manifest),
            scheduling=_metadata("prefix-second", cache_scope=scope),
            events=RecordingSink(),
        )
        stats = backend.cache_stats()

        assert first.output == {"result": "prefix-a"}
        assert second.output == {"result": "prefix-b"}
        assert first.usage["cache_hit"] is False
        assert second.usage["cache_hit"] is True, {
            "first": first.usage,
            "second": second.usage,
            "stats": stats,
        }
        assert second.usage["cache_match"] == "prefix"
        assert second.usage["cached_prompt_tokens"] >= 16
        assert stats["prefix_hits"] == 1
        assert stats["saved_prefill_tokens"] == second.usage["cached_prompt_tokens"]
    finally:
        assert scheduler.shutdown(timeout=20)


@pytest.mark.gpu
def test_real_runtime_session_lineage_is_immutable_and_copy_on_write(request):
    manifest, backend, scheduler = _components(request)
    scope = CacheScope(
        "tenant-session",
        "workflow-session",
        "agent-session",
        CacheVisibility.PRIVATE,
    )
    root_user = {
        "role": "user",
        "content": "The exact result value for this turn is session-root.",
    }
    try:
        backend.clear_cache()
        root = scheduler.infer(
            preflight(_conversation_body(manifest.id, [root_user]), manifest),
            scheduling=_metadata(
                "session-root",
                cache_scope=scope,
                session_cache=SessionCacheControl("session-a"),
            ),
            events=RecordingSink(),
        )
        assert root.output == {"result": "session-root"}
        assert root.timing["session_id"] == "session-a"
        assert root.usage["session_generation"] == 1

        results = {}
        failures = {}

        def branch(name: str) -> None:
            messages = [
                root_user,
                {
                    "role": "assistant",
                    "content": json.dumps(root.output, separators=(",", ":")),
                },
                {
                    "role": "user",
                    "content": f"The exact result value for this turn is {name}.",
                },
            ]
            try:
                results[name] = scheduler.infer(
                    preflight(_conversation_body(manifest.id, messages), manifest),
                    scheduling=_metadata(
                        name,
                        cache_scope=scope,
                        session_cache=SessionCacheControl(
                            "session-a",
                            parent_generation=1,
                        ),
                    ),
                    events=RecordingSink(),
                )
            except BaseException as exc:
                failures[name] = exc

        branches = [
            threading.Thread(target=branch, args=(name,))
            for name in ("session-branch-a", "session-branch-b")
        ]
        for thread in branches:
            thread.start()
        for thread in branches:
            thread.join(120)

        assert all(not thread.is_alive() for thread in branches)
        assert failures == {}, {
            "failures": failures,
            "root_usage": root.usage,
            "root_timing": root.timing,
            "stats": backend.cache_stats(),
        }
        assert {
            name: result.output for name, result in results.items()
        } == {
            "session-branch-a": {"result": "session-branch-a"},
            "session-branch-b": {"result": "session-branch-b"},
        }
        assert all(
            result.usage["cache_match"] == "session"
            and result.usage["cached_prompt_tokens"] > 0
            for result in results.values()
        )
        assert {
            result.usage["session_generation"] for result in results.values()
        } == {2, 3}
        assert any(
            result.usage["session_copy_on_write"] for result in results.values()
        )
        stats = backend.cache_stats()
        assert stats["session_hits"] == 2
        assert stats["session_entries"] == 3
        assert stats["cow_clones"] == 1

        with pytest.raises(InferenceRuntimeError) as captured:
            scheduler.infer(
                preflight(_conversation_body(manifest.id, [root_user]), manifest),
                scheduling=_metadata(
                    "session-stale",
                    cache_scope=scope,
                    session_cache=SessionCacheControl(
                        "session-a",
                        parent_generation=999,
                    ),
                ),
                events=RecordingSink(),
            )
        assert captured.value.code == "session_snapshot_missing"
    finally:
        assert scheduler.shutdown(timeout=20)
