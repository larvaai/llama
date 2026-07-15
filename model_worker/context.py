from __future__ import annotations

from collections.abc import Callable, Iterable

from .errors import WorkerError


def preflight_context(prompt_tokens: int, reserved_generation_tokens: int, n_ctx: int, safety_margin: int = 8) -> int:
    if any(type(value) is not int or value < 0 for value in (prompt_tokens, reserved_generation_tokens, n_ctx, safety_margin)):
        raise WorkerError("invalid_request", "context inputs must be non-negative integers")
    headroom = n_ctx - prompt_tokens - reserved_generation_tokens - safety_margin
    if headroom < 0:
        raise WorkerError("context_overflow", "prompt plus generation reserve exceeds context")
    return headroom


def prompt_chunks(tokens: list[int], n_batch: int) -> Iterable[list[int]]:
    if type(n_batch) is not int or n_batch <= 0:
        raise ValueError("n_batch must be positive")
    for start in range(0, len(tokens), n_batch):
        yield tokens[start : start + n_batch]


def decode_prompt(tokens: list[int], n_batch: int, decode: Callable[[list[int]], None], should_stop: Callable[[], bool]) -> None:
    for chunk in prompt_chunks(tokens, n_batch):
        if should_stop():
            raise WorkerError("cancelled", "request cancelled during prompt ingestion")
        decode(chunk)
