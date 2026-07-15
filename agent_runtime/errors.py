from __future__ import annotations


class AgentRuntimeError(Exception):
    """Fail-closed error carrying a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ContractError(AgentRuntimeError):
    pass


class ConcurrencyConflict(AgentRuntimeError):
    pass


class IntegrityViolation(AgentRuntimeError):
    pass
