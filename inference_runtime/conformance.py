from __future__ import annotations

from dataclasses import dataclass

from .contracts import BackendCapabilities
from .ports import (
    BackendConformanceError,
    ManagedBackend,
    SteppableBackend,
    require_managed_backend,
    require_steppable_backend,
)
from .registry import BackendMode


@dataclass(frozen=True, slots=True)
class BackendConformanceReport:
    backend: str
    mode: BackendMode
    checks: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return True


def inspect_backend_conformance(
    adapter: object,
    *,
    mode: BackendMode,
) -> BackendConformanceReport:
    """Validate the portable backend boundary without invoking model compute."""

    if type(mode) is not BackendMode:
        raise TypeError("mode must be BackendMode")
    capabilities = getattr(adapter, "capabilities", None)
    if type(capabilities) is not BackendCapabilities:
        raise BackendConformanceError("backend capabilities are not validated")
    checks = ["validated_capabilities"]
    if mode is BackendMode.MANAGED:
        require_managed_backend(adapter)
        if isinstance(adapter, SteppableBackend) or capabilities.supports_sequence_steps:
            raise BackendConformanceError(
                "managed backend must not expose sequence-step control"
            )
        for name in ("open_sequence", "prefill", "decode", "release"):
            if callable(getattr(adapter, name, None)):
                raise BackendConformanceError(
                    f"managed backend leaks steppable method {name}"
                )
        checks.extend(
            (
                "managed_protocol",
                "full_request_capability",
                "no_sequence_api_leak",
            )
        )
    else:
        require_steppable_backend(adapter)
        if isinstance(adapter, ManagedBackend) or capabilities.supports_full_request:
            raise BackendConformanceError(
                "steppable compute backend must not own full-request generation"
            )
        checks.extend(
            (
                "steppable_protocol",
                "sequence_step_capability",
                "explicit_release_boundary",
            )
        )
    return BackendConformanceReport(capabilities.backend, mode, tuple(checks))
