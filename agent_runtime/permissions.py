from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AbstractSet

from .contracts import (
    AuthorizationDecision,
    AuthorizationResult,
    CompiledToolCall,
    EffectClass,
)


@dataclass(frozen=True, slots=True)
class PermissionContext:
    role: str
    tenant_id: str
    allowed_tools: AbstractSet[str]
    catalog_digest: str
    expected_catalog_digest: str
    deadline_utc: str | None = None
    tainted: bool = False


class DeterministicPermissionGate:
    policy_id = "atomic-readonly.v1"

    def authorize(self, call: CompiledToolCall, context: PermissionContext) -> AuthorizationResult:
        if context.catalog_digest != context.expected_catalog_digest:
            return AuthorizationResult(AuthorizationDecision.DENY, "catalog changed", self.policy_id)
        if context.deadline_utc is not None:
            deadline = datetime.fromisoformat(context.deadline_utc.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) >= deadline:
                return AuthorizationResult(AuthorizationDecision.DENY, "deadline expired", self.policy_id)
        if context.tainted:
            return AuthorizationResult(AuthorizationDecision.DENY, "tainted arguments", self.policy_id)
        if call.tool_id not in context.allowed_tools:
            return AuthorizationResult(AuthorizationDecision.DENY, "tool denied by scope", self.policy_id)
        if call.effect_class is not EffectClass.READ_ONLY:
            return AuthorizationResult(
                AuthorizationDecision.REQUIRE_APPROVAL,
                "mutation requires bound one-shot approval",
                self.policy_id,
            )
        return AuthorizationResult(AuthorizationDecision.ALLOW, "read-only scope allowed", self.policy_id)
