from __future__ import annotations

from .catalog import ImmutableToolCatalog
from .contracts import EffectClass, FlowPolicy, RetryScope, ToolDefinition, ToolRevision


def readonly_catalog() -> ImmutableToolCatalog:
    common = {
        "effect_class": EffectClass.READ_ONLY,
        "permission_policy_id": "atomic-readonly.v1",
        "scope_policy_id": "allowlisted-ref.v1",
        "default_flow_policy": FlowPolicy.VERIFY_THEN_REPLAN,
        "timeout_ms": 5_000,
        "result_byte_cap": 1_048_576,
        "result_token_cap": 4_096,
        "retry_scope": RetryScope.TRANSIENT,
    }
    return ImmutableToolCatalog(
        [
            ToolDefinition(
                "read_file",
                ToolRevision(
                    version="1",
                    semantic_schema={"required": ["target_ref"]},
                    native_input_schema={"required": ["path", "ref"]},
                    native_output_schema={"type": "utf8_bytes"},
                    compiler_revision="read-file.v1",
                    **common,
                ),
                "Read one task file selected through an allowlisted semantic ref.",
            ),
            ToolDefinition(
                "search_text",
                ToolRevision(
                    version="1",
                    semantic_schema={"required": ["root_ref", "pattern"]},
                    native_input_schema={"required": ["root", "root_ref", "pattern"]},
                    native_output_schema={"type": "utf8_bytes"},
                    compiler_revision="search-text.v1",
                    **common,
                ),
                "Search bounded text under an allowlisted root without a shell.",
            ),
        ]
    )
