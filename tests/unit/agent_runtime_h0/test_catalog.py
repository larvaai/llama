from __future__ import annotations

from agent_runtime.catalog import ImmutableToolCatalog
from agent_runtime.contracts import EffectClass, FlowPolicy, RetryScope, ToolDefinition, ToolRevision


def tool(version="1"):
    return ToolDefinition(
        "read_file",
        ToolRevision(version, {"type": "object"}, {"type": "object"}, {"type": "object"}, "compiler-1", EffectClass.READ_ONLY, "permission-1", "scope-1", FlowPolicy.REPLAN_WITH_OBSERVATION, 1000, 4096, 512, RetryScope.TRANSIENT),
        "Read an allowlisted artifact",
    )


def test_catalog_digest_is_canonical_and_revision_sensitive():
    first = ImmutableToolCatalog([tool()])
    same = ImmutableToolCatalog([tool()])
    changed = ImmutableToolCatalog([tool("2")])
    assert first.digest == same.digest
    assert first.digest != changed.digest
    assert first.all()[0].tool_id == "read_file"
