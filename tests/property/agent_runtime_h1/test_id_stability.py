from pathlib import Path

from hypothesis import HealthCheck, given, settings, strategies as st

from agent_runtime.builtin_tools import readonly_catalog
from agent_runtime.compiler import CompileContext, DeterministicCompiler
from agent_runtime.contracts import DecisionAction, ToolIntent
from agent_runtime.tool_adapters import ReadFileAdapter, SearchTextAdapter
from agent_runtime.tool_adapters.scope import AllowlistedPathScope


@given(run_id=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=20), turn_id=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=20))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_compile_ids_are_stable_for_replay(tmp_path: Path, run_id: str, turn_id: str):
    target = tmp_path / "target.txt"
    target.write_text("stable", encoding="utf-8")
    scope = AllowlistedPathScope({"root": tmp_path}, {"file": target})
    catalog = readonly_catalog()
    compiler = DeterministicCompiler(catalog, {"read_file": ReadFileAdapter(scope), "search_text": SearchTextAdapter(scope)})
    intent = ToolIntent(DecisionAction.CALL_TOOL, "read_file", "read", "target", None)
    context = CompileContext(run_id, turn_id, catalog.digest, {"target": "file"})
    assert compiler.compile(intent, context) == compiler.compile(intent, context)
