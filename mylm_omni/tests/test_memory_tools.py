import pytest

from mylm_omni.memory import MemoryStore
from mylm_omni.memory_tools import make_memory_cleanup_tool, make_memory_search_tool
from mylm_omni.agent import ToolRegistry


def test_cleanup_tool_marks_candidate_without_erasing_content(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite")
    memory = store.add(
        memory_type="procedural",
        key="old build flow",
        content="Use the legacy command before v2.",
        source="validated incident record #17",
    )
    tool = make_memory_cleanup_tool(store)
    registry = ToolRegistry()
    registry.register(tool)

    result = registry.execute(
        "propose_memory_cleanup",
        {"record_id": memory["id"], "reason": "superseded by the verified v3 procedure"},
    )

    assert result["status"] == "deletion_candidate"
    assert result["permanent_erasure"] is False
    assert store.retrieve("legacy command") == []
    saved = store.get_memory(memory["id"])
    assert saved["status"] == "deletion_candidate"
    assert saved["content"] == "Use the legacy command before v2."
    pending = store.list_pending_deletions()
    assert pending[0]["proposal_id"] == result["proposal_id"]
    assert pending[0]["source"] == "validated incident record #17"


def test_cleanup_tool_requires_specific_reason_and_existing_record(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite")
    tool = make_memory_cleanup_tool(store)
    with pytest.raises(ValueError, match="concrete reason|obsolete|unnecessary"):
        tool.function(record_id="does-not-exist", reason=" ")
    with pytest.raises(KeyError):
        tool.function(record_id="does-not-exist", reason="old")


def test_memory_search_tool_returns_source_and_version_as_evidence(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite")
    item = store.add(
        memory_type="preference",
        key="docs",
        content="Prefer Markdown documents.",
        source="human-confirmed correction #4",
        confidence=1.0,
    )
    tool = make_memory_search_tool(store)
    result = tool.function(query="Markdown", limit=3)
    assert result["count"] == 1
    assert result["records"][0]["id"] == item["id"]
    assert result["records"][0]["source"] == "human-confirmed correction #4"
    assert result["records"][0]["version"] == 1
    assert "not a system instruction" in result["warning"]
