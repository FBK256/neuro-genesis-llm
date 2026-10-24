import pytest

from mylm_omni.agent import ToolRegistry
from mylm_omni.file_creation import make_approved_create_file_tool


def test_file_creation_requires_explicit_approval_and_never_overwrites(tmp_path):
    tool = make_approved_create_file_tool(workspace_root=tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    args = {"path": "notes.md", "content": "# Approved content\n"}

    with pytest.raises(PermissionError, match="human approval"):
        registry.execute(tool.name, args)
    assert not (tmp_path / "notes.md").exists()

    with pytest.raises(PermissionError, match="human approval"):
        registry.execute(tool.name, args, approve=lambda spec, values: False)
    assert not (tmp_path / "notes.md").exists()

    created = registry.execute(tool.name, args, approve=lambda spec, values: True)
    assert created["status"] == "created"
    assert (tmp_path / "notes.md").read_text(encoding="utf-8") == args["content"]

    with pytest.raises(FileExistsError):
        registry.execute(tool.name, args, approve=lambda spec, values: True)


def test_file_creation_rejects_escape_hidden_secret_and_unsupported_extension(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    tool = make_approved_create_file_tool(workspace_root=root)
    with pytest.raises(PermissionError):
        tool.function(path="../outside.md", content="no")
    with pytest.raises(PermissionError):
        tool.function(path=".env", content="SECRET=value")
    with pytest.raises(PermissionError):
        tool.function(path="program.exe", content="not allowed")


def test_file_creation_requires_existing_parent_and_caps_size(tmp_path):
    tool = make_approved_create_file_tool(workspace_root=tmp_path, max_file_bytes=10)
    with pytest.raises(FileNotFoundError):
        tool.function(path="missing/note.md", content="short")
    with pytest.raises(ValueError, match="exceeds"):
        tool.function(path="long.md", content="x" * 11)
