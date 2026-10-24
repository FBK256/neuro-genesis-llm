import pytest

from mylm_omni.workspace_tools import make_local_workspace_tools


def tools_by_name(root):
    return {
        spec.name: spec
        for spec in make_local_workspace_tools(allowed_roots=[root])
    }


def test_read_text_and_search_within_workspace(tmp_path):
    (tmp_path / "src").mkdir()
    file_path = tmp_path / "src" / "example.py"
    file_path.write_text("alpha = 4\nbeta = alpha + 2\n", encoding="utf-8")
    tools = tools_by_name(tmp_path)
    listing = tools["list_workspace_files"].function(path=str(tmp_path / "src"))
    assert listing["entries"][0]["name"] == "example.py"
    read = tools["read_workspace_text"].function(path=str(file_path))
    assert "alpha = 4" in read["text"]
    assert len(read["sha256"]) == 64
    search = tools["search_workspace_text"].function(
        path=str(tmp_path), query="beta ="
    )
    assert search["matches"][0]["line"] == 2


def test_workspace_tools_reject_path_traversal_and_sensitive_files(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    private_file = tmp_path / "private.txt"
    private_file.write_text("not in workspace", encoding="utf-8")
    env_file = root / ".env.production"
    env_file.write_text("SECRET=value", encoding="utf-8")
    tools = tools_by_name(root)
    with pytest.raises(PermissionError):
        tools["read_workspace_text"].function(path=str(private_file))
    with pytest.raises(PermissionError):
        tools["read_workspace_text"].function(path=str(env_file))


def test_arithmetic_tool_rejects_code_execution_and_bounds_exponents(tmp_path):
    calculator = tools_by_name(tmp_path)["calculate_arithmetic"].function
    assert calculator("2 * (3 + 4)")["result"] == 14
    assert calculator("2 ** 5")["result"] == 32
    with pytest.raises(ValueError):
        calculator("__import__('os').system('echo unsafe')")
    with pytest.raises(ValueError):
        calculator("999999 ** 13")


def test_workspace_read_is_bounded_and_rejects_binary(tmp_path):
    text_file = tmp_path / "large.txt"
    text_file.write_text("a" * 700, encoding="utf-8")
    binary_file = tmp_path / "binary.bin"
    binary_file.write_bytes(b"a\x00b")
    tools = make_local_workspace_tools(
        allowed_roots=[tmp_path], max_read_bytes=512, max_list_entries=10
    )
    read = next(t for t in tools if t.name == "read_workspace_text")
    with pytest.raises(ValueError, match="above max_read_bytes"):
        read.function(path=str(text_file), max_chars=300)
    with pytest.raises(ValueError, match="not valid UTF-8|binary"):
        read.function(path=str(binary_file))



def test_workspace_tools_block_hidden_files_and_hidden_directories(tmp_path):
    hidden_file = tmp_path / ".mylm-omni-agent-audit.jsonl"
    hidden_file.write_text('{"event":"metadata"}', encoding="utf-8")
    hidden_dir = tmp_path / ".private"
    hidden_dir.mkdir()
    (hidden_dir / "notes.md").write_text("private content", encoding="utf-8")

    tools = {
        tool.name: tool
        for tool in make_local_workspace_tools(allowed_roots=[tmp_path])
    }
    read = tools["read_workspace_text"].function
    search = tools["search_workspace_text"].function
    with pytest.raises(PermissionError, match="hidden"):
        read(path=str(hidden_file))
    with pytest.raises(PermissionError, match="hidden"):
        read(path=str(hidden_dir / "notes.md"))
    assert search(path=str(tmp_path), query="private content")["matches"] == []
