"""Explicitly approved file creation for autonomous local work.

This tool intentionally creates new files only. Existing files are never overwritten, and the
ToolSpec is high-risk so the AgentController requires an approval callback on every write.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from mylm_omni.agent import ToolSpec
from mylm_omni.workspace_tools import _resolve_under_roots, _workspace_roots


def make_approved_create_file_tool(
    *,
    workspace_root: str | Path,
    max_file_bytes: int = 32_000,
    allowed_suffixes: set[str] | None = None,
) -> ToolSpec:
    """Return a human-approved, no-overwrite create-file capability scoped to one workspace."""
    roots = _workspace_roots([workspace_root])
    root = roots[0]
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be positive")
    suffixes = allowed_suffixes or {
        ".txt", ".md", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".toml",
        ".py", ".html", ".css", ".js", ".ts", ".sql", ".xml", ".svg",
    }
    suffixes = {suffix.casefold() for suffix in suffixes}
    if any(not suffix.startswith(".") for suffix in suffixes):
        raise ValueError("allowed suffixes must include the leading dot")

    def create_file(path: str, content: str) -> dict[str, Any]:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty relative path")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        relative = Path(path)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise PermissionError("only clean relative paths are permitted")
        if relative.suffix.casefold() not in suffixes:
            raise PermissionError(f"file extension is not allowed: {relative.suffix}")
        target = _resolve_under_roots(root / relative, roots)
        if target == root:
            raise PermissionError("cannot write over the workspace root")
        if not target.parent.is_dir():
            raise FileNotFoundError("parent directory must already exist")
        if target.exists():
            raise FileExistsError("existing files are never overwritten")
        raw = content.encode("utf-8")
        if len(raw) > max_file_bytes:
            raise ValueError(f"content exceeds {max_file_bytes} bytes")
        # Exclusive create prevents overwriting a file that appears after validation.
        with target.open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        return {
            "status": "created",
            "path": str(target.relative_to(root).as_posix()),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    return ToolSpec(
        name="create_workspace_file",
        description=(
            "Create a new text file under the configured workspace. Never overwrites existing "
            "files. Requires per-call human approval before execution."
        ),
        function=create_file,
        requires_approval=True,
        risk_level="high",
        allowed_args=frozenset({"path", "content"}),
    )
