"""Safe local workspace tools for the MyLM-omni agent.

These tools are read-only, path-confined and bounded. They do not expose shell, Python eval,
arbitrary network access, credential files, write/delete capabilities or unrestricted recursion.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
from pathlib import Path
from typing import Any

from mylm_omni.agent import ToolSpec

_BLOCKED_NAMES = {
    ".env", ".env.local", ".env.production", ".env.development",
    "id_rsa", "id_ed25519", "credentials.json", "token.json",
    "secrets.json", "secrets.yaml", "secrets.yml",
}
_BLOCKED_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".kdbx"}
_BLOCKED_DIRS = {"secrets", "credentials", "private_keys", "private-keys"}
_IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules",
    "__pycache__", "dist", "build", ".pytest_cache", ".mypy_cache",
}
_TEXT_EXTENSIONS = {
    ".py", ".pyi", ".md", ".rst", ".txt", ".json", ".jsonl", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".csv", ".tsv", ".xml", ".html", ".css", ".js",
    ".ts", ".tsx", ".jsx", ".sql", ".sh", ".ps1", ".bat", ".cmd",
}


def _resolve_under_roots(path: str | Path, roots: list[Path]) -> Path:
    """Resolve symlinks and reject paths outside approved roots."""
    candidate = Path(path).expanduser().resolve()
    if not any(candidate == root or root in candidate.parents for root in roots):
        raise PermissionError("path is outside the configured workspace roots")
    if any(part in _IGNORED_DIRS for part in candidate.parts):
        raise PermissionError("path is within an excluded dependency/build directory")
    relative_parts = []
    for root in roots:
        if candidate == root or root in candidate.parents:
            relative_parts = list(candidate.relative_to(root).parts)
            break
    # Hide all dotfiles/directories by default, including internal memory and audit files.
    if any(part.startswith(".") for part in relative_parts):
        raise PermissionError("hidden files/directories are blocked by default")
    folded_parts = {part.casefold() for part in candidate.parts}
    if (
        candidate.name.casefold() in _BLOCKED_NAMES
        or candidate.name.casefold().startswith(".env")
        or candidate.suffix.casefold() in _BLOCKED_SUFFIXES
        or folded_parts.intersection(_BLOCKED_DIRS)
        or candidate.name.casefold().startswith(("id_rsa", "id_ed25519"))
    ):
        raise PermissionError("sensitive file types/locations are blocked by default")
    return candidate


def _workspace_roots(raw_roots: list[str | Path]) -> list[Path]:
    roots = [Path(root).expanduser().resolve() for root in raw_roots]
    if not roots:
        raise ValueError("at least one workspace root must be configured")
    for root in roots:
        if not root.is_dir():
            raise NotADirectoryError(root)
    return roots


def make_local_workspace_tools(
    *,
    allowed_roots: list[str | Path],
    max_read_bytes: int = 128_000,
    max_list_entries: int = 100,
    max_search_results: int = 30,
) -> list[ToolSpec]:
    """Return a read-only tool set constrained to explicit local directories."""
    if max_read_bytes < 256 or max_list_entries < 1 or max_search_results < 1:
        raise ValueError("tool limits must be positive")
    roots = _workspace_roots(allowed_roots)

    def list_files(path: str, recursive: bool = False, limit: int = 50) -> dict[str, Any]:
        """List file metadata; do not read content and cap traversal depth/entry count."""
        if limit < 1:
            raise ValueError("limit must be positive")
        limit = min(limit, max_list_entries)
        directory = _resolve_under_roots(path, roots)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        entries = []
        iterator = directory.rglob("*") if recursive else directory.iterdir()
        for item in iterator:
            try:
                resolved = item.resolve()
                _resolve_under_roots(resolved, roots)
            except (PermissionError, OSError):
                continue
            if item.is_dir():
                if recursive and item.name in _IGNORED_DIRS:
                    continue
                continue
            try:
                stat = item.stat()
            except OSError:
                continue
            entries.append({
                "path": str(resolved),
                "name": item.name,
                "suffix": item.suffix.lower(),
                "size_bytes": stat.st_size,
                "is_text_candidate": item.suffix.lower() in _TEXT_EXTENSIONS,
            })
            if len(entries) >= limit:
                break
        return {
            "root": str(directory),
            "entries": entries,
            "truncated": len(entries) >= limit,
            "max_entries": limit,
        }

    def read_text(path: str, max_chars: int = 12000) -> dict[str, Any]:
        """Read a bounded text excerpt; sensitive files and binary content are rejected."""
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        max_chars = min(max_chars, max_read_bytes)
        target = _resolve_under_roots(path, roots)
        if not target.is_file():
            raise FileNotFoundError(target)
        size = target.stat().st_size
        if size > max_read_bytes:
            raise ValueError(
                f"file is {size} bytes, above max_read_bytes={max_read_bytes}; "
                "use search_workspace_text for snippets"
            )
        raw = target.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("file is not valid UTF-8 text") from exc
        if "\x00" in text:
            raise ValueError("binary files are not accepted by the text reader")
        return {
            "path": str(target),
            "text": text[:max_chars],
            "characters_returned": min(len(text), max_chars),
            "truncated": len(text) > max_chars,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": size,
        }

    def search_text(path: str, query: str, limit: int = 20) -> dict[str, Any]:
        """Search text files under one approved root and return bounded matching lines."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if len(query) > 512:
            raise ValueError("query is too long")
        if limit < 1:
            raise ValueError("limit must be positive")
        limit = min(limit, max_search_results)
        root = _resolve_under_roots(path, roots)
        if not root.is_dir():
            raise NotADirectoryError(root)
        matches: list[dict[str, Any]] = []
        files_scanned = 0
        q = query.casefold()
        for file_path in root.rglob("*"):
            if len(matches) >= limit:
                break
            if any(part in _IGNORED_DIRS for part in file_path.parts):
                continue
            if not file_path.is_file() or file_path.suffix.lower() not in _TEXT_EXTENSIONS:
                continue
            try:
                file_path = _resolve_under_roots(file_path, roots)
                if file_path.stat().st_size > max_read_bytes:
                    continue
                raw = file_path.read_bytes()
                decoded = raw.decode("utf-8")
                if "\x00" in decoded:
                    continue
            except (PermissionError, OSError, UnicodeDecodeError):
                continue
            files_scanned += 1
            for line_number, line in enumerate(decoded.splitlines(), start=1):
                if q in line.casefold():
                    matches.append({
                        "path": str(file_path),
                        "line": line_number,
                        "snippet": line[:500],
                        "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                    })
                    if len(matches) >= limit:
                        break
        return {
            "root": str(root),
            "query": query,
            "files_scanned": files_scanned,
            "matches": matches,
            "truncated": len(matches) >= limit,
        }

    def calculate(expression: str) -> dict[str, Any]:
        """Evaluate bounded arithmetic only; no eval(), names, attributes, or function calls."""
        if not expression.strip() or len(expression) > 256:
            raise ValueError("expression must be 1–256 characters")
        tree = ast.parse(expression, mode="eval")
        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.UAdd: operator.pos,
            ast.USub: operator.neg,
        }
        node_count = 0

        def visit(node):
            nonlocal node_count
            node_count += 1
            if node_count > 64:
                raise ValueError("expression is too complex")
            if isinstance(node, ast.Expression):
                return visit(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                value = node.value
                if abs(value) > 1e12:
                    raise ValueError("numeric literal is too large")
                return value
            if isinstance(node, ast.UnaryOp) and type(node.op) in operations:
                return operations[type(node.op)](visit(node.operand))
            if isinstance(node, ast.BinOp) and type(node.op) in operations:
                left = visit(node.left)
                right = visit(node.right)
                if isinstance(node.op, ast.Pow):
                    if abs(right) > 12 or abs(left) > 1e6:
                        raise ValueError("power operation exceeds configured limits")
                result = operations[type(node.op)](left, right)
                if isinstance(result, complex):
                    raise ValueError("complex arithmetic results are unsupported")
                if isinstance(result, int):
                    if abs(result) > 10**100:
                        raise ValueError("result is non-finite or too large")
                elif not math.isfinite(result) or abs(result) > 1e100:
                    raise ValueError("result is non-finite or too large")
                return result
            raise ValueError(f"unsupported arithmetic syntax: {type(node).__name__}")

        result = visit(tree)
        return {"expression": expression, "result": result}

    return [
        ToolSpec(
            name="list_workspace_files",
            description="List file metadata within configured workspace roots; read-only and bounded.",
            function=list_files,
            allowed_args=frozenset({"path", "recursive", "limit"}),
        ),
        ToolSpec(
            name="read_workspace_text",
            description="Read a bounded UTF-8 text excerpt from a permitted non-sensitive local file.",
            function=read_text,
            allowed_args=frozenset({"path", "max_chars"}),
        ),
        ToolSpec(
            name="search_workspace_text",
            description="Search matching lines in permitted local text files; returns bounded snippets.",
            function=search_text,
            allowed_args=frozenset({"path", "query", "limit"}),
        ),
        ToolSpec(
            name="calculate_arithmetic",
            description="Evaluate a bounded arithmetic expression without arbitrary code execution.",
            function=calculate,
            allowed_args=frozenset({"expression"}),
        ),
    ]
