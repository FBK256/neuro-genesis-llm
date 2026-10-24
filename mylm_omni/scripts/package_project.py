"""Create a clean distributable ZIP of the MyLM-omni source scaffold.

Only approved project source/config/docs/test paths are included. Downloaded datasets, model
weights, run outputs, credentials, hidden runtime memory, audit logs and JSONL training rows are
excluded by default. The resulting ZIP contains no pretrained or trained model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT_FILES = {
    "README.md",
    "pyproject.toml",
}
ALLOWED_ROOT_DIRS = {
    ".github",
    "configs",
    "docs",
    "scripts",
    "src",
    "tests",
}
ALLOWED_DATA_FILES = {
    "dataset_registry.example.json",
}
EXCLUDED_DIR_NAMES = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "build", "dist", "runs", "run", "checkpoints", "checkpoint",
    "models", "model", "weights", "tokenizer", "tokens", "outputs",
    "artifacts", "wandb", "logs", "cache", "caches",
    "private", "secrets", "credentials",
}
EXCLUDED_FILE_NAMES = {
    ".env", ".env.local", ".env.development", ".env.production",
    "id_rsa", "id_ed25519", "credentials.json", "token.json",
    "secrets.json", "secrets.yaml", "secrets.yml",
}
EXCLUDED_SUFFIXES = {
    ".gguf", ".safetensors", ".pt", ".pth", ".ckpt", ".bin",
    ".onnx", ".sqlite", ".sqlite3", ".db", ".log",
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff",
    ".jsonl", ".parquet", ".arrow", ".npz", ".npy",
}
EXCLUDED_NAME_FRAGMENTS = (
    "audit", "private-key", "private_key", "secret", "credential",
    "wandb", "training-state", "optimizer-state",
)


def _is_excluded(path: Path) -> bool:
    """Reject runtime/private artifacts regardless of where a walk finds them."""
    relative_parts = path.relative_to(PROJECT_ROOT).parts
    if any(part in EXCLUDED_DIR_NAMES for part in relative_parts[:-1]):
        return True
    name = path.name
    folded = name.casefold()
    if folded in EXCLUDED_FILE_NAMES or folded.startswith(".env"):
        return True
    if path.suffix.casefold() in EXCLUDED_SUFFIXES:
        return True
    if folded.startswith(".mylm-omni-"):
        return True
    if any(fragment in folded for fragment in EXCLUDED_NAME_FRAGMENTS):
        return True
    return False


def _selected_roots() -> list[Path]:
    selected = [
        PROJECT_ROOT / name
        for name in sorted(ALLOWED_ROOT_FILES)
        if (PROJECT_ROOT / name).is_file()
    ]
    selected.extend(
        PROJECT_ROOT / name
        for name in sorted(ALLOWED_ROOT_DIRS)
        if (PROJECT_ROOT / name).is_dir()
    )
    data_dir = PROJECT_ROOT / "data"
    if data_dir.is_dir():
        selected.extend(
            data_dir / name
            for name in sorted(ALLOWED_DATA_FILES)
            if (data_dir / name).is_file()
        )
    return selected


def _iter_files() -> list[Path]:
    """Enumerate selected regular files; symlinks are intentionally not followed."""
    files: list[Path] = []
    for root in _selected_roots():
        if root.is_symlink():
            continue
        if root.is_file():
            if not _is_excluded(root):
                files.append(root)
            continue
        for current_root, directories, file_names in os.walk(root, followlinks=False):
            current = Path(current_root)
            directories[:] = sorted(
                name for name in directories
                if name not in EXCLUDED_DIR_NAMES
                and not (current / name).is_symlink()
            )
            for name in sorted(file_names):
                candidate = current / name
                if candidate.is_symlink() or not candidate.is_file() or _is_excluded(candidate):
                    continue
                files.append(candidate)
    # Deduplicate and sort by archive path for deterministic output.
    return sorted(set(files), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def create_zip(output_path: Path) -> dict:
    output_path = output_path.expanduser().resolve()
    if output_path == PROJECT_ROOT or PROJECT_ROOT in output_path.parents:
        raise ValueError("output ZIP must be placed outside the project tree")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = _iter_files()
    if not files:
        raise ValueError("no project files found to package")

    archive_name = "mylm-omni"
    fd, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    try:
        with zipfile.ZipFile(
            temporary_name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8
        ) as archive:
            for source in files:
                relative = source.relative_to(PROJECT_ROOT).as_posix()
                internal_path = str(PurePosixPath(archive_name) / relative)
                # Preserve source-code permissions only as ordinary ZIP metadata, not executable
                # host file permissions; launchers can be run explicitly through bash/PowerShell.
                info = zipfile.ZipInfo(internal_path)
                info.date_time = (2020, 1, 1, 0, 0, 0)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, source.read_bytes())

            manifest = {
                "package": "MyLM-omni research scaffold",
                "package_version": "0.1.0",
                "source_root": str(PROJECT_ROOT.name),
                "files": [
                    {
                        "path": path.relative_to(PROJECT_ROOT).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in files
                ],
                "file_count": len(files),
                "contains_model_weights": False,
                "contains_training_data": False,
                "contains_runtime_memory_or_audit_logs": False,
                "notes": [
                    "No pretrained or trained model weights are bundled.",
                    "Downloaded datasets and JSONL training/trajectory files are excluded.",
                    "Local model, mmproj and Whisper dependencies must be prepared separately.",
                    "Check model and dataset licenses before use.",
                ],
            }
            archive.writestr(
                f"{archive_name}/PACKAGE_MANIFEST.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

    # Validate the archive by reopening it and checking paths + the manifest.
    with zipfile.ZipFile(output_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"ZIP integrity failure at {bad_member}")
        names = archive.namelist()
        if not names or any(name.startswith("/") or ".." in PurePosixPath(name).parts for name in names):
            raise RuntimeError("ZIP contains unsafe or empty paths")
        manifest = json.loads(archive.read(f"{archive_name}/PACKAGE_MANIFEST.json"))
        if manifest["file_count"] != len(files):
            raise RuntimeError("package manifest file count mismatch")
        for name in names:
            p = PurePosixPath(name)
            if any(part in EXCLUDED_DIR_NAMES for part in p.parts):
                raise RuntimeError(f"excluded directory leaked into ZIP: {name}")
            if p.suffix.casefold() in EXCLUDED_SUFFIXES:
                raise RuntimeError(f"excluded file type leaked into ZIP: {name}")
            if p.name.casefold().startswith(".env") or p.name.casefold() in EXCLUDED_FILE_NAMES:
                raise RuntimeError(f"sensitive filename leaked into ZIP: {name}")

    return {
        "output": str(output_path),
        "size_bytes": output_path.stat().st_size,
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "files": len(files),
        "manifest": manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="ZIP path, outside project directory")
    args = parser.parse_args()
    result = create_zip(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
