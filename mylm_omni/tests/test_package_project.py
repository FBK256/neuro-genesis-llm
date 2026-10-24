import importlib.util
import json
from pathlib import Path, PurePosixPath
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
PACKAGER = PROJECT / "scripts" / "package_project.py"
spec = importlib.util.spec_from_file_location("mylm_omni_packager", PACKAGER)
assert spec is not None and spec.loader is not None
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


def test_package_zip_is_valid_deterministic_and_contains_source_only(tmp_path):
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    result1 = packager.create_zip(first)
    result2 = packager.create_zip(second)

    assert result1["files"] > 50
    assert result1["size_bytes"] == first.stat().st_size
    assert result2["size_bytes"] == second.stat().st_size
    assert result1["manifest"]["contains_model_weights"] is False
    assert result1["manifest"]["contains_training_data"] is False

    # Archive-level metadata may vary by ZIP implementation; file membership and payload bytes
    # must be reproducible even when the compressed archive hash is not byte-identical.
    with zipfile.ZipFile(first, "r") as archive_first, zipfile.ZipFile(second, "r") as archive_second:
        assert archive_first.namelist() == archive_second.namelist()
        for name in archive_first.namelist():
            assert archive_first.read(name) == archive_second.read(name)
    with zipfile.ZipFile(first, "r") as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert "mylm-omni/README.md" in names
        assert "mylm-omni/scripts/run_local_agent.py" in names
        assert "mylm-omni/PACKAGE_MANIFEST.json" in names
        assert not any(PurePosixPath(name).suffix.lower() in {".gguf", ".safetensors", ".jsonl", ".sqlite"} for name in names)
        assert not any(part in {".git", ".venv", "runs", "models", "checkpoints"} for name in names for part in PurePosixPath(name).parts)
        manifest = json.loads(archive.read("mylm-omni/PACKAGE_MANIFEST.json"))
        assert manifest["file_count"] == result1["files"]
        assert manifest["contains_runtime_memory_or_audit_logs"] is False


def test_package_rejects_output_inside_source_tree(tmp_path):
    output = PROJECT / "nested-test-output.zip"
    try:
        try:
            packager.create_zip(output)
        except ValueError as exc:
            assert "outside the project tree" in str(exc)
        else:
            raise AssertionError("packager should reject output under the source tree")
    finally:
        output.unlink(missing_ok=True)
