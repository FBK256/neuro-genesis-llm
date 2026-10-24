from pathlib import Path

import pytest

from mylm_omni.tools import make_local_image_inspection_tool


class FakeVLM:
    def complete(self, prompt, **kwargs):
        return f"checked: {Path(kwargs['image_paths'][0]).name}"


def test_image_tool_accepts_path_inside_allowed_root(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = image_root / "sample.png"
    image.write_bytes(b"test-placeholder")
    tool = make_local_image_inspection_tool(FakeVLM(), allowed_roots=[image_root])
    result = tool.function(path=str(image), question="What is this?")
    assert result["description"] == "checked: sample.png"


def test_image_tool_rejects_path_outside_allowed_root(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    outside = tmp_path / "private.txt"
    outside.write_text("private", encoding="utf-8")
    tool = make_local_image_inspection_tool(FakeVLM(), allowed_roots=[image_root])
    with pytest.raises(PermissionError):
        tool.function(path=str(outside), question="read file")
