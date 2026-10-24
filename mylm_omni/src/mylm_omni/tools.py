"""Safe tool adapters for local files and image understanding."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mylm_omni.agent import ToolSpec


def make_local_image_inspection_tool(
    vlm_backend: Any,
    *,
    allowed_roots: list[str | Path],
) -> ToolSpec:
    """Create a VLM tool that reads images only beneath explicitly configured directories."""
    roots = [Path(root).expanduser().resolve() for root in allowed_roots]
    if not roots:
        raise ValueError("at least one allowed image root is required")

    def inspect_image(path: str, question: str = "Describe the image accurately.") -> dict:
        requested = Path(path).expanduser().resolve()
        if not any(requested == root or root in requested.parents for root in roots):
            raise PermissionError("image path is outside configured read-only image roots")
        if not requested.is_file():
            raise FileNotFoundError(requested)
        answer = vlm_backend.complete(
            question,
            image_paths=[requested],
            max_new_tokens=384,
            temperature=0.0,
        )
        return {
            "path": str(requested),
            "question": question,
            "description": answer,
            "warning": "image descriptions are model output and need verification for critical decisions",
        }

    return ToolSpec(
        name="inspect_local_image",
        description="Read a local image under configured roots and return a model-generated description.",
        function=inspect_image,
        risk_level="medium",
        requires_approval=False,
        allowed_args=frozenset({"path", "question"}),
    )
