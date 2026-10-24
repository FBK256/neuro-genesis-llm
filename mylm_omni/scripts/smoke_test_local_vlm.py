"""Smoke-test a llama.cpp multimodal endpoint through the loopback-only backend.

This does not download weights or send prompts to cloud providers. Use --image to test vision
input separately from the basic text endpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mylm_omni.llama_cpp_backend import LlamaCppLoopbackBackend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default="local-vlm")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    backend = LlamaCppLoopbackBackend(
        model=args.model,
        base_url=args.endpoint,
        timeout_seconds=args.timeout_seconds,
    )
    report: dict[str, object] = {
        "endpoint": args.endpoint,
        "model_alias": args.model,
        "local_only": True,
        "text_test": {},
    }
    text_answer = backend.complete(
        "Return exactly the token LOCAL_VLM_OK and nothing else.",
        max_new_tokens=12,
        temperature=0.0,
    )
    text_test = {
        "passed": "LOCAL_VLM_OK" in text_answer,
        "response_characters": len(text_answer),
        "stats": backend.last_call_stats,
    }
    report["text_test"] = text_test
    image_passed = True
    if args.image is not None:
        image_path = args.image.expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        answer = backend.complete(
            "Describe the visible contents of this image in one concise sentence. "
            "Mention uncertainty instead of guessing illegible details.",
            image_paths=[image_path],
            max_new_tokens=128,
            temperature=0.0,
        )
        image_test = {
            "passed": bool(answer.strip()),
            "image_name": image_path.name,
            "response_characters": len(answer),
            "stats": backend.last_call_stats,
        }
        report["image_test"] = image_test
        image_passed = bool(image_test["passed"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if bool(text_test["passed"]) and image_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
