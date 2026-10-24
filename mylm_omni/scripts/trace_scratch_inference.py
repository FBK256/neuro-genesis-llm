"""Generate from a MyLM-omni scratch checkpoint and export per-layer inference traces.

Example:
  python scripts/trace_scratch_inference.py \
    --checkpoint runs/mylm-scratch \
    --tokenizer-dir ./tokenizer/mylm-omni \
    --prompt "Summarize the task" \
    --output ./artifacts/inference-trace.json

This script uses only local model/tokenizer files, reads safetensors weights (not pickle trainer
state), and stores hashes rather than the raw prompt by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, load_model


def sha256_file(path: Path) -> str:
    """Hash a local artifact in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_local_image_features(
    path: Path,
    *,
    feature_key: str,
    mask_key: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load precomputed vision features as safetensors, never an arbitrary pickle."""
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        raise ValueError("--image-features must point to a local .safetensors file")
    tensors = load_file(str(path), device="cpu")
    if feature_key not in tensors:
        raise KeyError(f"image feature tensor {feature_key!r} not found in {path}")
    features = tensors[feature_key]
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3 or features.shape[0] != 1:
        raise ValueError("image features must have shape [patches, dim] or [1, patches, dim]")
    mask = tensors.get(mask_key)
    if mask is not None:
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != features.shape[:2]:
            raise ValueError("image feature mask must match [batch, patches]")
        mask = mask.to(device=device)
    return features.to(device=device), mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="CheckpointManager output root containing latest/")
    parser.add_argument("--tokenizer-dir", type=Path, required=True,
                        help="already downloaded/local tokenizer directory")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--image-features", type=Path, default=None,
                        help="optional precomputed image features in safetensors")
    parser.add_argument("--image-feature-key", default="image_features")
    parser.add_argument("--image-mask-key", default="image_attention_mask")
    parser.add_argument("--trace-token-limit", type=int, default=128)
    parser.add_argument("--include-prompt", action="store_true",
                        help="save raw prompt in output (off by default for privacy)")
    args = parser.parse_args()

    if not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if not 1 <= args.max_new_tokens <= 256:
        parser.error("--max-new-tokens must be between 1 and 256 for trace output")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be >=0 and top-p must be in (0, 1]")
    if args.trace_token_limit < 1:
        parser.error("--trace-token-limit must be positive")

    root = args.checkpoint.expanduser().resolve()
    latest = root / "latest"
    manifest_path = latest / "manifest.json"
    weights_path = latest / "model.safetensors"
    if not manifest_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"expected {manifest_path} and {weights_path}; create a checkpoint first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_config = manifest.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint manifest has no model config")
    if manifest.get("format") != "mylm-omni-checkpoint-v1":
        raise ValueError("checkpoint manifest format is not a MyLM-omni scratch checkpoint")

    from transformers import AutoTokenizer
    from mylm_omni.modeling import MyLMConfig, MyLMForCausalLM

    tokenizer_dir = args.tokenizer_dir.expanduser().resolve()
    if not tokenizer_dir.is_dir():
        raise NotADirectoryError(tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    config = MyLMConfig(**raw_config)
    if len(tokenizer) != config.vocab_size:
        raise ValueError(
            f"tokenizer size {len(tokenizer)} does not match checkpoint vocab_size "
            f"{config.vocab_size}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyLMForCausalLM(config)
    load_model(model, str(weights_path), strict=True)
    model.to(device).eval()

    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    image_features = None
    image_attention_mask = None
    if args.image_features:
        image_features, image_attention_mask = load_local_image_features(
            args.image_features.expanduser().resolve(),
            feature_key=args.image_feature_key,
            mask_key=args.image_mask_key,
            device=device,
        )
        if config.image_feature_size is None:
            raise ValueError("checkpoint config has no image_feature_size")
        if image_features.shape[-1] != config.image_feature_size:
            raise ValueError(
                f"image feature dim {image_features.shape[-1]} does not match "
                f"image_feature_size {config.image_feature_size}"
            )

    with torch.inference_mode():
        result_ids, trace = model.generate_with_trace(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            temperature=args.temperature,
            top_p=args.top_p,
            attention_mask=attention_mask,
            image_features=image_features,
            image_attention_mask=image_attention_mask,
            trace_token_limit=args.trace_token_limit,
        )
    prompt_length = int(input_ids.shape[1])
    generated_ids = result_ids[0, prompt_length:].detach().cpu().tolist()
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    trace.update({
        "checkpoint_root": str(root),
        "checkpoint_step": manifest.get("step"),
        "checkpoint_weights_sha256": sha256_file(weights_path),
        "tokenizer_dir": str(tokenizer_dir),
        "tokenizer_size": len(tokenizer),
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "prompt": args.prompt if args.include_prompt else None,
        "generated_text": answer,
        "generated_token_ids": generated_ids,
        "device": str(device),
        "image_features_file": str(args.image_features.expanduser().resolve())
        if args.image_features else None,
        "generation_settings": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "trace_token_limit": args.trace_token_limit,
        },
    })
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(answer)
    print(f"Trace saved: {output}")
    print(
        f"steps={trace['generated_tokens']} "
        f"parameters={trace['parameter_count']:,} "
        f"elapsed_ms={trace['total_elapsed_ms']:.2f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        raise SystemExit(f"Trace export failed: {type(exc).__name__}: {exc}") from exc
