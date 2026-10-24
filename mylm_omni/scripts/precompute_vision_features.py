"""Encode downloaded image-text manifest files into safe precomputed vision feature tensors.

This keeps image encoder parameters separate from the 1–3B text decoder and makes encoder
choice explicit. The output is a tokenized JSONL corpus consumable by train_scratch.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def _tokenize_row(row: dict, tokenizer, *, text_field: str, prompt_field: str | None,
                  target_field: str | None, max_length: int) -> tuple[list[int], list[int]] | None:
    eos = tokenizer.eos_token_id
    if prompt_field and target_field and row.get(prompt_field) is not None and row.get(target_field) is not None:
        prompt = str(row[prompt_field])
        target = str(row[target_field])
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        prompt_ids = prompt_ids[-max(1, max_length - 1):]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        if eos is not None:
            target_ids.append(eos)
        target_ids = target_ids[: max_length - len(prompt_ids)]
        if not target_ids:
            return None
        return prompt_ids + target_ids, [-100] * len(prompt_ids) + target_ids
    text = row.get(text_field)
    if not isinstance(text, str) or not text.strip():
        return None
    ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    if eos is not None and len(ids) < max_length and (not ids or ids[-1] != eos):
        ids.append(eos)
    return (ids, list(ids)) if len(ids) >= 2 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL from download_multimodal_datasets.py")
    parser.add_argument("--vision-model", default="google/siglip-base-patch16-224")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-jsonl", default="data/multimodal/tokenized/train.jsonl")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--target-field", default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()
    if args.max_length < 2 or args.batch_size < 1:
        parser.error("max-length must be >= 2 and batch-size must be positive")

    from PIL import Image
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )
    processor = AutoProcessor.from_pretrained(args.vision_model, trust_remote_code=False)
    vision = AutoModel.from_pretrained(
        args.vision_model,
        trust_remote_code=False,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    for parameter in vision.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=False)

    # SigLIP and several CLIP-family checkpoints expose vision_model; pure vision models
    # expose last_hidden_state directly. Unknown output shapes fail loudly.
    vision_tower = getattr(vision, "vision_model", vision)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_dir = output_path.parent / "vision_features"
    features_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                row = json.loads(line)
                if not row.get("image_path"):
                    raise ValueError(f"{manifest_path}:{line_number} lacks image_path")
                rows.append(row)

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as output:
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            images = []
            accepted_rows = []
            for row in chunk:
                try:
                    image = Image.open(row["image_path"]).convert("RGB")
                    images.append(image)
                    accepted_rows.append(row)
                except Exception as exc:
                    skipped += 1
                    print(f"SKIP image {row.get('image_path')}: {exc}")
            if not images:
                continue

            batch = processor(images=images, return_tensors="pt")
            pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
            with torch.inference_mode():
                result = vision_tower(pixel_values=pixel_values)
            hidden = getattr(result, "last_hidden_state", None)
            if hidden is None:
                hidden = getattr(getattr(result, "vision_model_output", None), "last_hidden_state", None)
            if hidden is None or hidden.ndim != 3:
                raise RuntimeError(
                    "vision model must return last_hidden_state shaped [batch, patches, features]"
                )
            hidden = hidden.detach().to(dtype=torch.float16, device="cpu").contiguous()

            for index, row in enumerate(accepted_rows):
                digest = row.get("image_sha256")
                if not digest:
                    digest = hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()
                feature_path = features_dir / f"{digest}.safetensors"
                if not feature_path.exists():
                    save_file({"features": hidden[index]}, str(feature_path))
                tokenized = _tokenize_row(
                    row, tokenizer, text_field=args.text_field,
                    prompt_field=args.prompt_field, target_field=args.target_field,
                    max_length=args.max_length,
                )
                if tokenized is None:
                    skipped += 1
                    continue
                input_ids, labels = tokenized
                relative_feature_path = feature_path.relative_to(output_path.parent).as_posix()
                record = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "image_features_path": relative_feature_path,
                    "image_sha256": digest,
                    "source_dataset": row.get("source_dataset"),
                    "source_split": row.get("source_split"),
                    "source_row": row.get("source_row"),
                    "license": row.get("license"),
                    "vision_model": args.vision_model,
                    "tokenizer": args.tokenizer,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            for image in images:
                image.close()

    summary = {
        "manifest": str(manifest_path),
        "vision_model": args.vision_model,
        "vision_hidden_size": int(hidden.shape[-1]) if 'hidden' in locals() else None,
        "decoder_tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "rows_written": written,
        "rows_skipped": skipped,
        "output_jsonl": str(output_path),
        "feature_cache": str(features_dir),
        "output_feature_dtype": "float16",
        "device": str(device),
        "note": "Set MyLMConfig.image_feature_size to vision_hidden_size before scratch training.",
    }
    (output_path.parent / "vision_features_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
