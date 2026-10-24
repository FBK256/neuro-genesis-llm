"""Create compact top-k teacher-logit traces from fixed input/output JSONL examples.

Input records must contain {"instruction": "...", "output": "..."} or {"text": "..."}.
The teacher is evaluated with teacher forcing; this script does not invent ground-truth answers.
Each output record includes top-k logits and a tail log-sum-exp computed at the same temperature.
Student and teacher must use a token-compatible vocabulary for the later sparse-KD stage.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from mylm_omni.training.teacher_data import TeacherTrace, append_jsonl


def _build_prompt(tokenizer, row: dict) -> tuple[list[int], list[int]]:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("teacher tokenizer must define eos_token_id")
    if "instruction" in row and "output" in row:
        instruction = str(row["instruction"])
        if str(row.get("input", "")).strip():
            instruction += "\n\nInput:\n" + str(row["input"])
        if getattr(tokenizer, "chat_template", None):
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            prompt_ids = tokenizer(
                f"### Instruction\n{instruction}\n\n### Response\n",
                add_special_tokens=True,
            )["input_ids"]
        answer_ids = tokenizer(str(row["output"]), add_special_tokens=False)["input_ids"]
    elif "text" in row:
        # Treat the first portion as context and distill only the continuation portion.
        # This avoids teacher logits at position zero, which cannot predict the first token.
        full_text_ids = tokenizer(str(row["text"]), add_special_tokens=True)["input_ids"]
        if len(full_text_ids) < 3:
            raise ValueError("text rows must contain at least three tokens for context and target")
        split_at = max(1, min(len(full_text_ids) - 1, len(full_text_ids) // 2))
        prompt_ids = full_text_ids[:split_at]
        answer_ids = full_text_ids[split_at:]
    else:
        raise ValueError("each input row needs instruction/output or text")
    if not answer_ids:
        raise ValueError("answer has no tokens")
    if answer_ids[-1] != eos_id:
        answer_ids.append(eos_id)
    return list(prompt_ids), list(answer_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-revision", default="main",
                        help="Record an immutable commit hash when reproducibility matters")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", default="data/teacher/traces.jsonl")
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    args = parser.parse_args()
    if args.top_k < 1 or args.temperature <= 0 or args.max_length < 2:
        parser.error("top-k, temperature, and max-length must be positive")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "fp32":
        dtype = torch.float32
    else:
        dtype = (
            torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
            else (torch.float16 if device.type == "cuda" else torch.float32)
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model, revision=args.teacher_revision, use_fast=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("teacher tokenizer does not define EOS")
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        revision=args.teacher_revision,
        trust_remote_code=False,
        torch_dtype=dtype,
        attn_implementation="sdpa" if device.type == "cuda" else "eager",
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vocab_size = int(model.config.vocab_size)
    if args.top_k >= vocab_size:
        raise ValueError(f"top-k={args.top_k} must be smaller than teacher vocabulary {vocab_size}")

    source_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = 0
    written = 0
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if args.max_rows and written >= args.max_rows:
                break
            row = json.loads(line)
            prompt_ids, answer_ids = _build_prompt(tokenizer, row)
            # The sequence always needs one context token to predict its first answer token.
            # Keep the answer tail when it is longer than context budget, and reserve EOS.
            if len(answer_ids) >= args.max_length:
                answer_ids = answer_ids[: args.max_length - 1]
                answer_ids[-1] = tokenizer.eos_token_id
                prompt_ids = []
            else:
                prompt_budget = args.max_length - len(answer_ids)
                prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
            if not prompt_ids:
                # For tokenizers without BOS, EOS is a safe explicit context sentinel.
                context_token = (
                    tokenizer.bos_token_id
                    if tokenizer.bos_token_id is not None
                    else tokenizer.eos_token_id
                )
                if len(answer_ids) >= args.max_length:
                    answer_ids = answer_ids[: args.max_length - 1]
                    answer_ids[-1] = tokenizer.eos_token_id
                prompt_ids = [context_token]
            full_ids = prompt_ids + answer_ids
            if len(full_ids) > args.max_length:
                raise RuntimeError("teacher sequence length exceeded max_length after truncation")
            if len(full_ids) < 2:
                print(f"SKIP {source_path}:{line_number}: sequence too short")
                continue
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                # Position p-1 predicts answer token p. Keep only answer positions before
                # allocating the much larger FP32 top-k/tail calculations.
                start = len(prompt_ids) - 1
                stop = start + len(answer_ids)
                answer_logits = outputs.logits[0, start:stop].float()
                del outputs
                if answer_logits.shape[0] != len(answer_ids):
                    raise RuntimeError("teacher output length does not align with answer tokens")
                scaled = answer_logits / args.temperature
                top_logits, top_ids = torch.topk(scaled, k=args.top_k, dim=-1)
                tail_mask = torch.zeros_like(scaled, dtype=torch.bool)
                tail_mask.scatter_(-1, top_ids, True)
                tail_logsumexp = torch.logsumexp(
                    scaled.masked_fill(tail_mask, float("-inf")), dim=-1
                )
            trace = TeacherTrace(
                teacher_model=args.teacher_model,
                teacher_revision=args.teacher_revision,
                tokenizer_name=args.teacher_model,
                tokenizer_revision=args.teacher_revision,
                prompt_ids=prompt_ids,
                answer_ids=answer_ids,
                topk_ids=top_ids.cpu().tolist(),
                topk_logits=top_logits.cpu().tolist(),
                tail_logsumexp=tail_logsumexp.cpu().tolist(),
                temperature=args.temperature,
                task_type=str(row.get("task_type", "instruction")),
                modality=str(row.get("modality", "text")),
                data_source=str(row.get("source_dataset", source_path.name)),
                license_id=str(row.get("license", "unknown")),
                quality_score=float(row.get("quality_score", 0.0)),
                metadata={
                    "source_path": str(source_path),
                    "source_row": line_number,
                    "source_id": row.get("id"),
                    "source_sha256": row.get("sha256"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "sequence_length": len(full_ids),
                    "vocab_size": vocab_size,
                    "tokenizer_revision": args.teacher_revision,
                },
            )
            append_jsonl(output_path, trace)
            seen += 1
            written += 1
            if written == 1 or written % 10 == 0:
                print(
                    f"teacher traces {written:,} | source row {line_number:,} | "
                    f"sequence={len(full_ids)} | top-k={args.top_k}"
                )
    print(json.dumps({
        "input": str(source_path),
        "output": str(output_path),
        "teacher": args.teacher_model,
        "teacher_revision": args.teacher_revision,
        "tokenizer": args.teacher_model,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "records_written": written,
        "source_records_seen": seen,
        "note": "Sparse top-k plus one tail bucket approximates the full distribution; it is not lossless.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
