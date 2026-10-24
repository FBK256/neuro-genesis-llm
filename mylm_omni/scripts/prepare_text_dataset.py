"""Parallelize tokenization of curated text JSONL into token-id JSONL for scratch training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True,
                        help="local tokenizer path or explicitly selected Hugging Face tokenizer ID")
    parser.add_argument("--output-dir", default="data/text/tokenized")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.max_length < 2 or args.processes < 1:
        parser.error("max-length must be >= 2 and processes must be >= 1")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=False)
    eos_id = tokenizer.eos_token_id

    # Load JSONL files using Apache Arrow, then use Dataset.map(num_proc=...) for CPU parallelism.
    dataset = load_dataset("json", data_files=args.input_jsonl, split="train")
    field_name = args.text_field

    def tokenize_batch(batch):
        texts = [str(text or "") for text in batch[field_name]]
        encoded = tokenizer(texts, truncation=True, max_length=args.max_length, add_special_tokens=True)
        ids_rows = encoded["input_ids"]
        if eos_id is not None:
            ids_rows = [ids + ([] if ids and ids[-1] == eos_id else [eos_id]) for ids in ids_rows]
        return {"input_ids": ids_rows}

    tokenized = dataset.map(
        tokenize_batch, batched=True, batch_size=args.batch_size, num_proc=args.processes,
        remove_columns=dataset.column_names, desc="Tokenizing curated text",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tokens.jsonl"
    with output_path.open("w", encoding="utf-8") as out:
        for index, row in enumerate(tokenized):
            ids = row["input_ids"][:args.max_length]
            if len(ids) >= 2:
                out.write(json.dumps({"input_ids": ids}, ensure_ascii=False) + "\n")
    manifest = {
        "inputs": args.input_jsonl,
        "tokenizer": args.tokenizer,
        "vocab_size": len(tokenizer),
        "max_length": args.max_length,
        "processes": args.processes,
        "rows": len(tokenized),
        "output": str(output_path),
        "note": "Review source licenses and deduplicate before using these tokens for training.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
