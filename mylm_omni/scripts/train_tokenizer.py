"""Train a reusable byte-level BPE tokenizer from curated local JSONL files.

Every input JSONL row must contain a text field. The resulting Hugging Face-compatible
tokenizer is saved locally; the emitted manifest records vocabulary size so the model config
can be made consistent with the tokenizer. No network download is required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<bos>", "<eos>",
    "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|image|>", "<|audio|>",
]


def read_texts(files: list[str], field: str):
    """Stream text examples so a large corpus does not have to fit in RAM."""
    for filename in files:
        with open(filename, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {filename}:{line_number}: {exc}") from exc
                value = row.get(field)
                if isinstance(value, str) and value.strip():
                    yield value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", nargs="+", required=True)
    parser.add_argument("--output-dir", default="tokenizer/mylm-omni")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--vocab-size", type=int, default=262_144)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--allow-smaller-vocab", action="store_true",
                        help="allow a small corpus to produce fewer tokens than requested")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()
    if args.vocab_size < len(SPECIAL_TOKENS) + 256:
        parser.error("vocab-size must leave room for all special tokens and byte alphabet")
    if args.min_frequency < 1:
        parser.error("min-frequency must be >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # ByteLevel's full alphabet prevents unknown UTF-8 bytes even for Japanese or code.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=args.show_progress,
    )
    tokenizer.train_from_iterator(
        read_texts(args.input_jsonl, args.text_field),
        trainer=trainer,
        length=None,
    )
    actual_vocab_size = len(tokenizer.get_vocab())
    if actual_vocab_size < args.vocab_size and not args.allow_smaller_vocab:
        raise RuntimeError(
            f"Tokenizer corpus produced {actual_vocab_size:,} tokens, below the requested "
            f"{args.vocab_size:,}. Add representative text or explicitly pass "
            "--allow-smaller-vocab and copy the actual size into the model config."
        )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=SPECIAL_TOKENS[4:],
        clean_up_tokenization_spaces=False,
    )
    fast.save_pretrained(output_dir)
    manifest = {
        "tokenizer_type": "byte-level BPE",
        "vocab_size": len(fast),
        "special_tokens": {
            "pad_token_id": fast.pad_token_id,
            "unk_token_id": fast.unk_token_id,
            "bos_token_id": fast.bos_token_id,
            "eos_token_id": fast.eos_token_id,
        },
        "inputs": args.input_jsonl,
        "text_field": args.text_field,
        "minimum_frequency": args.min_frequency,
        "output_dir": str(output_dir),
        "note": "ModelConfig.vocab_size must equal this vocab_size; freeze tokenizer after training.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
