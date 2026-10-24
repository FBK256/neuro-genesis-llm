"""Evaluate a MyLM-omni scratch checkpoint against a frozen tokenized JSONL split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mylm_omni.modeling import MyLMConfig, MyLMForCausalLM
from mylm_omni.training.checkpoints import CheckpointManager
from mylm_omni.training.data import JsonlTokenDataset, TokenBatchCollator
from mylm_omni.training.train_scratch import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.checkpoint_dir)
    manifest_path = root / "latest" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"latest/manifest.json not found in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = MyLMConfig(**manifest["config"])
    model = MyLMForCausalLM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loaded = CheckpointManager(root).load_latest(model=model, map_location="cpu")
    if loaded is None:
        raise RuntimeError("checkpoint is incomplete; model.safetensors/trainer_state.pt required")

    data = JsonlTokenDataset(args.eval_jsonl)
    loader = DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        collate_fn=TokenBatchCollator(args.pad_token_id, config.max_seq_len),
    )
    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported()
        else (torch.float16 if amp_enabled else torch.float32)
    )
    result = evaluate(
        model, loader, device, amp_enabled, amp_dtype, max_batches=args.max_batches,
    )
    result.update({
        "checkpoint_step": int(loaded["step"]),
        "eval_set": str(Path(args.eval_jsonl).resolve()),
        "parameter_count": model.num_parameters(),
        "device": str(device),
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
