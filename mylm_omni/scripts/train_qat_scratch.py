"""Int4 QAT for the custom MyLM-omni scratch architecture.

Input must be a valid MyLM-omni checkpoint saved by CheckpointManager. The model is prepared
with TorchAO fake quantization, trained on local pre-tokenized JSONL, checkpointed every N
optimizer steps, and finally converted to an int4 weight-only inference representation.

This script does not perform QAT on the Qwen3-VL pretrained model; use the dedicated supported
Transformers/torchtune/Unsloth recipes for that architecture. Check torchao/model compatibility.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_model

from mylm_omni.modeling import MyLMConfig, MyLMForCausalLM
from mylm_omni.training.checkpoints import CheckpointManager
from mylm_omni.training.data import EpochRandomSampler, JsonlTokenDataset, TokenBatchCollator
from mylm_omni.training.train_scratch import create_run_name, evaluate, _write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True,
                        help="existing MyLM-omni checkpoint root containing latest/manifest.json")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", default=None)
    parser.add_argument("--output-dir", default="runs/mylm-omni-qat-int4")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--keep-steps", type=int, default=3)
    parser.add_argument("--keep-best", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--tokenizer-dir", default=None,
                        help="optional trained tokenizer directory copied into quantized-final")
    parser.add_argument("--parameter-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--optimizer", choices=("adamw", "adamw8bit"), default="adamw")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mylm-omni")
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    args = parser.parse_args()

    if args.max_steps < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("max-steps, batch-size, and gradient-accumulation must be positive")
    if args.checkpoint_steps < 1 or args.group_size < 1 or args.log_every < 1:
        parser.error("checkpoint-steps, group-size, and log-every must be positive")
    if args.parameter_dtype == "bf16" and (
        not torch.cuda.is_available() or args.optimizer != "adamw8bit"
    ):
        parser.error("bf16 parameters require a CUDA GPU and --optimizer adamw8bit")

    try:
        from torchao.quantization import quantize_, Int4WeightOnlyConfig
        from torchao.quantization.qat import QATConfig
    except ImportError as exc:
        raise RuntimeError("Install the optional QAT dependencies: pip install -e '.[qat]'") from exc

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported()
        else (torch.float16 if amp_enabled else torch.float32)
    )
    parameter_dtype = torch.bfloat16 if args.parameter_dtype == "bf16" else torch.float32

    base_root = Path(args.base_checkpoint)
    base_manifest_path = base_root / "latest" / "manifest.json"
    if not base_manifest_path.is_file():
        raise FileNotFoundError(f"missing base checkpoint manifest: {base_manifest_path}")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if not base_manifest.get("config"):
        raise ValueError("base checkpoint manifest does not include model config")
    config = MyLMConfig(**base_manifest["config"])
    config.max_seq_len = min(config.max_seq_len, args.max_seq_len)
    model = MyLMForCausalLM(config).to(device=device, dtype=parameter_dtype)
    restored_base = CheckpointManager(base_root).load_latest(model=model, map_location="cpu")
    if restored_base is None:
        raise RuntimeError("base checkpoint is incomplete")
    print(f"Loaded base model from step {restored_base['step']}")

    # Keep the tied vocabulary projection out of QAT replacement: it shares token embeddings.
    qat_base_config = Int4WeightOnlyConfig(group_size=args.group_size)
    quantize_(
        model,
        QATConfig(qat_base_config, step="prepare"),
        filter_fn=lambda module, fqn: isinstance(module, torch.nn.Linear)
        and not fqn.endswith("lm_head"),
    )

    train_data = JsonlTokenDataset(args.train_jsonl)
    collator = TokenBatchCollator(args.pad_token_id, config.max_seq_len)
    sampler = EpochRandomSampler(train_data, seed=args.seed, epoch=0)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collator, num_workers=0, pin_memory=device.type == "cuda",
    )
    eval_loader = None
    if args.eval_jsonl:
        eval_loader = DataLoader(
            JsonlTokenDataset(args.eval_jsonl), batch_size=args.batch_size, shuffle=False,
            collate_fn=collator, num_workers=0, pin_memory=device.type == "cuda",
        )

    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.jsonl"
    checkpoints = CheckpointManager(
        output_root, checkpoint_steps=args.checkpoint_steps,
        keep_steps=args.keep_steps, keep_best=args.keep_best,
        best_metric="eval_loss", lower_is_better=True,
    )
    run_name = create_run_name(model.num_parameters())
    global_step = 0
    epoch = 0
    start_epoch = 0
    next_batch = 0
    if args.resume == "auto":
        restored = checkpoints.load_latest(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, map_location="cpu"
        )
        if restored:
            global_step = int(restored["step"])
            state = restored.get("training_state", {})
            epoch = int(state.get("epoch", 0))
            start_epoch = epoch
            next_batch = int(state.get("next_batch_index", 0))
            print(f"Resumed QAT from step={global_step}, epoch={epoch}, next_batch={next_batch}")

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"{run_name}-qat-int4",
                config={
                    "base_checkpoint": str(base_root),
                    "base_step": restored_base["step"],
                    "group_size": args.group_size,
                    "max_steps": args.max_steps,
                    "batch_size": args.batch_size,
                    "gradient_accumulation": args.gradient_accumulation,
                    "parameter_dtype": args.parameter_dtype,
                    "optimizer": args.optimizer,
                    "train_data": args.train_jsonl,
                    "eval_data": args.eval_jsonl,
                },
            )
        except Exception as exc:
            print(f"W&B initialization warning: {exc}")

    started = time.monotonic()
    last_loss = float("nan")
    micro_count = 0
    window_loss = 0.0
    window_micro_batches = 0
    window_optimizer_steps = 0
    window_start_step = global_step
    window_start = started
    optimizer.zero_grad(set_to_none=True)
    model.train()

    while global_step < args.max_steps:
        sampler.set_epoch(epoch)
        for batch_index, raw_batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < next_batch:
                continue
            next_batch = batch_index + 1
            batch = {key: value.to(device, non_blocking=True) for key, value in raw_batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    image_features=batch.get("image_features"),
                    image_attention_mask=batch.get("image_attention_mask"),
                )
                loss = output["loss"]
            if loss is None or not bool(torch.isfinite(loss)):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"non-finite QAT loss at step {global_step}")
            last_loss = float(loss.detach().cpu())
            micro_count += 1
            window_loss += last_loss
            window_micro_batches += 1
            loss_for_backward = loss / args.gradient_accumulation
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            end_epoch = batch_index == len(train_loader) - 1
            if micro_count < args.gradient_accumulation and not end_epoch:
                continue
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            if micro_count < args.gradient_accumulation:
                correction = args.gradient_accumulation / micro_count
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError(f"non-finite QAT gradient at step {global_step}")
            if scaler.is_enabled():
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                did_update = scaler.get_scale() >= old_scale
            else:
                optimizer.step()
                did_update = True
            optimizer.zero_grad(set_to_none=True)
            if did_update:
                global_step += 1
                scheduler.step()
                window_optimizer_steps += 1
            micro_count = 0

            if (
                global_step > 0
                and global_step % args.log_every == 0
                and window_optimizer_steps > 0
            ):
                now = time.monotonic()
                elapsed = max(1e-9, now - window_start)
                measured_steps = max(1, global_step - window_start_step)
                record = {
                    "step": global_step,
                    "qat_train_loss_window": window_loss / max(1, window_micro_batches),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(grad_norm.detach().float().cpu()),
                    "steps_per_second": measured_steps / elapsed,
                    "elapsed_seconds": now - started,
                    "eta_seconds": max(0, args.max_steps - global_step)
                    / max(1e-9, measured_steps / elapsed),
                    "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()) if device.type == "cuda" else 0,
                }
                _write_jsonl(metrics_path, record)
                if wandb_run is not None:
                    try:
                        wandb_run.log(record, step=global_step)
                    except Exception:
                        pass
                print(
                    f"QAT step={global_step}/{args.max_steps} loss={record['qat_train_loss_window']:.4f} "
                    f"ETA~{record['eta_seconds']/60:.1f}m"
                )
                window_start = now
                window_start_step = global_step
                window_loss = 0.0
                window_micro_batches = 0
                window_optimizer_steps = 0

            if did_update and global_step > 0 and global_step % args.checkpoint_steps == 0:
                eval_metrics = evaluate(model, eval_loader, device, amp_enabled, amp_dtype) if eval_loader else {}
                state = {
                    "epoch": epoch,
                    "next_batch_index": batch_index + 1,
                    "seed": args.seed,
                    "run_name": run_name,
                    "qat_config": {"group_size": args.group_size, "weight_dtype": "int4"},
                }
                checkpoints.save(
                    step=global_step, model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, training_state=state,
                    metrics=eval_metrics or {"train_loss": last_loss},
                )
                if wandb_run is not None and eval_metrics:
                    try:
                        wandb_run.log(eval_metrics, step=global_step)
                    except Exception:
                        pass

            if global_step >= args.max_steps:
                break
        else:
            epoch += 1
            next_batch = 0
            continue
        if global_step >= args.max_steps:
            break
        epoch += 1
        next_batch = 0

    final_metrics = evaluate(model, eval_loader, device, amp_enabled, amp_dtype) if eval_loader else {
        "train_loss": last_loss
    }
    checkpoints.save(
        step=global_step, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        training_state={
            "epoch": epoch, "next_batch_index": next_batch, "seed": args.seed,
            "run_name": run_name, "qat_group_size": args.group_size,
        },
        metrics=final_metrics,
    )

    # Keep the resumable fake-quant checkpoint intact. Export converted weights separately.
    export_dir = output_root / "quantized-final"
    export_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    quantize_(
        model,
        QATConfig(qat_base_config, step="convert"),
        filter_fn=lambda module, fqn: isinstance(module, torch.nn.Linear)
        and not fqn.endswith("lm_head"),
    )
    save_model(model, str(export_dir / "model.safetensors"))
    if args.tokenizer_dir:
        import shutil
        tokenizer_dir = Path(args.tokenizer_dir)
        if not tokenizer_dir.is_dir():
            raise NotADirectoryError(tokenizer_dir)
        for source in tokenizer_dir.iterdir():
            if source.is_file():
                shutil.copy2(source, export_dir / source.name)
    (export_dir / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (export_dir / "quantization.json").write_text(
        json.dumps({
            "method": "TorchAO QAT Int4WeightOnlyConfig",
            "group_size": args.group_size,
            "source_checkpoint": str(base_root),
            "training_steps": global_step,
            "format": "safetensors",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if wandb_run is not None:
        try:
            wandb_run.log(final_metrics, step=global_step)
            wandb_run.finish()
        except Exception:
            pass
    print(f"QAT finished: step={global_step}, exported quantized model to {export_dir}")


if __name__ == "__main__":
    main()
