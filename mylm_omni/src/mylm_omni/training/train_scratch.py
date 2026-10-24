"""Train the MyLM-omni scratch architecture on pre-tokenized JSONL sequences.

Expected line format:
  {"input_ids": [1, 2, ...]}
  {"input_ids": [1, 2, ...], "labels": [-100, ..., 2, ...]}

For practical language ability, use the separate Hugging Face SFT path documented in the README.
This script is principally for reproducible architecture research and controlled ablations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import DataLoader

from mylm_omni.modeling import MyLMConfig, MyLMForCausalLM
from mylm_omni.training.checkpoints import CheckpointManager
from mylm_omni.training.data import EpochRandomSampler, JsonlTokenDataset, TokenBatchCollator
from mylm_omni.training.distillation import causal_lm_loss


def create_run_name(parameter_count: int) -> str:
    """Use the requested MyLM-omni parameter-count and JST execution-time convention."""
    if parameter_count >= 1_000_000_000:
        count = f"{parameter_count / 1_000_000_000:.2f}B"
    elif parameter_count >= 1_000_000:
        count = f"{parameter_count / 1_000_000:.1f}M"
    else:
        count = str(parameter_count)
    timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d-%H%M%S")
    return f"MyLM-omni-{count}-reasoning-{timestamp}"


def _read_json(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _autocast_settings(device: torch.device) -> tuple[bool, torch.dtype]:
    if device.type != "cuda":
        return False, torch.float32
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: MyLMForCausalLM,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_batches: int = 0,
) -> dict[str, float]:
    """Evaluate loss, perplexity, next-token accuracy, and routed compute depth."""
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    token_count = 0
    correct_count = 0
    expected_depth_sum = 0.0
    activation_sparsity_sum = 0.0
    activation_abs_mean_sum = 0.0
    activation_std_sum = 0.0
    activation_summary_count = 0
    finite_fraction_min = 1.0
    depth_histogram_sum: dict[str, float] = {}
    batches = 0
    for batch in loader:
        if max_batches and batches >= max_batches:
            break
        batch = _move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(
                batch["input_ids"], attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                image_features=batch.get("image_features"),
                image_attention_mask=batch.get("image_attention_mask"),
                return_trace=True,
                return_logits=False,
            )
            loss = output["lm_loss"]
        # Validation computes exact top-1 accuracy and loss in chunks instead of allocating
        # the full [batch, sequence, vocabulary] tensor.
        correct_count += int(output["correct_tokens"])
        valid_n = int(output["supervised_tokens"])
        token_count += valid_n
        if valid_n:
            loss_sum += float(loss.item()) * valid_n
        trace = output["trace"]
        expected_depth_sum += float(trace["mean_expected_depth"])
        for stage in (
            trace["unique_layer_activation_stats"] + trace["recurrent_activation_stats"]
        ):
            activation_sparsity_sum += float(stage["near_zero_fraction"])
            activation_abs_mean_sum += abs(float(stage["mean"]))
            activation_std_sum += float(stage["std"])
            finite_fraction_min = min(finite_fraction_min, float(stage["finite_fraction"]))
            activation_summary_count += 1
        for depth, fraction in trace["depth_histogram"].items():
            depth_histogram_sum[depth] = depth_histogram_sum.get(depth, 0.0) + float(fraction)
        batches += 1

    if was_training:
        model.train()
    if batches == 0 or token_count == 0:
        return {"eval_loss": math.nan, "eval_perplexity": math.nan, "eval_accuracy": math.nan,
                "mean_expected_depth": math.nan, "activation_sparsity": math.nan,
                "activation_abs_mean": math.nan, "activation_std": math.nan,
                "activation_finite_fraction_min": math.nan, "eval_tokens": 0}
    mean_loss = loss_sum / token_count
    return {
        "eval_loss": mean_loss,
        "eval_perplexity": math.exp(min(mean_loss, 20.0)),
        "eval_accuracy": correct_count / token_count,
        "mean_expected_depth": expected_depth_sum / batches,
        "activation_sparsity": activation_sparsity_sum / max(1, activation_summary_count),
        "activation_abs_mean": activation_abs_mean_sum / max(1, activation_summary_count),
        "activation_std": activation_std_sum / max(1, activation_summary_count),
        "activation_finite_fraction_min": finite_fraction_min,
        **{
            f"predicted_depth_{depth}_fraction": fraction / batches
            for depth, fraction in depth_histogram_sum.items()
        },
        "eval_tokens": float(token_count),
    }


def _write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True, help="JSON file with MyLMConfig fields")
    parser.add_argument("--train-jsonl", required=True, help="pre-tokenized training JSONL")
    parser.add_argument("--eval-jsonl", default=None, help="optional frozen evaluation JSONL")
    parser.add_argument("--output-dir", default="checkpoints/MyLM-omni")
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=("adamw", "adamw8bit"), default="adamw",
                        help="fused CUDA AdamW is used when supported; adamw8bit saves optimizer VRAM")
    parser.add_argument("--parameter-dtype", choices=("fp32", "bf16"), default="fp32",
                        help="bf16 reduces memory, but requires CUDA plus AdamW8bit for this script")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--keep-steps", type=int, default=3)
    parser.add_argument("--keep-best", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument("--wandb", action="store_true", help="enable optional Weights & Biases logging")
    parser.add_argument("--wandb-project", default="mylm-omni")
    parser.add_argument("--eval-max-batches", type=int, default=0,
                        help="0 means evaluate all validation batches")
    args = parser.parse_args()

    if args.max_steps < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("max-steps, batch-size, and gradient-accumulation must be positive")
    if args.log_every < 1 or args.max_seq_len < 2:
        parser.error("log-every must be positive and max-seq-len must be >= 2")
    if args.parameter_dtype == "bf16" and (args.optimizer != "adamw8bit" or not torch.cuda.is_available()):
        parser.error("--parameter-dtype bf16 requires a CUDA GPU and --optimizer adamw8bit")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    raw_config = _read_json(args.model_config)
    model_config = MyLMConfig(**raw_config)
    if args.max_seq_len > model_config.max_position_embeddings:
        parser.error("--max-seq-len cannot exceed max_position_embeddings")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled, amp_dtype = _autocast_settings(device)

    parameter_dtype = (
        torch.bfloat16 if args.parameter_dtype == "bf16" else torch.float32
    )
    model = MyLMForCausalLM(model_config).to(device=device, dtype=parameter_dtype)
    total_parameters = model.num_parameters()
    run_name = create_run_name(total_parameters)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    train_data = JsonlTokenDataset(args.train_jsonl)
    collator = TokenBatchCollator(args.pad_token_id, args.max_seq_len)
    sampler = EpochRandomSampler(train_data, seed=args.seed, epoch=0)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=sampler,
        collate_fn=collator, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = None
    if args.eval_jsonl:
        eval_data = JsonlTokenDataset(args.eval_jsonl)
        eval_loader = DataLoader(
            eval_data, batch_size=args.batch_size, shuffle=False,
            collate_fn=collator, num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError(
                "Install bitsandbytes for --optimizer adamw8bit: pip install bitsandbytes"
            ) from exc
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
            betas=(0.9, 0.95), eps=1e-8,
        )
    else:
        optimizer_kwargs = {
            "lr": args.learning_rate, "weight_decay": args.weight_decay,
            "betas": (0.9, 0.95), "eps": 1e-8,
        }
        if device.type == "cuda":
            try:
                optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs, fused=True)
            except (TypeError, RuntimeError):
                optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    def lr_lambda(step: int) -> float:
        if args.warmup_steps and step < args.warmup_steps:
            return max(1e-8, (step + 1) / args.warmup_steps)
        decay_steps = max(1, args.max_steps - args.warmup_steps)
        progress = min(1.0, max(0.0, (step - args.warmup_steps) / decay_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    checkpoints = CheckpointManager(
        output_dir, checkpoint_steps=args.checkpoint_steps, keep_steps=args.keep_steps,
        keep_best=args.keep_best, best_metric="eval_loss", lower_is_better=True,
    )

    start_step = 0
    start_epoch = 0
    next_batch = 0
    if args.resume == "auto":
        restored = checkpoints.load_latest(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            map_location="cpu",
        )
        if restored:
            start_step = int(restored["step"])
            state = restored.get("training_state", {})
            start_epoch = int(state.get("epoch", 0))
            next_batch = int(state.get("next_batch_index", 0))
            print(f"Resumed from step {start_step}, epoch {start_epoch}, batch {next_batch}")

    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config={
                    **model_config.to_dict(),
                    "optimizer": args.optimizer,
                    "parameter_dtype": args.parameter_dtype,
                    "gradient_checkpointing": model_config.gradient_checkpointing,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "gradient_accumulation": args.gradient_accumulation,
                    "seed": args.seed,
                    "device": str(device),
                    "model_parameters": total_parameters,
                    "estimated_train_flops_per_token_linear_component": estimated_train_flops_per_token,
                    "training_data": str(args.train_jsonl),
                    "eval_data": str(args.eval_jsonl) if args.eval_jsonl else None,
                },
            )
        except Exception as exc:
            print(f"WARNING: W&B initialization failed; metrics remain in {metrics_path}: {exc}")

    print(f"Run: {run_name}")
    print(f"Device: {device} | Parameters: {total_parameters:,} | Train rows: {len(train_data):,}")
    recurrent_parameter_count = sum(p.numel() for p in model.recurrent_block.parameters())
    unique_parameter_count = total_parameters - recurrent_parameter_count
    # Leading linear-in-parameters estimate only; excludes attention's quadratic-in-length term,
    # optimizer kernels, data stalls, vision encoder preprocessing, and evaluation overhead.
    estimated_train_flops_per_token = 6 * (
        unique_parameter_count + model_config.max_recurrent_steps * recurrent_parameter_count
    )
    print(
        f"AMP: {amp_dtype if amp_enabled else 'disabled'} | Parameters: {parameter_dtype} | "
        f"Optimizer: {args.optimizer} | Gradient checkpointing: {model_config.gradient_checkpointing}"
    )
    print(f"Output: {output_dir.resolve()}")
    print(
        f"Approx. training FLOPs/token (linear parameter term only): "
        f"{estimated_train_flops_per_token:,}; excludes attention quadratic term"
    )
    global_step = start_step
    epoch = start_epoch
    total_start = time.monotonic()
    window_start = total_start
    window_loss = 0.0
    last_loss_value = float("nan")
    window_micro_batches = 0
    window_input_tokens = 0
    micro_since_update = 0
    last_epoch = epoch
    last_next_batch_index = next_batch
    window_start_step = global_step
    optimizer.zero_grad(set_to_none=True)
    model.train()

    while global_step < args.max_steps:
        sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < next_batch:
                continue
            last_epoch = epoch
            last_next_batch_index = batch_index + 1
            batch = _move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                result = model(
                    batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    image_features=batch.get("image_features"),
                    image_attention_mask=batch.get("image_attention_mask"),
                    return_logits=False,
                )
                loss = result["loss"]
            if loss is None or not bool(torch.isfinite(loss)):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"non-finite training loss at step={global_step}, epoch={epoch}, batch={batch_index}"
                )

            last_loss_value = float(loss.detach().item())
            loss_for_backward = loss / args.gradient_accumulation
            if use_scaler:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
            window_loss += float(loss.detach().item())
            window_micro_batches += 1
            window_input_tokens += int(batch["attention_mask"].sum().item())
            micro_since_update += 1

            end_of_epoch = batch_index == len(train_loader) - 1
            accumulation_complete = micro_since_update >= args.gradient_accumulation
            if not (accumulation_complete or end_of_epoch):
                continue

            if use_scaler:
                scaler.unscale_(optimizer)
            # If the epoch ends with a partial accumulation group, use its actual size
            # rather than underweighting the final micro-batches by 1/gradient_accumulation.
            if micro_since_update < args.gradient_accumulation:
                correction = args.gradient_accumulation / micro_since_update
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not bool(torch.isfinite(grad_norm)):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"non-finite gradient norm at step={global_step}, epoch={epoch}, batch={batch_index}"
                )

            did_update = True
            if use_scaler:
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                did_update = scaler.get_scale() >= old_scale
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_since_update = 0
            if did_update:
                scheduler.step()
                global_step += 1

            if global_step and global_step % args.log_every == 0 and window_micro_batches:
                now = time.monotonic()
                elapsed = max(now - window_start, 1e-9)
                measured_steps = max(1, global_step - window_start_step)
                steps_per_second = measured_steps / elapsed
                mean_loss = window_loss / window_micro_batches
                remaining_seconds = max(0, args.max_steps - global_step) / max(steps_per_second, 1e-9)
                record = {
                    "step": global_step,
                    "train_loss_window": mean_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(grad_norm.detach().float().cpu()),
                    "tokens_per_second_estimate": window_input_tokens / elapsed,
                    "step_seconds_estimate": elapsed / measured_steps,
                    "estimated_linear_train_flops_per_second": (
                        estimated_train_flops_per_token * window_input_tokens / elapsed
                    ),
                    "elapsed_seconds": now - total_start,
                    "eta_seconds": remaining_seconds,
                    "gpu_memory_allocated_bytes": (
                        int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
                    ),
                    "gpu_memory_reserved_bytes": (
                        int(torch.cuda.memory_reserved(device)) if device.type == "cuda" else 0
                    ),
                }
                _write_jsonl(metrics_path, record)
                if wandb_run is not None:
                    try:
                        wandb_run.log(record, step=global_step)
                    except Exception as exc:
                        print(f"W&B log warning: {exc}")
                print(
                    f"step={global_step}/{args.max_steps} loss={mean_loss:.4f} "
                    f"grad={record['gradient_norm']:.3f} tok/s~{record['tokens_per_second_estimate']:.1f} "
                    f"ETA~{remaining_seconds/60:.1f}m"
                )
                window_start, window_loss, window_micro_batches = now, 0.0, 0
                window_input_tokens = 0
                window_start_step = global_step

            if did_update and global_step > 0 and global_step % args.checkpoint_steps == 0:
                eval_metrics: dict[str, float] = {}
                if eval_loader is not None:
                    eval_metrics = evaluate(
                        model, eval_loader, device, amp_enabled, amp_dtype,
                        max_batches=args.eval_max_batches,
                    )
                    print(
                        f"eval step={global_step}: loss={eval_metrics['eval_loss']:.4f}, "
                        f"ppl={eval_metrics['eval_perplexity']:.3f}, "
                        f"acc={eval_metrics['eval_accuracy']:.4f}, "
                        f"depth={eval_metrics['mean_expected_depth']:.2f}"
                    )
                ckpt_metrics = eval_metrics or {"train_loss": float(loss.detach().item())}
                state = {
                    "epoch": epoch,
                    "next_batch_index": batch_index + 1,
                    "seed": args.seed,
                    "run_name": run_name,
                }
                checkpoints.save(
                    step=global_step, model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, training_state=state, metrics=ckpt_metrics,
                )
                if wandb_run is not None and eval_metrics:
                    try:
                        wandb_run.log(eval_metrics, step=global_step)
                    except Exception as exc:
                        print(f"W&B eval log warning: {exc}")

            if global_step >= args.max_steps:
                break

        else:
            epoch += 1
            next_batch = 0
            last_epoch = epoch
            last_next_batch_index = 0
            continue

        if global_step >= args.max_steps:
            break
        epoch += 1
        next_batch = 0
        start_epoch = epoch
        start_step = global_step

    # Always save the final state, even if the max step count is not a checkpoint multiple.
    final_metrics: dict[str, float] = {}
    if eval_loader is not None:
        final_metrics = evaluate(model, eval_loader, device, amp_enabled, amp_dtype,
                                 max_batches=args.eval_max_batches)
    final_metrics = final_metrics or {"train_loss": last_loss_value}
    checkpoints.save(
        step=global_step, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        training_state={
            "epoch": last_epoch,
            "next_batch_index": last_next_batch_index,
            "seed": args.seed,
            "run_name": run_name,
        },
        metrics=final_metrics,
    )
    if wandb_run is not None:
        try:
            wandb_run.log(final_metrics, step=global_step)
            wandb_run.finish()
        except Exception:
            pass
    print(
        f"Finished at step {global_step}; final metrics={json.dumps(final_metrics, ensure_ascii=False)}; "
        f"elapsed={(time.monotonic() - total_start)/3600:.2f} h"
    )


if __name__ == "__main__":
    main()
