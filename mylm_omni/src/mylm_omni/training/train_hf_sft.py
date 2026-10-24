"""Fine-tune a licensed Hugging Face causal LM on instruction/output JSONL.

Accepted records:
  {"instruction": "...", "output": "..."}
  {"instruction": "...", "input": "...", "output": "..."}
  {"text": "..."}  # language-model continuation; all tokens contribute to loss

This path is recommended for practical behavior on one RTX 3090. It does not train the custom
scratch architecture or add vision features to a text-only base checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import Dataset


def create_run_name(parameter_count: int) -> str:
    """Create the required run ID with parameter size and Japan Standard Time."""
    if parameter_count >= 1_000_000_000:
        count = f"{parameter_count / 1_000_000_000:.2f}B"
    elif parameter_count >= 1_000_000:
        count = f"{parameter_count / 1_000_000:.1f}M"
    else:
        count = str(parameter_count)
    timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d-%H%M%S")
    return f"MyLM-omni-{count}-reasoning-{timestamp}"


class InstructionJsonlDataset(Dataset):
    """Prepare examples with prompt tokens masked so the loss targets assistant output."""

    def __init__(self, path: str, tokenizer, max_length: int) -> None:
        self.rows: list[dict[str, list[int]]] = []
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer must define eos_token_id")
        with open(path, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if "instruction" in record and "output" in record:
                    context = str(record["instruction"])
                    extra = str(record.get("input", "")).strip()
                    prompt = context if not extra else f"{context}\n\nInput:\n{extra}"
                    target = str(record["output"])
                    if getattr(tokenizer, "chat_template", None):
                        prompt_ids = tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=True, add_generation_prompt=True,
                        )
                    else:
                        prompt = f"### Instruction\n{prompt}\n\n### Response\n"
                        prompt_ids = tokenizer(
                            prompt, add_special_tokens=True, truncation=False
                        )["input_ids"]
                    # Preserve the end of long prompts and reserve at least one output position.
                    prompt_ids = prompt_ids[-max(1, max_length - 1):]
                    budget = max_length - len(prompt_ids)
                    target_ids = tokenizer(
                        target, add_special_tokens=False, truncation=False
                    )["input_ids"][:budget]
                    if len(target_ids) < budget:
                        target_ids.append(eos_id)
                    if not target_ids:
                        target_ids = [eos_id]
                        prompt_ids = prompt_ids[-(max_length - 1):]
                    input_ids = prompt_ids + target_ids[: max_length - len(prompt_ids)]
                    labels = [-100] * len(prompt_ids) + target_ids[: max_length - len(prompt_ids)]
                elif "text" in record:
                    input_ids = tokenizer(
                        str(record["text"]), add_special_tokens=True,
                        truncation=True, max_length=max_length
                    )["input_ids"]
                    labels = list(input_ids)
                else:
                    raise ValueError(
                        f"{path}:{line_number} needs instruction/output or text fields"
                    )
                if len(input_ids) < 2 or len(labels) != len(input_ids):
                    continue
                self.rows.append({"input_ids": input_ids, "labels": labels})
        if not self.rows:
            raise ValueError(f"no valid training rows found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class SFTCollator:
    """Dynamic right padding; loss ignores padding and masked prompt tokens."""

    def __init__(self, tokenizer) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            # A tokenizer may omit PAD for causal LM; EOS is a safe padding symbol when masked.
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, rows: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in rows)
        ids = torch.full((len(rows), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
        mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for index, row in enumerate(rows):
            length = len(row["input_ids"])
            ids[index, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
            labels[index, :length] = torch.tensor(row["labels"], dtype=torch.long)
            mask[index, :length] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _mirror_checkpoint(source: Path, target: Path) -> None:
    """Build a temporary hard-link mirror and swap it into place with recovery."""
    temporary = target.with_name(target.name + ".tmp")
    previous = target.with_name(target.name + ".prev")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=True)
    try:
        for file_path in source.iterdir():
            if file_path.is_file():
                _link_or_copy(file_path, temporary / file_path.name)
        shutil.rmtree(previous, ignore_errors=True)
        if target.exists():
            os.replace(target, previous)
        try:
            os.replace(temporary, target)
        except Exception:
            if previous.exists() and not target.exists():
                os.replace(previous, target)
            raise
        shutil.rmtree(previous, ignore_errors=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


class ResearchCheckpointCallback:
    """Create steps/step-N and latest/ mirrors of safe Trainer checkpoints."""

    def __init__(self, root: Path, keep_steps: int = 3, keep_best: int = 3) -> None:
        self.root = root
        self.keep_steps = keep_steps
        self.keep_best = keep_best
        self.best: list[dict] = []
        index_path = root / "best" / "index.json"
        if index_path.is_file():
            try:
                self.best = json.loads(index_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self.best = []
        self.steps_dir = root / "steps"
        self.best_dir = root / "best"
        self.steps_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)

    def as_callback(self):
        from transformers import TrainerCallback

        outer = self

        class Callback(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                step = int(state.global_step)
                source = Path(args.output_dir) / f"checkpoint-{step}"
                if not source.exists():
                    return control
                step_dir = outer.steps_dir / f"step-{step:06d}"
                _mirror_checkpoint(source, step_dir)
                _mirror_checkpoint(step_dir, outer.root / "latest")
                outer._update_best(step, state, step_dir)
                outer._prune()
                manifest = {
                    "latest_step": step,
                    "path": str(Path("steps") / step_dir.name),
                    "updated_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
                }
                (outer.root / "manifest.json.tmp").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                os.replace(outer.root / "manifest.json.tmp", outer.root / "manifest.json")
                return control

        return Callback()

    def _update_best(self, step: int, state, step_dir: Path) -> None:
        latest_metrics = next(
            (item for item in reversed(state.log_history) if "eval_loss" in item), {}
        )
        if "eval_loss" not in latest_metrics:
            return
        self.best = [item for item in self.best if item["step"] != step]
        self.best.append({"step": step, "eval_loss": float(latest_metrics["eval_loss"])})
        self.best.sort(key=lambda item: item["eval_loss"])
        self.best = self.best[: self.keep_best]
        for rank in range(1, self.keep_best + 1):
            shutil.rmtree(self.best_dir / f"rank-{rank}", ignore_errors=True)
        for rank, entry in enumerate(self.best, start=1):
            source = self.steps_dir / f"step-{int(entry['step']):06d}"
            if not source.exists():
                continue
            target = self.best_dir / f"rank-{rank}"
            target.mkdir(parents=True)
            # Copy inference metadata as well as weights. Do not copy optimizer state, RNG
            # state, or training arguments into the compact best-inference snapshot.
            inference_names = {
                "model.safetensors", "pytorch_model.bin",
                "adapter_model.safetensors", "adapter_config.json",
                "config.json", "generation_config.json",
                "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "added_tokens.json", "vocab.json", "merges.txt", "spiece.model",
                "tokenizer.model", "preprocessor_config.json", "processor_config.json",
                "chat_template.json", "chat_template.jinja", "video_preprocessor_config.json",
                "image_processor_config.json",
            }
            for source_file in source.iterdir():
                if source_file.is_file() and source_file.name in inference_names:
                    _link_or_copy(source_file, target / source_file.name)
            (target / "manifest.json").write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        (self.best_dir / "index.json").write_text(
            json.dumps(self.best, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _prune(self) -> None:
        dirs = sorted(
            (p for p in self.steps_dir.glob("step-*") if p.is_dir()),
            key=lambda p: int(p.name.split("-")[-1]), reverse=True,
        )
        keep = {p for p in dirs[: self.keep_steps]}
        keep.update(
            self.steps_dir / f"step-{entry['step']:06d}" for entry in self.best
        )
        for path in dirs:
            if path not in keep:
                shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", default=None)
    parser.add_argument("--output-dir", default="runs/mylm-omni-sft")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--qlora", action="store_true",
                        help="4-bit NF4 base weights plus trainable LoRA adapters (recommended for RTX 3090)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mylm-omni")
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    args = parser.parse_args()

    if args.max_length < 32 or args.max_steps < 1:
        parser.error("max-length must be >= 32 and max-steps must be positive")

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    root = Path(args.output_dir)
    hf_dir = root / "hf"
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Trust remote code is intentionally false; do not enable it to solve a loading issue blindly.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"trust_remote_code": False, "low_cpu_mem_usage": True}
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = compute_dtype
        model_kwargs["attn_implementation"] = "sdpa"
    if args.qlora:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires a CUDA GPU in this script")
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False
    if args.qlora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            ),
        )
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_data = InstructionJsonlDataset(args.train_jsonl, tokenizer, args.max_length)
    eval_data = (
        InstructionJsonlDataset(args.eval_jsonl, tokenizer, args.max_length)
        if args.eval_jsonl else None
    )
    run_name = create_run_name(model.num_parameters())
    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)
    arg_values = dict(
        output_dir=str(hf_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_steps=args.log_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        save_total_limit=3,
        save_safetensors=True,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to=["wandb"] if args.wandb else "none",
        run_name=run_name,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        load_best_model_at_end=eval_data is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    # Compatibility with both recent Transformers (eval_strategy) and older 4.x releases.
    import inspect
    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        arg_values["eval_strategy"] = "steps" if eval_data is not None else "no"
    else:
        arg_values["evaluation_strategy"] = "steps" if eval_data is not None else "no"
    training_args = TrainingArguments(**arg_values)

    callback = ResearchCheckpointCallback(root, keep_steps=3, keep_best=3)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=SFTCollator(tokenizer),
        processing_class=tokenizer,
        callbacks=[callback.as_callback()],
    )
    resume_path = None
    if args.resume == "auto":
        resume_path = get_last_checkpoint(str(hf_dir))
        if resume_path is None and (root / "latest" / "trainer_state.json").exists():
            resume_path = str(root / "latest")

    print(f"Run: {run_name}")
    print(f"Base model: {args.base_model}")
    print(f"Parameters: {model.num_parameters():,}")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    print(f"QLoRA: {args.qlora}")
    print(f"Train rows: {len(train_data):,}; eval rows: {len(eval_data) if eval_data else 0}")
    print(f"Auto-resume: {resume_path or 'none'}")
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(str(root / "final"))
    tokenizer.save_pretrained(str(root / "final"))
    print(f"Model and tokenizer saved to {root / 'final'}")


if __name__ == "__main__":
    main()
