"""Fine-tune a small vision-language model from local image/instruction/answer JSONL.

Recommended practical starting point:
  Qwen/Qwen3-VL-2B-Instruct + QLoRA + local, licensed image/text examples.

Each row must include:
  {"image_path": "/absolute/or/relative/local.png",
   "instruction": "What does this figure show?",
   "output": "It shows ...",
   "license": "..."}

This script loads only local image files at training time; remote image URLs are rejected.
It uses batch size 1 to avoid incorrect image-grid padding across different resolutions.
Gradient accumulation provides a larger effective batch. With QLoRA, the final export is an
adapter and must be combined with the exact same base model to use as a complete model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


class VisionInstructionDataset(Dataset):
    """Line-indexed local JSONL dataset; image decoding is deferred to the collator."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.offsets: list[int] = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f"VLM training manifest is empty: {self.path}")
        self._stream = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict:
        if self._stream is None or self._stream.closed:
            self._stream = self.path.open("rb")
        self._stream.seek(self.offsets[index])
        row = json.loads(self._stream.readline())
        for field in ("image_path", "instruction", "output"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{self.path}: row {index} needs non-empty {field}")
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = (self.path.parent / image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"{self.path}: row {index} image file missing: {image_path}")
        if row["image_path"].startswith(("http://", "https://")):
            raise ValueError("VLM SFT requires local image paths; remote URLs are refused")
        row["image_path"] = str(image_path)
        return row


class VLMInstructionCollator:
    """Build Qwen3-VL chat inputs and mask user/image-prefix tokens from the loss."""

    def __init__(self, processor, max_length: int) -> None:
        self.processor = processor
        self.max_length = max_length

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        if len(rows) != 1:
            raise ValueError(
                "This first VLM collator requires batch_size=1. "
                "Use gradient accumulation for a larger effective batch."
            )
        from PIL import Image

        row = rows[0]
        with Image.open(row["image_path"]) as image_file:
            image = image_file.convert("RGB")
            user_content = [{"type": "image"}, {"type": "text", "text": row["instruction"]}]
            prompt_messages = [{"role": "user", "content": user_content}]
            full_messages = prompt_messages + [
                {"role": "assistant", "content": row["output"]}
            ]
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            prompt_inputs = self.processor(
                text=[prompt_text], images=[image], padding=False,
                truncation=False, return_tensors="pt",
            )
            full_inputs = self.processor(
                text=[full_text], images=[image], padding=False,
                truncation=False, return_tensors="pt",
            )
            # Qwen3-VL's documented forward path does not consume tokenizer token_type_ids.
            prompt_inputs.pop("token_type_ids", None)
            full_inputs.pop("token_type_ids", None)

        input_ids = full_inputs["input_ids"]
        if input_ids.shape[1] > self.max_length:
            # Preserve the start containing vision tokens and the end of the response by
            # trimming only text tokens from the middle. Token IDs belonging to image grids
            # cannot be arbitrarily sliced; fail instead of silently corrupting multimodal input.
            raise ValueError(
                f"processed sample has {input_ids.shape[1]} tokens, exceeding --max-length "
                f"{self.max_length}; choose a smaller image resolution or a larger context"
            )
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        if prompt_length >= input_ids.shape[1]:
            raise ValueError(
                "processed prompt leaves no assistant target tokens; shorten instruction or increase context"
            )
        labels = input_ids.clone()
        labels[:, :prompt_length] = -100
        if "attention_mask" in full_inputs:
            labels[full_inputs["attention_mask"] == 0] = -100
        full_inputs["labels"] = labels
        return dict(full_inputs)


class VLMCheckpointCallback:
    """Use the shared HF checkpoint mirrors so latest and step-N paths are available."""

    def __new__(cls, root: Path):
        from mylm_omni.training.train_hf_sft import ResearchCheckpointCallback
        return ResearchCheckpointCallback(root, keep_steps=3, keep_best=3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", default=None)
    parser.add_argument("--output-dir", default="runs/mylm-omni-vlm")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mylm-omni")
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    args = parser.parse_args()
    if args.max_length < 64 or args.max_steps < 1 or args.gradient_accumulation < 1:
        parser.error("max-length must be >= 64; max-steps and gradient-accumulation must be positive")
    if args.save_steps < 1 or args.log_steps < 1:
        parser.error("save-steps and log-steps must be positive")

    try:
        from transformers import (
            AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig,
            Trainer, TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install the multimodal extra: pip install -e '.[multimodal]'"
        ) from exc
    from transformers.trainer_utils import get_last_checkpoint
    import inspect
    from datetime import datetime
    from zoneinfo import ZoneInfo

    root = Path(args.output_dir)
    hf_dir = root / "hf"
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=False)

    compute_dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )
    model_kwargs = {"trust_remote_code": False, "low_cpu_mem_usage": True}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = compute_dtype
        model_kwargs["attn_implementation"] = "sdpa"
    if args.qlora:
        if device.type != "cuda":
            raise RuntimeError("QLoRA requires a CUDA GPU")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}

    if "Qwen3-VL" not in args.base_model:
        raise ValueError(
            "This collator is currently specialized for Qwen3-VL chat/image processing. "
            "Use --base-model Qwen/Qwen3-VL-2B-Instruct or implement and test a model-specific collator."
        )
    # Use the model-specific documented class, supported by Transformers releases with Qwen3-VL.
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.base_model, **model_kwargs)
    model.config.use_cache = False
    if args.qlora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        # Qwen3-VL language-side projections only. Vision tower remains frozen for this phase.
        language_lora_pattern = (
            r"^model\.language_model\.layers\.\d+\."
            r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
            r"mlp\.(gate_proj|up_proj|down_proj))$"
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=language_lora_pattern,
            ),
        )
    else:
        model.gradient_checkpointing_enable()

    train_data = VisionInstructionDataset(args.train_jsonl)
    eval_data = VisionInstructionDataset(args.eval_jsonl) if args.eval_jsonl else None
    collator = VLMInstructionCollator(processor, args.max_length)
    run_name = (
        f"MyLM-omni-{model.num_parameters()/1e9:.2f}B-reasoning-"
        f"{datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d-%H%M%S')}"
    )
    hf_dir.mkdir(parents=True, exist_ok=True)
    arg_values = dict(
        output_dir=str(hf_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_steps=args.log_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        save_safetensors=True,
        bf16=bool(device.type == "cuda" and compute_dtype == torch.bfloat16),
        fp16=bool(device.type == "cuda" and compute_dtype == torch.float16),
        gradient_checkpointing=not args.qlora,
        dataloader_num_workers=0,
        report_to=["wandb"] if args.wandb else "none",
        run_name=run_name,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        load_best_model_at_end=eval_data is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        arg_values["eval_strategy"] = "steps" if eval_data is not None else "no"
    else:
        arg_values["evaluation_strategy"] = "steps" if eval_data is not None else "no"
    training_args = TrainingArguments(**arg_values)
    callbacks = [VLMCheckpointCallback(root)]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
        processing_class=processor,
        callbacks=callbacks,
    )
    resume_path = get_last_checkpoint(str(hf_dir)) if args.resume == "auto" else None
    print(f"Run: {run_name}")
    print(f"Base model: {args.base_model} | QLoRA: {args.qlora} | Device: {device}")
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Training rows: {len(train_data)} | Eval rows: {len(eval_data) if eval_data else 0}")
    print(f"Auto-resume: {resume_path or 'none'}")
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(str(root / "final"))
    processor.save_pretrained(root / "final")
    print(f"Saved final model/adapter and processor under {root / 'final'}")


if __name__ == "__main__":
    main()
