"""Online full-logit TAID fine-tuning for a compact student causal LM.

Use this when teacher and student have the same tokenizer/vocabulary. A frozen teacher provides
full next-token logits on the student's teacher-forced contexts; TAID interpolates detached
student logits with teacher logits and progresses toward the teacher distribution during training.

This implementation is text-only. The Qwen3-VL-specific supervised path lives in train_vlm_sft.py;
online full-logit VLM TAID is a separate extension because multimodal teacher/student inputs and
vision encoders must also be compatible.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from mylm_omni.training.distillation import TAIDSchedule, taid_loss
from mylm_omni.training.train_hf_sft import InstructionJsonlDataset, SFTCollator, ResearchCheckpointCallback


def _load_taid_state(checkpoint: str | None, schedule: TAIDSchedule) -> None:
    if not checkpoint:
        return
    state_path = Path(checkpoint) / "taid_state.json"
    if not state_path.is_file():
        return
    schedule.load_state_dict(json.loads(state_path.read_text(encoding="utf-8")))
    print(f"Restored TAID schedule from {state_path}: t={schedule.t:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--teacher-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl", default=None)
    parser.add_argument("--output-dir", default="runs/mylm-omni-taid")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--distil-ratio", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--t-start", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--taid-alpha", type=float, default=5e-4)
    parser.add_argument("--taid-beta", type=float, default=0.99)
    parser.add_argument("--disable-adaptive-taid", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mylm-omni")
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    args = parser.parse_args()

    if args.max_steps < 1 or args.max_length < 32 or args.gradient_accumulation < 1:
        parser.error("max-steps, gradient-accumulation must be positive and max-length >= 32")
    if args.save_steps < 1 or args.log_steps < 1 or args.temperature <= 0:
        parser.error("save/log steps and temperature must be positive")
    if not 0.0 <= args.distil_ratio <= 1.0:
        parser.error("distil-ratio must be in [0, 1]")

    import inspect
    from transformers import TrainingArguments

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.qlora and device.type != "cuda":
        raise RuntimeError("QLoRA requires CUDA")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    compute_dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )

    # Tokenizers must match, not just language/model names. Otherwise token IDs/logits cannot
    # be compared element-wise. Model logits dimensions are checked after both are loaded.
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student_model, use_fast=True, trust_remote_code=False
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model, use_fast=True, trust_remote_code=False
    )
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("TAID requires identical student and teacher tokenizers")
    if student_tokenizer.eos_token_id is None:
        raise ValueError("student tokenizer must define EOS")
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    model_kwargs = {"trust_remote_code": False, "low_cpu_mem_usage": True}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = compute_dtype
        model_kwargs["attn_implementation"] = "sdpa"
    if args.qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
    student = AutoModelForCausalLM.from_pretrained(args.student_model, **model_kwargs)
    student.config.use_cache = False
    if args.qlora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        student = prepare_model_for_kbit_training(student, use_gradient_checkpointing=True)
        student = get_peft_model(
            student,
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
    else:
        student.gradient_checkpointing_enable()

    # Keep teacher frozen and in eval mode. On a 24GB RTX 3090 this can still be tight for long
    # contexts, so start at 512–768 tokens; use smaller teacher or Colab 80/96GB if needed.
    teacher_kwargs = {"trust_remote_code": False, "low_cpu_mem_usage": True}
    if device.type == "cuda":
        teacher_kwargs["torch_dtype"] = compute_dtype
        teacher_kwargs["attn_implementation"] = "sdpa"
        teacher_kwargs["device_map"] = {"": torch.cuda.current_device()}
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, **teacher_kwargs).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if int(student.config.vocab_size) != int(teacher.config.vocab_size):
        raise ValueError(
            f"model output vocabulary sizes differ: {student.config.vocab_size} vs "
            f"{teacher.config.vocab_size}"
        )

    train_data = InstructionJsonlDataset(args.train_jsonl, student_tokenizer, args.max_length)
    eval_data = (
        InstructionJsonlDataset(args.eval_jsonl, student_tokenizer, args.max_length)
        if args.eval_jsonl else None
    )
    output_root = Path(args.output_dir)
    hf_dir = output_root / "hf"
    output_root.mkdir(parents=True, exist_ok=True)
    schedule = TAIDSchedule(
        t_start=args.t_start, t_end=args.t_end, alpha=args.taid_alpha,
        beta=args.taid_beta, disable_adaptive=args.disable_adaptive_taid,
    )
    resume_path = get_last_checkpoint(str(hf_dir)) if args.resume == "auto" else None
    _load_taid_state(resume_path, schedule)

    class OnlineTAIDTrainer(Trainer):
        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self._last_schedule_step = int(self.state.global_step)
            self._last_kd_loss: float | None = None
            self._last_log_values: dict[str, float] = {}

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            inputs.pop("token_type_ids", None)
            labels = inputs["labels"]
            attention_mask = inputs.get("attention_mask")
            student_outputs = model(**inputs, use_cache=False)
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=inputs["input_ids"],
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            if student_outputs.logits.shape != teacher_outputs.logits.shape:
                raise ValueError(
                    "teacher/student logits shapes differ; cannot use token-level TAID"
                )
            # Advance schedule once per completed optimizer step using the previous step's
            # KD loss, not on every gradient-accumulation micro-batch.
            current_step = int(self.state.global_step)
            if current_step > self._last_schedule_step and self._last_kd_loss is not None:
                schedule.update(
                    self._last_kd_loss,
                    global_step=current_step,
                    total_steps=args.max_steps,
                )
                self._last_schedule_step = current_step

            target_mask = labels[:, 1:].ne(-100)
            student_next = student_outputs.logits[:, :-1, :]
            teacher_next = teacher_outputs.logits[:, :-1, :]
            kd = taid_loss(
                student_next,
                teacher_next,
                target_mask,
                t=schedule.t,
                temperature=args.temperature,
            )
            sft = student_outputs.loss
            loss = (1.0 - args.distil_ratio) * sft + args.distil_ratio * kd
            self._last_kd_loss = float(kd.detach().float().cpu())
            self._last_log_values = {
                "taid_loss": self._last_kd_loss,
                "sft_loss": float(sft.detach().float().cpu()),
                "taid_t": float(schedule.t),
            }
            return (loss, student_outputs) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            if "loss" in logs and self._last_log_values:
                logs = {**logs, **self._last_log_values}
            return super().log(logs, *args, **kwargs)

    class TAIDStateCallback:
        """Persist TAID schedule state after Trainer has written each step checkpoint."""
        def __new__(cls, schedule_state: TAIDSchedule):
            from transformers import TrainerCallback
            schedule_object = schedule_state

            class Callback(TrainerCallback):
                def on_save(self, args, state, control, **kwargs):
                    checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                    if checkpoint.exists():
                        (checkpoint / "taid_state.json").write_text(
                            json.dumps(schedule_object.state_dict(), ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    return control
            return Callback()

    run_name = (
        f"MyLM-omni-{student.num_parameters()/1e9:.2f}B-reasoning-"
        f"{datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d-%H%M%S')}"
    )
    arg_values = dict(
        output_dir=str(hf_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
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
        gradient_checkpointing=False if args.qlora else True,
        dataloader_num_workers=args.num_workers,
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
    trainer = OnlineTAIDTrainer(
        model=student,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=SFTCollator(student_tokenizer),
        processing_class=student_tokenizer,
        # Write the schedule into the original Trainer checkpoint before the mirror callback
        # copies it into steps/ and latest/.
        callbacks=[
            TAIDStateCallback(schedule),
            ResearchCheckpointCallback(output_root, keep_steps=3, keep_best=3),
        ],
    )
    print(f"Run: {run_name}")
    print(f"Student: {args.student_model} | Teacher: {args.teacher_model}")
    print(
        f"Student params: {student.num_parameters():,} | Trainable: "
        f"{sum(p.numel() for p in student.parameters() if p.requires_grad):,}"
    )
    print(f"Teacher frozen: yes | Device: {device} | dtype: {compute_dtype}")
    print(f"TAID: t={schedule.t:.3f} -> {schedule.t_end:.3f} | Distill ratio={args.distil_ratio}")
    print(f"Training rows: {len(train_data)} | Eval rows: {len(eval_data) if eval_data else 0}")
    print(f"Auto-resume: {resume_path or 'none'}")
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(str(output_root / "final"))
    student_tokenizer.save_pretrained(output_root / "final")
    (output_root / "final" / "taid_state.json").write_text(
        json.dumps(schedule.state_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved student/adapter and TAID state to {output_root / 'final'}")


if __name__ == "__main__":
    main()
