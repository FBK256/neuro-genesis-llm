"""Train a causal LM from sparse top-k teacher traces.

This is the compressed off-line KD path. The sparse loss preserves teacher probability mass in
top-k classes plus an aggregated tail bucket, but cannot reconstruct individual tail logits.
If --eval-jsonl is supplied, it must be a *separate held-out teacher-trace JSONL* generated with
the same tokenizer, teacher revision, temperature and top-k. Use online train_taid.py for TAID.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from mylm_omni.training.distillation import sparse_topk_tail_kl


class SparseTraceDataset(Dataset):
    """Index validated teacher trace JSONL without keeping all token arrays in RAM."""

    def __init__(self, path: str | Path) -> None:
        from mylm_omni.training.teacher_data import TeacherTrace

        self.path = Path(path)
        self.offsets: list[int] = []
        self.tokenizer_name: str | None = None
        self.temperature: float | None = None
        self.teacher_revision: str | None = None
        self.tokenizer_revision: str | None = None
        self.top_k: int | None = None
        self.vocab_size: int | None = None
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = TeacherTrace(**json.loads(line))
                row.validate()
                if not self.tokenizer_name:
                    self.tokenizer_name = row.tokenizer_name
                    self.temperature = row.temperature
                    self.teacher_revision = row.teacher_revision
                    self.tokenizer_revision = row.tokenizer_revision
                    self.top_k = len(row.topk_ids[0])
                    self.vocab_size = int((row.metadata or {}).get("vocab_size", 0))
                if row.tokenizer_name != self.tokenizer_name:
                    raise ValueError("all traces must use the same tokenizer")
                if row.temperature != self.temperature:
                    raise ValueError("all traces in one dataset must use the same temperature")
                if row.teacher_revision != self.teacher_revision:
                    raise ValueError("all traces must use the same immutable teacher revision")
                if row.tokenizer_revision != self.tokenizer_revision:
                    raise ValueError("all traces must use the same immutable tokenizer revision")
                if len(row.topk_ids[0]) != self.top_k:
                    raise ValueError("all traces must use the same top-k size")
                row_vocab = int((row.metadata or {}).get("vocab_size", 0))
                if self.vocab_size and row_vocab and row_vocab != self.vocab_size:
                    raise ValueError("teacher trace vocabulary sizes differ")
                self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f"no teacher traces in {self.path}")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            row = json.loads(stream.readline())
        return row


class SparseTraceCollator:
    """Pad the student sequence and teacher arrays while retaining answer offsets."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        lengths = [len(row["prompt_ids"]) + len(row["answer_ids"]) for row in rows]
        max_seq = max(lengths)
        max_answer = max(len(row["answer_ids"]) for row in rows)
        top_k = len(rows[0]["topk_ids"][0])
        batch = len(rows)
        input_ids = torch.full((batch, max_seq), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, max_seq), dtype=torch.long)
        labels = torch.full((batch, max_seq), -100, dtype=torch.long)
        prompt_lengths = torch.tensor([len(row["prompt_ids"]) for row in rows], dtype=torch.long)
        answer_lengths = torch.tensor([len(row["answer_ids"]) for row in rows], dtype=torch.long)
        topk_ids = torch.zeros((batch, max_answer, top_k), dtype=torch.long)
        topk_logits = torch.zeros((batch, max_answer, top_k), dtype=torch.float32)
        tail_lse = torch.zeros((batch, max_answer), dtype=torch.float32)
        for i, row in enumerate(rows):
            prompt = row["prompt_ids"]
            answer = row["answer_ids"]
            sequence = prompt + answer
            if len(sequence) < 2:
                raise ValueError("trace sequence must contain at least two token IDs")
            n = len(sequence)
            p = len(prompt)
            a = len(answer)
            if len(row["topk_ids"]) != a or len(row["topk_logits"]) != a or len(row["tail_logsumexp"]) != a:
                raise ValueError("teacher top-k rows do not match answer length")
            if any(len(ids) != top_k for ids in row["topk_ids"]):
                raise ValueError("top-k vocabulary width differs across trace rows")
            if any(len(logits) != top_k for logits in row["topk_logits"]):
                raise ValueError("teacher logit width differs across trace rows")
            input_ids[i, :n] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[i, :n] = 1
            labels[i, p:n] = torch.tensor(answer, dtype=torch.long)
            topk_ids[i, :a] = torch.tensor(row["topk_ids"], dtype=torch.long)
            topk_logits[i, :a] = torch.tensor(row["topk_logits"], dtype=torch.float32)
            tail_lse[i, :a] = torch.tensor(row["tail_logsumexp"], dtype=torch.float32)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_lengths": prompt_lengths,
            "answer_lengths": answer_lengths,
            "topk_ids": topk_ids,
            "topk_logits": topk_logits,
            "tail_logsumexp": tail_lse,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--student-revision", default="main")
    parser.add_argument("--teacher-traces", required=True)
    parser.add_argument("--eval-jsonl", default=None,
                        help="optional held-out teacher-trace JSONL from the same teacher/tokenizer/temperature")
    parser.add_argument("--output-dir", default="runs/mylm-omni-sparse-kd")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--distil-ratio", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--qlora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mylm-omni")
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_length < 2 or args.gradient_accumulation < 1:
        parser.error("max-steps, max-length, and gradient-accumulation must be positive")
    if not 0.0 <= args.distil_ratio <= 1.0:
        parser.error("distil-ratio must be between 0 and 1")

    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint
    import inspect
    from datetime import datetime
    from zoneinfo import ZoneInfo

    data = SparseTraceDataset(args.teacher_traces)
    tokenizer_name = data.tokenizer_name
    if not tokenizer_name:
        raise ValueError("teacher traces must identify tokenizer_name")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, revision=data.tokenizer_revision,
        use_fast=True, trust_remote_code=False
    )
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student_model, revision=args.student_revision,
        use_fast=True, trust_remote_code=False
    )
    if tokenizer.get_vocab() != student_tokenizer.get_vocab():
        raise ValueError(
            "student tokenizer differs from teacher trace tokenizer; token IDs/logits are incompatible"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )
    model_kwargs = {"trust_remote_code": False, "low_cpu_mem_usage": True}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = dtype
        model_kwargs["attn_implementation"] = "sdpa"
    if args.qlora:
        if device.type != "cuda":
            raise RuntimeError("QLoRA requires CUDA")
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(
        args.student_model, revision=args.student_revision, **model_kwargs
    )
    if int(model.config.vocab_size) != int(tokenizer.vocab_size):
        # Many tokenizers include newly-added special tokens while model logits keep a padded
        # vocabulary. Top-k labels remain compatible as long as their IDs fit and semantics align.
        max_id = max(tokenizer.get_vocab().values())
        if max_id >= int(model.config.vocab_size):
            raise ValueError("tokenizer contains token IDs outside student logits vocabulary")
    if int(model.config.vocab_size) < 2:
        raise ValueError("invalid student vocabulary size")
    if args.qlora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias="none", task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            ),
        )
    else:
        model.gradient_checkpointing_enable()

    train_data = data
    eval_data = SparseTraceDataset(args.eval_jsonl) if args.eval_jsonl else None
    if eval_data is not None:
        if eval_data.tokenizer_name != data.tokenizer_name:
            raise ValueError("train/eval traces use different tokenizers")
        if eval_data.temperature != data.temperature:
            raise ValueError("train/eval traces use different teacher temperatures")
        if eval_data.teacher_revision != data.teacher_revision:
            raise ValueError("train/eval traces use different teacher revisions")
        if eval_data.tokenizer_revision != data.tokenizer_revision:
            raise ValueError("train/eval traces use different tokenizer revisions")
        if eval_data.top_k != data.top_k:
            raise ValueError("train/eval traces use different top-k widths")

    from mylm_omni.training.train_hf_sft import ResearchCheckpointCallback
    output_root = Path(args.output_dir)
    hf_dir = output_root / "hf"
    hf_dir.mkdir(parents=True, exist_ok=True)

    # The teacher temperature is already applied to stored top-k logits and tail logsumexp.
    temperature = float(data.temperature or 1.0)

    class SparseKDTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            prompt_lengths = inputs.pop("prompt_lengths").tolist()
            answer_lengths = inputs.pop("answer_lengths").tolist()
            topk_ids_batch = inputs.pop("topk_ids")
            teacher_logits_batch = inputs.pop("topk_logits")
            tail_batch = inputs.pop("tail_logsumexp")
            student_outputs = model(**inputs, use_cache=False)
            student_logits = student_outputs.logits
            selected_student = []
            selected_ids = []
            selected_teacher = []
            selected_tail = []
            for i, (prompt_len, answer_len) in enumerate(zip(prompt_lengths, answer_lengths)):
                start = max(0, prompt_len - 1)
                stop = start + answer_len
                if prompt_len < 1 or stop > student_logits.shape[1] - 0:
                    raise ValueError("trace offsets exceed student model input sequence")
                selected_student.append(student_logits[i, start:stop, :])
                selected_ids.append(topk_ids_batch[i, :answer_len])
                selected_teacher.append(teacher_logits_batch[i, :answer_len])
                selected_tail.append(tail_batch[i, :answer_len])
            flat_student = torch.cat(selected_student, dim=0)
            flat_ids = torch.cat(selected_ids, dim=0)
            flat_teacher = torch.cat(selected_teacher, dim=0)
            flat_tail = torch.cat(selected_tail, dim=0)
            kd_loss = sparse_topk_tail_kl(
                flat_student, flat_ids, flat_teacher, flat_tail, temperature=temperature
            )
            lm_loss = student_outputs.loss
            loss = (1.0 - args.distil_ratio) * lm_loss + args.distil_ratio * kd_loss
            self._last_sparse_kd = {
                "sparse_kd_loss": float(kd_loss.detach().cpu()),
                "sft_loss": float(lm_loss.detach().cpu()),
                "distil_ratio": args.distil_ratio,
            }
            return (loss, student_outputs) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            if "loss" in logs and hasattr(self, "_last_sparse_kd"):
                logs = {**logs, **self._last_sparse_kd}
            return super().log(logs, *args, **kwargs)

    dataset = train_data
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    )
    collator = SparseTraceCollator(pad_token_id)
    # Trainer can only forward standard model inputs; custom metadata is removed in compute_loss.
    run_name = (
        f"MyLM-omni-{model.num_parameters()/1e9:.2f}B-reasoning-"
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
        bf16=bool(device.type == "cuda" and dtype == torch.bfloat16),
        fp16=bool(device.type == "cuda" and dtype == torch.float16),
        gradient_checkpointing=not args.qlora,
        dataloader_num_workers=0,
        report_to=["wandb"] if args.wandb else "none",
        run_name=run_name,
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )
    if "eval_strategy" in inspect.signature(TrainingArguments).parameters:
        arg_values["eval_strategy"] = "steps" if eval_data is not None else "no"
    else:
        arg_values["evaluation_strategy"] = "steps" if eval_data is not None else "no"
    training_args = TrainingArguments(**arg_values)
    trainer = SparseKDTrainer(
        model=model, args=training_args, train_dataset=dataset, eval_dataset=eval_data,
        data_collator=collator, processing_class=tokenizer,
        callbacks=[ResearchCheckpointCallback(output_root, keep_steps=3, keep_best=3)],
    )
    resume = get_last_checkpoint(str(hf_dir)) if args.resume == "auto" else None
    print(
        f"Run: {run_name} | student={args.student_model}@{args.student_revision} | "
        f"qlora={args.qlora}"
    )
    print(
        f"Teacher tokenizer={tokenizer_name}, revision={data.teacher_revision}, "
        f"temperature={temperature}, top-k={data.top_k}, rows={len(data)}"
    )
    print(f"Auto-resume={resume or 'none'}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_root / "final"))
    tokenizer.save_pretrained(output_root / "final")
    print(f"Saved sparse KD student/adapter to {output_root / 'final'}")


if __name__ == "__main__":
    main()
