"""Validated teacher trace schema for knowledge distillation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class TeacherTrace:
    """One completion with per-token teacher top-k logits and aggregated tail mass.

    topk_logits and tail_logsumexp must use the *same pre-scaled logit temperature*.
    A top-k list alone is not the full teacher distribution; tail_logsumexp is an
    approximation-preserving summary that lets the student treat omitted classes as a
    single probability bucket. This bucket still loses distinctions among tail classes.
    """

    teacher_model: str
    teacher_revision: str
    tokenizer_name: str
    tokenizer_revision: str
    prompt_ids: list[int]
    answer_ids: list[int]
    topk_ids: list[list[int]]
    topk_logits: list[list[float]]
    tail_logsumexp: list[float]
    temperature: float = 1.0
    task_type: str = "instruction"
    modality: str = "text"
    data_source: str = "unknown"
    license_id: str = "unknown"
    quality_score: float = 0.0
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        if not self.teacher_model or not self.teacher_revision:
            raise ValueError("teacher model and immutable revision are required")
        if not self.tokenizer_name or not self.tokenizer_revision:
            raise ValueError("tokenizer name and immutable revision are required")
        if not self.answer_ids:
            raise ValueError("answer_ids must not be empty")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not 0 <= self.quality_score <= 1:
            raise ValueError("quality_score must be in [0, 1]")
        lengths = (len(self.topk_ids), len(self.topk_logits), len(self.tail_logsumexp))
        if any(length != len(self.answer_ids) for length in lengths):
            raise ValueError(
                "top-k ids, logits, and tail mass must have one row per answer token; "
                "do not silently pad missing teacher traces"
            )
        for token_ids, token_logits in zip(self.topk_ids, self.topk_logits):
            if not token_ids or len(token_ids) != len(token_logits):
                raise ValueError("each top-k id row must match its logit row and be non-empty")
            if len(set(token_ids)) != len(token_ids):
                raise ValueError("top-k token IDs must be unique within a row")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def append_jsonl(path: str | Path, trace: TeacherTrace) -> None:
    """Append one validated record; create parent directories when needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> Iterable[TeacherTrace]:
    """Yield validated trace records one line at a time."""
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                trace = TeacherTrace(**json.loads(line))
                trace.validate()
            except Exception as exc:
                raise ValueError(f"invalid teacher trace at {path}:{line_number}: {exc}") from exc
            yield trace
