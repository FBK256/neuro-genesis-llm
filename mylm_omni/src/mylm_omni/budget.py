"""Deterministic task difficulty routing for compute-budget allocation.

This is a transparent bootstrap heuristic, not a learned difficulty head. It provides a baseline
whose decisions can be measured and later replaced by a calibrated classifier trained on verified
task outcomes. It never changes tool permissions or bypasses verification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskBudget:
    mode: str
    max_steps: int
    max_new_tokens: int
    rationale: tuple[str, ...]


_DEEP_CUES = (
    "debug", "fix", "implement", "build", "research", "compare", "analyze", "analyse",
    "multiple", "several", "step by step", "verify", "test", "review", "investigate",
    "調査", "比較", "分析", "実装", "修正", "デバッグ", "作成", "検証", "テスト",
    "レビュー", "複数", "段階", "原因", "改善", "整理",
)
_FILE_CUES = (
    "file", "folder", "workspace", "read", "write", "save", "create", "inspect",
    "ファイル", "フォルダ", "作業フォルダ", "読み", "書き", "保存", "作成", "調べ",
)
_SHORT_ANSWER_CUES = (
    "what is", "define", "translate", "summarize", "意味", "翻訳", "要約", "説明して",
)


def choose_task_budget(
    goal: str,
    *,
    has_image: bool = False,
    has_audio: bool = False,
    has_workspace_tools: bool = True,
    mode: str = "auto",
    max_allowed_steps: int = 12,
    max_allowed_tokens: int = 768,
) -> TaskBudget:
    """Pick a reflex/deliberative/deep budget under hard caller-supplied upper limits."""
    if not goal.strip():
        raise ValueError("goal must not be empty")
    if mode not in {"auto", "reflex", "deliberative", "deep"}:
        raise ValueError("mode must be auto, reflex, deliberative, or deep")
    if not 1 <= max_allowed_steps <= 20:
        raise ValueError("max_allowed_steps must be in [1, 20]")
    if not 32 <= max_allowed_tokens <= 4096:
        raise ValueError("max_allowed_tokens must be in [32, 4096]")

    normalized = goal.casefold()
    rationale: list[str] = []
    if mode == "auto":
        has_complex_cue = any(cue in normalized for cue in _DEEP_CUES)
        has_file_cue = has_workspace_tools and any(cue in normalized for cue in _FILE_CUES)
        has_short_cue = any(cue in normalized for cue in _SHORT_ANSWER_CUES)
        if has_complex_cue or (has_image and has_audio):
            selected_mode = "deep"
            rationale.append("multi-step/analysis cue or combined modality detected")
        elif has_short_cue and not (has_image or has_audio or has_file_cue) and len(goal) <= 180:
            selected_mode = "reflex"
            rationale.append("short-answer cue without visual/audio or workspace-operation cues")
        elif has_image or has_audio or has_file_cue or len(goal) > 180:
            selected_mode = "deliberative"
            rationale.append("modality, workspace operation, or long request needs verification")
        else:
            selected_mode = "deliberative"
            rationale.append("conservative default when task type is ambiguous")
    else:
        selected_mode = mode
        rationale.append("mode explicitly selected by caller")

    budgets = {
        "reflex": (1, 160),
        "deliberative": (6, 512),
        "deep": (12, 768),
    }
    steps, tokens = budgets[selected_mode]
    steps = min(steps, max_allowed_steps)
    tokens = min(tokens, max_allowed_tokens)
    if has_image:
        rationale.append("image input is enabled")
    if has_audio:
        rationale.append("audio will be transcribed locally before planning")
    rationale.append("verification and tool permission policy remain unchanged")
    return TaskBudget(
        mode=selected_mode,
        max_steps=steps,
        max_new_tokens=tokens,
        rationale=tuple(rationale),
    )


def budget_as_dict(budget: TaskBudget) -> dict:
    """Serialize the chosen budget for audit logs and W&B metadata."""
    return asdict(budget)
