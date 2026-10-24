"""Distillation losses for sparse teacher logit traces."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_topk_tail_kl(
    student_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    teacher_tail_logsumexp: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(teacher || student) over top-k symbols plus one aggregated tail bucket.

    Shapes:
      student_logits: [..., vocabulary]
      topk_ids, teacher_topk_logits: [..., k]
      teacher_tail_logsumexp: [...]

    Teacher values are assumed to have already been divided by the teacher temperature.
    This is necessary because a tail log-sum-exp cannot be correctly temperature-scaled
    after compression to one scalar. Student logits are divided by temperature here.

    The tail bucket preserves the teacher's total probability outside top-k but discards
    which tail token received that mass. Compare against full-logit KD when possible.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if student_logits.ndim < 2:
        raise ValueError("student_logits must include sequence and vocabulary dimensions")
    if topk_ids.shape != teacher_topk_logits.shape:
        raise ValueError("topk_ids and teacher_topk_logits must have identical shapes")
    if topk_ids.shape[:-1] != student_logits.shape[:-1]:
        raise ValueError("top-k leading dimensions must match student logits")
    if teacher_tail_logsumexp.shape != student_logits.shape[:-1]:
        raise ValueError("tail logsumexp must match the leading student-logit dimensions")
    if topk_ids.shape[-1] >= student_logits.shape[-1]:
        raise ValueError("top-k must leave at least one class in the tail bucket")
    if bool((topk_ids < 0).any()) or bool((topk_ids >= student_logits.shape[-1]).any()):
        raise ValueError("top-k token id outside the student vocabulary")

    scaled_student = student_logits.float() / temperature
    student_log_z = torch.logsumexp(scaled_student, dim=-1, keepdim=True)
    student_top_logp = (
        scaled_student.gather(-1, topk_ids.long()) - student_log_z
    )
    is_top = torch.zeros_like(scaled_student, dtype=torch.bool)
    is_top.scatter_(-1, topk_ids.long(), True)
    student_tail_logits = scaled_student.masked_fill(is_top, float("-inf"))
    student_tail_logp = (
        torch.logsumexp(student_tail_logits, dim=-1) - student_log_z.squeeze(-1)
    )

    teacher_bucket_logits = torch.cat(
        (teacher_topk_logits.float(), teacher_tail_logsumexp.float().unsqueeze(-1)), dim=-1
    )
    teacher_logp = F.log_softmax(teacher_bucket_logits, dim=-1)
    teacher_p = teacher_logp.exp()
    student_bucket_logp = torch.cat(
        (student_top_logp, student_tail_logp.unsqueeze(-1)), dim=-1
    )
    per_token_kl = (teacher_p * (teacher_logp - student_bucket_logp)).sum(dim=-1)
    return per_token_kl.mean() * (temperature**2)


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy with standard -100 ignore-index support."""
    if logits.shape[:2] != labels.shape:
        raise ValueError("labels must match logits batch and sequence dimensions")
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().float().view(-1, logits.shape[-1]),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )


class TAIDSchedule:
    """Stateful temporal interpolation schedule based on SakanaAI's published recipe.

    t moves from the student distribution toward the teacher distribution over time.
    The schedule uses relative distillation-loss improvement and an EMA momentum term.
    Store state_dict() in checkpoints so resume does not reset the curriculum.
    """

    def __init__(
        self,
        *,
        t_start: float = 0.4,
        t_end: float = 1.0,
        alpha: float = 5e-4,
        beta: float = 0.99,
        disable_adaptive: bool = False,
    ) -> None:
        if not 0.0 <= t_start < 1.0:
            raise ValueError("t_start must be in [0, 1)")
        if not 0.0 < t_end <= 1.0 or t_end < t_start:
            raise ValueError("t_end must be in (0, 1] and >= t_start")
        if not 0.0 <= alpha <= 1.0 or not 0.0 <= beta < 1.0:
            raise ValueError("alpha must be in [0, 1] and beta must be in [0, 1)")
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.disable_adaptive = bool(disable_adaptive)
        self.t = float(t_start)
        self.prev_loss: float | None = None
        self.momentum = 0.0

    def update(self, loss_value: float, *, global_step: int, total_steps: int) -> float:
        """Update t from observed KD loss and training progress; return adaptive delta."""
        import math

        if not math.isfinite(loss_value):
            raise FloatingPointError("TAID received a non-finite loss")
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        delta_t = 0.0
        if self.prev_loss is None:
            self.prev_loss = loss_value
            return delta_t
        if self.prev_loss is not None:
            relative_change = (self.prev_loss - loss_value) / (abs(self.prev_loss) + 1e-15)
            self.momentum = self.beta * self.momentum + (1.0 - self.beta) * relative_change
            # Numerically stable sigmoid in Python float domain.
            if self.momentum >= 0:
                adaptive_delta = 1.0 / (1.0 + math.exp(-min(self.momentum, 60.0)))
            else:
                exp_value = math.exp(max(self.momentum, -60.0))
                adaptive_delta = exp_value / (1.0 + exp_value)
            delta_t = self.alpha * adaptive_delta * (1.0 - self.t)

        progress = min(1.0, max(0.0, global_step / total_steps))
        target_t = self.t_start + (self.t_end - self.t_start) * progress
        if self.disable_adaptive:
            self.t = target_t
        else:
            self.t = min(self.t_end, max(target_t, self.t + delta_t))
        self.prev_loss = loss_value
        return delta_t

    def state_dict(self) -> dict[str, float | bool | None]:
        return {
            "t_start": self.t_start,
            "t_end": self.t_end,
            "alpha": self.alpha,
            "beta": self.beta,
            "disable_adaptive": self.disable_adaptive,
            "t": self.t,
            "prev_loss": self.prev_loss,
            "momentum": self.momentum,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("t_start", "t_end", "alpha", "beta", "disable_adaptive"):
            if key in state and state[key] != getattr(self, key):
                raise ValueError(f"TAID checkpoint setting mismatch for {key}")
        t = float(state["t"])
        if not self.t_start <= t <= self.t_end:
            raise ValueError("checkpoint TAID interpolation t is outside configured bounds")
        self.t = t
        self.prev_loss = (
            float(state["prev_loss"]) if state.get("prev_loss") is not None else None
        )
        self.momentum = float(state.get("momentum", 0.0))


def taid_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    t: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """TAID's intermediate teacher: interpolate detached student and teacher logits.

    Exact TAID needs dense teacher logits on the same vocabulary/tokenization as the student.
    Use sparse_topk_tail_kl only for the separately labelled sparse-KD approximation.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("TAID requires student/teacher logits with the same shape and vocabulary")
    if student_logits.shape[:-1] != target_mask.shape:
        raise ValueError("target_mask must match logits' batch/sequence dimensions")
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    mask = target_mask.to(device=student_logits.device, dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("target_mask contains no supervised tokens")

    # Teacher distribution p_t = softmax((1-t) * stopgrad(student logits) + t * teacher logits).
    student = student_logits.float() / temperature
    teacher = teacher_logits.detach().float() / temperature
    interpolated_logits = (1.0 - t) * student.detach() + t * teacher
    target_probs = F.softmax(interpolated_logits, dim=-1)
    student_log_probs = F.log_softmax(student, dim=-1)
    # Cross-entropy equals KL up to the target entropy and has the same student gradient.
    token_loss = -(target_probs * student_log_probs).sum(dim=-1)
    loss = token_loss.masked_select(mask).mean()
    return loss * (temperature ** 2)


def token_reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Full-vocabulary per-token reverse KL baseline, not the complete MiniLLM algorithm.

    This provides an ablation for KL(student || teacher) on teacher-forced contexts. It does
    not implement MiniLLM's student-rollout/on-policy sequence-level optimization.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("reverse KL requires matching teacher/student logits")
    if student_logits.shape[:-1] != target_mask.shape:
        raise ValueError("target_mask must match logits' batch/sequence dimensions")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    mask = target_mask.to(device=student_logits.device, dtype=torch.bool)
    if not bool(mask.any()):
        raise ValueError("target_mask contains no supervised tokens")
    student_log_p = F.log_softmax(student_logits.float() / temperature, dim=-1)
    student_p = student_log_p.exp()
    teacher_log_p = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    token_loss = (student_p * (student_log_p - teacher_log_p)).sum(dim=-1)
    return token_loss.masked_select(mask).mean() * (temperature ** 2)
