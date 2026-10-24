import torch

from mylm_omni.training.distillation import causal_lm_loss, sparse_topk_tail_kl


def test_topk_plus_tail_kl_is_zero_when_teacher_is_same_student():
    torch.manual_seed(19)
    logits = torch.randn(2, 3, 17)
    topk_ids = torch.tensor([[[0, 2, 7, 10]] * 3] * 2)
    topk_values = logits.gather(-1, topk_ids)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, topk_ids, True)
    tail_lse = torch.logsumexp(logits.masked_fill(mask, float("-inf")), dim=-1)
    loss = sparse_topk_tail_kl(logits, topk_ids, topk_values, tail_lse)
    assert torch.isfinite(loss)
    assert loss.item() < 1e-5


def test_sparse_distillation_backpropagates():
    student = torch.randn(1, 2, 12, requires_grad=True)
    top_ids = torch.tensor([[[0, 3, 8], [1, 2, 7]]])
    teacher_logits = torch.randn(1, 2, 3)
    tail = torch.randn(1, 2)
    loss = sparse_topk_tail_kl(student, top_ids, teacher_logits, tail, temperature=2.0)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


def test_causal_lm_loss_ignores_masked_targets():
    logits = torch.randn(1, 4, 20)
    labels = torch.tensor([[1, 2, -100, 4]])
    loss = causal_lm_loss(logits, labels)
    assert torch.isfinite(loss)
