import torch

from mylm_omni.training.distillation import TAIDSchedule, taid_loss, token_reverse_kl_loss


def test_taid_loss_has_finite_gradients_and_masks_prompt_tokens():
    torch.manual_seed(3)
    student = torch.randn(2, 4, 19, requires_grad=True)
    teacher = torch.randn(2, 4, 19)
    mask = torch.tensor([[False, True, True, False], [False, True, True, True]])
    loss = taid_loss(student, teacher, mask, t=0.6)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert torch.all(student.grad[~mask] == 0)


def test_taid_schedule_is_monotonic_and_persists_state():
    schedule = TAIDSchedule(t_start=0.4, t_end=0.9, alpha=0.1, beta=0.9)
    values = []
    for step, loss in enumerate([3.0, 2.5, 2.2, 2.3, 1.8, 1.5], start=1):
        schedule.update(loss, global_step=step, total_steps=10)
        values.append(schedule.t)
    assert values == sorted(values)
    assert values[-1] <= 0.9
    restored = TAIDSchedule(t_start=0.4, t_end=0.9, alpha=0.1, beta=0.9)
    restored.load_state_dict(schedule.state_dict())
    assert restored.state_dict() == schedule.state_dict()


def test_reverse_kl_is_non_negative_and_differentiable():
    student = torch.randn(1, 3, 13, requires_grad=True)
    teacher = torch.randn(1, 3, 13)
    mask = torch.tensor([[False, True, True]])
    loss = token_reverse_kl_loss(student, teacher, mask)
    assert torch.isfinite(loss)
    assert loss.item() >= -1e-6
    loss.backward()
    assert student.grad is not None
