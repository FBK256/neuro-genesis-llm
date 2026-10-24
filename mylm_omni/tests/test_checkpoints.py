import torch

from mylm_omni.modeling import MyLMConfig, MyLMForCausalLM
from mylm_omni.training.checkpoints import CheckpointManager


def make_model():
    return MyLMForCausalLM(MyLMConfig(
        vocab_size=32, hidden_size=16, num_layers=1, num_heads=2, num_kv_heads=1,
        intermediate_size=32, max_seq_len=16, max_recurrent_steps=2,
    ))


def test_latest_checkpoint_restores_model_and_optimizer(tmp_path):
    torch.manual_seed(10)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.tensor([[1, 2, 3, 4]])
    loss = model(ids, labels=ids)["loss"]
    loss.backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    manager = CheckpointManager(tmp_path / "run", keep_steps=2, keep_best=2)
    saved = manager.save(
        step=7, model=model, optimizer=optimizer, training_state={"epoch": 1},
        metrics={"eval_loss": 1.7},
    )
    assert saved.name == "step-000007"
    assert (tmp_path / "run" / "latest" / "model.safetensors").is_file()
    assert (tmp_path / "run" / "best" / "rank-1" / "model.safetensors").is_file()

    restored_model = make_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    manifest = manager.load_latest(model=restored_model, optimizer=restored_optimizer)
    assert manifest["step"] == 7
    assert manifest["training_state"]["epoch"] == 1
    for name, value in restored_model.state_dict().items():
        assert torch.allclose(value, expected[name])
