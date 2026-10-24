import torch

from mylm_omni.modeling import MyLMConfig
from mylm_omni.modeling.mor import TransformerBlock


def make_block():
    config = MyLMConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=16,
        max_recurrent_steps=3,
        dropout=0.0,
        gradient_checkpointing=False,
    )
    return TransformerBlock(config).eval()


def test_active_query_block_matches_full_block_when_all_tokens_are_active():
    torch.manual_seed(8)
    block = make_block()
    x = torch.randn(2, 6, 32)
    mask = torch.ones(2, 6, dtype=torch.long)
    with torch.no_grad():
        full = block(x, attention_mask=mask)
        sparse = block.forward_active(x, mask.bool(), attention_mask=mask)
    assert torch.allclose(full, sparse, atol=1e-5, rtol=1e-5)


def test_inactive_states_are_frozen_and_active_states_update():
    torch.manual_seed(9)
    block = make_block()
    x = torch.randn(1, 6, 32)
    active = torch.tensor([[True, False, True, False, False, True]])
    mask = torch.ones(1, 6, dtype=torch.long)
    with torch.no_grad():
        output = block.forward_active(x, active, attention_mask=mask)
    assert torch.equal(output[:, 1], x[:, 1])
    assert torch.equal(output[:, 3], x[:, 3])
    assert torch.equal(output[:, 4], x[:, 4])
    assert not torch.allclose(output[:, 0], x[:, 0])
    assert not torch.allclose(output[:, 2], x[:, 2])
    assert not torch.allclose(output[:, 5], x[:, 5])


def test_padding_tokens_are_never_updated_even_if_active_flag_is_true():
    torch.manual_seed(10)
    block = make_block()
    x = torch.randn(1, 5, 32)
    active = torch.ones(1, 5, dtype=torch.bool)
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    with torch.no_grad():
        output = block.forward_active(x, active, attention_mask=mask)
    assert torch.equal(output[:, 3:], x[:, 3:])
