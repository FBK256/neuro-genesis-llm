import pytest
import torch

from mylm_omni.modeling import MyLMConfig, MyLMForCausalLM


def make_model(image_feature_size=None):
    config = MyLMConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=32,
        max_recurrent_steps=3,
        dropout=0.0,
        image_feature_size=image_feature_size,
    )
    return MyLMForCausalLM(config)


def test_causal_attention_does_not_leak_future_tokens():
    torch.manual_seed(4)
    model = make_model().eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = first.clone()
    second[0, 4] = 21
    with torch.no_grad():
        logits_a = model(first)["logits"]
        logits_b = model(second)["logits"]
    assert torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-5, rtol=1e-5)


def test_forward_loss_and_backward_are_finite():
    torch.manual_seed(2)
    model = make_model().train()
    tokens = torch.tensor([[1, 2, 3, 4, 5], [3, 8, 9, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    labels = tokens.clone()
    labels[mask == 0] = -100
    output = model(tokens, attention_mask=mask, labels=labels, return_trace=True)
    assert output["logits"].shape == (2, 5, 64)
    assert torch.isfinite(output["loss"])
    assert 1.0 <= output["trace"]["mean_expected_depth"] <= 3.0
    output["loss"].backward()
    assert model.depth_router.weight.grad is not None
    assert torch.isfinite(model.depth_router.weight.grad).all()


def test_image_features_support_padded_images_per_example():
    torch.manual_seed(3)
    model = make_model(image_feature_size=12).train()
    tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 9, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    labels = tokens.clone()
    labels[mask == 0] = -100
    features = torch.randn(2, 3, 12)
    image_mask = torch.tensor([[1, 1, 1], [0, 0, 0]])
    output = model(
        tokens, attention_mask=mask, labels=labels,
        image_features=features, image_attention_mask=image_mask, return_trace=True,
    )
    assert output["logits"].shape == (2, 7, 64)
    assert torch.isfinite(output["loss"])
    assert output["trace"]["image_prefix_tokens"] == 3


def test_image_features_fail_loudly_without_projector():
    model = make_model().eval()
    with torch.no_grad():
        try:
            model(torch.tensor([[1, 2]]), image_features=torch.randn(1, 2, 12))
        except ValueError as error:
            assert "image_feature_size" in str(error)
        else:
            raise AssertionError("expected a configuration error for image features")



def test_logits_to_keep_returns_only_requested_sequence_positions():
    model = make_model().eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        full = model(tokens)["logits"]
        last = model(tokens, logits_to_keep=1)["logits"]
    assert full.shape == (1, 5, 64)
    assert last.shape == (1, 1, 64)
    # The two equivalent paths differ slightly in floating-point accumulation order.
    assert torch.allclose(full[:, -1:], last, atol=1e-5, rtol=1e-5)


def test_greedy_generation_supports_zero_temperature():
    torch.manual_seed(11)
    model = make_model().eval()
    tokens = torch.tensor([[1, 2, 3]])
    output = model.generate(tokens, max_new_tokens=3, temperature=0.0, top_p=1.0)
    assert output.shape == (1, 6)
    assert output[0, :3].tolist() == tokens[0].tolist()



def test_all_masked_sequences_are_rejected_instead_of_entering_sparse_depth_routing():
    model = make_model().eval()
    tokens = torch.tensor([[1, 2, 3]])
    mask = torch.zeros_like(tokens)
    with torch.no_grad():
        try:
            model(tokens, attention_mask=mask, adaptive_inference=True)
        except ValueError as error:
            assert "at least one unmasked token" in str(error)
        else:
            raise AssertionError("expected an empty attention row to be rejected")



def test_generate_with_trace_exports_routing_and_per_layer_diagnostics():
    torch.manual_seed(12)
    model = make_model().eval()
    prompt = torch.tensor([[1, 2, 3, 4]])
    generated, trace = model.generate_with_trace(
        prompt,
        max_new_tokens=2,
        temperature=0.0,
        top_p=1.0,
        trace_token_limit=3,
    )
    assert generated.shape == (1, 6)
    assert trace["format"] == "mylm-omni-inference-trace-v1"
    assert trace["parameter_count"] == model.num_parameters()
    assert trace["generated_tokens"] == 2
    assert len(trace["trace_steps"]) == 2
    first_step = trace["trace_steps"][0]
    assert len(first_step["unique_layer_activation_stats"]) == model.config.num_layers
    assert first_step["recurrent_activation_stats"]
    probabilities = first_step["router_probabilities_for_next_token"][0]
    assert len(probabilities) == model.config.max_recurrent_steps
    assert abs(sum(probabilities) - 1.0) < 1e-5
    assert len(first_step["selected_depth_for_context_tokens"][0]) <= 3
    assert first_step["forward_elapsed_ms"] >= 0
    assert trace["total_elapsed_ms"] >= 0



def test_max_position_embeddings_is_canonical_and_legacy_alias_still_works():
    config = MyLMConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_position_embeddings=8,
        max_recurrent_steps=2,
    )
    assert config.max_position_embeddings == 8
    assert config.max_seq_len == 8
    assert config.to_dict()["max_position_embeddings"] == 8
    assert "max_seq_len" not in config.to_dict()

    legacy = MyLMConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_seq_len=6,
        max_recurrent_steps=2,
    )
    assert legacy.max_position_embeddings == 6
    assert legacy.max_seq_len == 6


def test_forward_enforces_max_position_embeddings():
    model = MyLMForCausalLM(MyLMConfig(
        vocab_size=64,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=64,
        max_position_embeddings=4,
        max_recurrent_steps=2,
    )).eval()
    with pytest.raises(ValueError, match="max_position_embeddings=4"):
        model(torch.tensor([[1, 2, 3, 4, 5]]))
