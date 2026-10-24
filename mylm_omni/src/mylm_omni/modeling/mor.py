"""Small causal decoder with MoR-inspired shared recurrence.

This is a research implementation, not a reproduction of the published MoR system.
Adaptive inference skips inactive tokens' Query/attention-output/FFN work, but still rebuilds
K/V for all sequence tokens at every recurrence step. There is no fused sparse CUDA kernel or
KV-cache sharing yet, so wall-clock gains must be measured rather than inferred from FLOPs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MyLMConfig:
    """Shape and runtime configuration for the scratch research model."""

    vocab_size: int = 262_144
    embedding_dim: int = 384
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 2
    intermediate_size: int = 1_376
    max_position_embeddings: int = 262_144
    # Backward-compatibility alias. New configs should use max_position_embeddings.
    max_seq_len: int | None = None
    max_recurrent_steps: int = 2
    gradient_checkpointing: bool = True
    dropout: float = 0.0
    image_feature_size: int | None = None
    depth_cost_weight: float = 0.002
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    vocab_loss_chunk_size: int = 32

    def __post_init__(self) -> None:
        # Old configs and callers may still provide max_seq_len.
        if self.max_seq_len is not None:
            if (
                self.max_position_embeddings != 262_144
                and self.max_position_embeddings != self.max_seq_len
            ):
                raise ValueError(
                    "max_position_embeddings and legacy max_seq_len must match"
                )
            self.max_position_embeddings = self.max_seq_len
        # Keep the legacy attribute synchronized for existing code.
        self.max_seq_len = self.max_position_embeddings
        if self.vocab_size < 8:
            raise ValueError("vocab_size must be at least 8")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if self.hidden_size <= 0 or self.num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.vocab_loss_chunk_size < 1:
            raise ValueError("vocab_loss_chunk_size must be positive")
        if self.num_heads < 1 or self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_kv_heads < 1 or self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("head dimension must be even for RoPE")
        if self.max_recurrent_steps < 1:
            raise ValueError("max_recurrent_steps must be >= 1")
        if self.max_position_embeddings < 1:
            raise ValueError("max_position_embeddings must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        """Return canonical JSON settings without the deprecated max_seq_len alias."""
        config = asdict(self)
        config.pop("max_seq_len", None)
        return config


class RMSNorm(nn.Module):
    """RMS normalization with FP32 variance math for numerical stability."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.float()
        scale = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_fp32 * scale).to(input_dtype) * self.weight.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two channel halves used by the split-half RoPE convention."""
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_positions: torch.Tensor | None = None,
    k_positions: torch.Tensor | None = None,
    theta: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply split-half RoPE for full sequences or query subsets at original token positions."""
    head_dim = q.shape[-1]
    device = q.device
    calc_dtype = torch.float32
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=calc_dtype) / head_dim)
    )

    def position_angles(
        positions: torch.Tensor | None,
        length: int,
        *,
        batch_size: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions is None:
            positions = torch.arange(length, device=device, dtype=calc_dtype)
        else:
            positions = positions.to(device=device, dtype=calc_dtype)
        if positions.ndim == 1:
            angles = torch.outer(positions, inv_freq)
            angles = torch.cat((angles, angles), dim=-1)
            cos = angles.cos().to(dtype=dtype)[None, None, :, :]
            sin = angles.sin().to(dtype=dtype)[None, None, :, :]
        elif positions.ndim == 2 and positions.shape[0] == batch_size:
            angles = positions.unsqueeze(-1) * inv_freq
            angles = torch.cat((angles, angles), dim=-1)
            cos = angles.cos().to(dtype=dtype).unsqueeze(1)
            sin = angles.sin().to(dtype=dtype).unsqueeze(1)
        else:
            raise ValueError("RoPE positions must have shape [sequence] or [batch, sequence]")
        return cos, sin

    q_cos, q_sin = position_angles(
        q_positions, q.shape[-2], batch_size=q.shape[0], dtype=q.dtype
    )
    k_cos, k_sin = position_angles(
        k_positions, k.shape[-2], batch_size=k.shape[0], dtype=k.dtype
    )
    q = (q * q_cos) + (_rotate_half(q) * q_sin)
    k = (k * k_cos) + (_rotate_half(k) * k_sin)
    return q, k


class CausalSelfAttention(nn.Module):
    """Grouped-query attention using PyTorch scaled-dot-product attention."""

    def __init__(self, config: MyLMConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.dropout
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, theta=self.rope_theta)

        # Expand only the view; grouped KV heads reduce learned parameters and cache footprint.
        repeat_factor = self.num_heads // self.num_kv_heads
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        if attention_mask is None or not bool(
            (
                attention_mask[:, 1:].to(dtype=torch.bool)
                & ~attention_mask[:, :-1].to(dtype=torch.bool)
            ).any()
        ):
            # Standard right padding is already hidden from valid queries by causal attention.
            # Avoid materializing a [sequence, sequence] causal-plus-padding mask.
            attn_mask = None
            use_causal_flag = True
        else:
            # Non-monotone masks (e.g. padded image prefixes) need an explicit mask.
            causal = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
            padding = attention_mask.to(device=x.device, dtype=torch.bool)[:, None, None, :]
            attn_mask = causal[None, None, :, :] & padding
            use_causal_flag = False

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal_flag,
        )
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.o_proj(output)

    def forward_active_queries(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute attention only for active queries while retaining every token as K/V context.

        Returns active-query outputs, original sequence indices, and a padded-query validity mask.
        The attention score tensor is [batch, heads, active_queries, sequence], not sequence^2.
        """
        batch, seq_len, hidden = x.shape
        if active_mask.shape != (batch, seq_len):
            raise ValueError("active_mask must match [batch, sequence]")
        active_mask = active_mask.to(device=x.device, dtype=torch.bool)
        counts = active_mask.sum(dim=1)
        max_active = int(counts.max().item()) if counts.numel() else 0
        if max_active == 0:
            return (
                x.new_zeros((batch, 0, hidden)),
                torch.zeros((batch, 0), dtype=torch.long, device=x.device),
                torch.zeros((batch, 0), dtype=torch.bool, device=x.device),
            )

        indices = torch.zeros((batch, max_active), dtype=torch.long, device=x.device)
        query_valid = torch.arange(max_active, device=x.device)[None, :] < counts[:, None]
        for batch_index in range(batch):
            active_indices = torch.where(active_mask[batch_index])[0]
            if active_indices.numel():
                indices[batch_index, :active_indices.numel()] = active_indices

        active_x = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, hidden))
        q = self.q_proj(active_x).view(
            batch, max_active, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        key_positions = torch.arange(seq_len, device=x.device).expand(batch, -1)
        q, k = _apply_rope(
            q, k, q_positions=indices, k_positions=key_positions, theta=self.rope_theta
        )

        repeat_factor = self.num_heads // self.num_kv_heads
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        # An active query can attend to all non-padding keys at or before its original position.
        causal = key_positions[:, None, :] <= indices[:, :, None]
        if attention_mask is None:
            key_valid = torch.ones(
                batch, seq_len, dtype=torch.bool, device=x.device
            )
        else:
            if attention_mask.shape != (batch, seq_len):
                raise ValueError("attention_mask must match [batch, sequence]")
            key_valid = attention_mask.to(device=x.device, dtype=torch.bool)
        allowed = causal & key_valid[:, None, :] & query_valid[:, :, None]
        # Padded query rows need one finite softmax position; their outputs are zeroed immediately.
        if seq_len:
            allowed[:, :, 0] |= ~query_valid
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=allowed[:, None, :, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(batch, max_active, hidden)
        output = self.o_proj(output)
        output = output * query_valid.unsqueeze(-1).to(output.dtype)
        return output, indices, query_valid


class SwiGLU(nn.Module):
    """Gated feed-forward network."""

    def __init__(self, config: MyLMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with stable residual updates."""

    def __init__(self, config: MyLMConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def forward_active(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update only active tokens; inactive states remain frozen but still provide K/V context."""
        batch, seq_len, hidden = x.shape
        active_mask = active_mask.to(device=x.device, dtype=torch.bool)
        if attention_mask is not None:
            active_mask = active_mask & attention_mask.to(device=x.device, dtype=torch.bool)
        if not bool(active_mask.any()):
            return x
        normalized = self.attn_norm(x)
        attention_output, indices, query_valid = self.attn.forward_active_queries(
            normalized, active_mask, attention_mask=attention_mask
        )
        active_x = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, hidden))
        active_x = active_x + attention_output
        active_x = active_x + self.ffn(self.ffn_norm(active_x))

        # Scatter only genuine active tokens. The Python batch loop avoids padded duplicate
        # indices overwriting a valid token in a row with fewer active queries than the batch max.
        result = x.clone()
        for batch_index in range(batch):
            valid_indices = indices[batch_index, query_valid[batch_index]]
            if valid_indices.numel():
                result[batch_index, valid_indices] = active_x[batch_index, query_valid[batch_index]]
        return result


class MyLMForCausalLM(nn.Module):
    """Causal LM with unique layers followed by a shared recurrent reasoning block.

    In training mode the hidden states from each recurrence depth are softly mixed,
    allowing gradients to train the depth router. In adaptive inference mode inactive tokens
    remain frozen and only active tokens form attention queries / receive FFN updates, while all
    visible token states remain K/V context. K/V projections are still dense and recomputed at
    each step; this is not a fused sparse kernel or an optimized KV-cache serving engine.
    """

    def __init__(self, config: MyLMConfig) -> None:
        super().__init__()
        self.config = config
        # Factorized embeddings preserve a large vocabulary without a [vocab, hidden] table.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.embed_projection = (
            nn.Linear(config.embedding_dim, config.hidden_size, bias=False)
            if config.embedding_dim != config.hidden_size else nn.Identity()
        )
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.recurrent_block = TransformerBlock(config)
        self.depth_router = nn.Linear(config.hidden_size, config.max_recurrent_steps)
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # This projects to the token-table dimension; vocabulary weights stay tied.
        self.lm_head = nn.Linear(config.hidden_size, config.embedding_dim, bias=False)
        self.vision_projector = (
            nn.Sequential(
                nn.Linear(config.image_feature_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            if config.image_feature_size is not None
            else None
        )
        self._initialize_transformer_weights()

    def _initialize_transformer_weights(self) -> None:
        """Use Transformer-style normal init and scale residual projections for stability."""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

        # The shared recurrent block is reused multiple times. Scale every residual-output
        # projection by the effective number of block applications to control residual growth.
        effective_depth = self.config.num_layers + self.config.max_recurrent_steps
        residual_std = 0.02 / (2.0 * effective_depth) ** 0.5
        for block in [*self.layers, self.recurrent_block]:
            nn.init.normal_(block.attn.o_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=residual_std)

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Count parameters; set trainable_only=True to count only parameters with gradients."""
        return sum(
            p.numel() for p in self.parameters() if not trainable_only or p.requires_grad
        )

    def _vocabulary_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states to embedding space, then apply the tied token matrix."""
        return F.linear(self.lm_head(hidden), self.embed_tokens.weight)

    @torch.no_grad()
    def _chunked_lm_accuracy(
        self, hidden: torch.Tensor, shifted_labels: torch.Tensor
    ) -> tuple[int, int]:
        """Exact top-1 accuracy in chunks without retaining full-sequence logits."""
        valid = shifted_labels.ne(-100)
        correct = torch.zeros((), dtype=torch.long, device=hidden.device)
        chunk_size = self.config.vocab_loss_chunk_size
        for start in range(0, hidden.shape[1], chunk_size):
            end = min(start + chunk_size, hidden.shape[1])
            logits = self._vocabulary_logits(hidden[:, start:end, :])
            targets = shifted_labels[:, start:end]
            correct += ((logits.argmax(dim=-1) == targets) & targets.ne(-100)).sum()
        return int(correct.item()), int(valid.sum().item())

    def _chunked_lm_loss(
        self, hidden: torch.Tensor, shifted_labels: torch.Tensor
    ) -> torch.Tensor:
        """Exact cross entropy with bounded logit memory; recompute vocabulary projection in BP."""
        from torch.utils.checkpoint import checkpoint

        valid_count = shifted_labels.ne(-100).sum()
        if int(valid_count.item()) == 0:
            raise ValueError("labels contain no supervised next-token targets")
        total_loss = hidden.new_zeros((), dtype=torch.float32)
        chunk_size = self.config.vocab_loss_chunk_size

        def chunk_loss(hidden_chunk: torch.Tensor, labels_chunk: torch.Tensor) -> torch.Tensor:
            chunk_logits = self._vocabulary_logits(hidden_chunk)
            return F.cross_entropy(
                chunk_logits.float().reshape(-1, self.config.vocab_size),
                labels_chunk.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )

        for start in range(0, hidden.shape[1], chunk_size):
            end = min(start + chunk_size, hidden.shape[1])
            hidden_chunk = hidden[:, start:end, :]
            labels_chunk = shifted_labels[:, start:end]
            if self.training and torch.is_grad_enabled():
                chunk_sum = checkpoint(
                    chunk_loss, hidden_chunk, labels_chunk, use_reentrant=False
                )
            else:
                chunk_sum = chunk_loss(hidden_chunk, labels_chunk)
            total_loss = total_loss + chunk_sum
        return total_loss / valid_count

    @staticmethod
    def _activation_summary(
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> dict[str, float]:
        """Return scalar diagnostics for active token states without retaining activations."""
        values = hidden.detach().float()
        if attention_mask is not None and attention_mask.shape == values.shape[:2]:
            active = attention_mask.to(device=values.device, dtype=torch.bool)
            values = values[active]
        else:
            values = values.reshape(-1, values.shape[-1])
        if values.numel() == 0:
            return {"mean": 0.0, "std": 0.0, "near_zero_fraction": 0.0,
                    "max_abs": 0.0, "finite_fraction": 1.0}
        finite = torch.isfinite(values)
        safe_values = torch.where(finite, values, torch.zeros_like(values))
        return {
            "mean": float(safe_values.mean().cpu()),
            "std": float(safe_values.std(unbiased=False).cpu()),
            "near_zero_fraction": float((safe_values.abs() < 1e-3).float().mean().cpu()),
            "max_abs": float(safe_values.abs().max().cpu()),
            "finite_fraction": float(finite.float().mean().cpu()),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
        image_attention_mask: torch.Tensor | None = None,
        adaptive_inference: bool = False,
        return_trace: bool = False,
        logits_to_keep: int = 0,
        return_logits: bool = True,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids batch and sequence dimensions must be non-empty")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids before image prefix insertion")
        if attention_mask is not None and not bool(attention_mask.to(dtype=torch.bool).any(dim=1).all()):
            raise ValueError("every sequence must contain at least one unmasked token")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids before image prefix insertion")
        if logits_to_keep < 0:
            raise ValueError("logits_to_keep must be non-negative")
        if logits_to_keep and labels is not None:
            raise ValueError("logits_to_keep cannot be used when labels are provided")
        x = self.embed_projection(self.embed_tokens(input_ids))
        prefix_len = 0

        if image_features is not None:
            if self.vision_projector is None:
                raise ValueError(
                    "image_features supplied but image_feature_size is not configured"
                )
            if image_features.ndim != 3:
                raise ValueError("image_features must have shape [batch, patches, feature_size]")
            if image_features.shape[0] != x.shape[0]:
                raise ValueError("image and text batch sizes differ")
            image_features = image_features.to(device=x.device, dtype=x.dtype)
            image_tokens = self.vision_projector(image_features)
            prefix_len = image_tokens.shape[1]
            x = torch.cat((image_tokens, x), dim=1)
            if image_attention_mask is None:
                prefix_mask = torch.ones(
                    x.shape[0], prefix_len, device=x.device, dtype=torch.long
                )
            else:
                if image_attention_mask.shape != (x.shape[0], prefix_len):
                    raise ValueError("image_attention_mask must match [batch, image_patches]")
                prefix_mask = image_attention_mask.to(device=x.device, dtype=torch.long)
            if attention_mask is None:
                attention_mask = torch.ones(
                    input_ids.shape[0], input_ids.shape[1], device=x.device, dtype=torch.long
                )
            attention_mask = torch.cat(
                (prefix_mask.to(dtype=attention_mask.dtype), attention_mask), dim=1
            )
            if labels is not None:
                prefix_labels = torch.full(
                    (labels.shape[0], prefix_len), -100,
                    device=labels.device, dtype=labels.dtype
                )
                labels = torch.cat((prefix_labels, labels), dim=1)

        if x.shape[1] > self.config.max_position_embeddings:
            raise ValueError(
                "input length "
                f"{x.shape[1]} exceeds max_position_embeddings="
                f"{self.config.max_position_embeddings}"
            )

        layer_activation_stats = []
        from torch.utils.checkpoint import checkpoint

        for layer_index, layer in enumerate(self.layers):
            if self.training and self.config.gradient_checkpointing:
                x = checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask=attention_mask)
            if return_trace:
                layer_activation_stats.append({
                    "layer": f"unique-{layer_index + 1}",
                    **self._activation_summary(x, attention_mask),
                })

        depth_probs = F.softmax(self.depth_router(x).float(), dim=-1).to(dtype=x.dtype)
        recurrent_activation_stats = []
        expected_depth = (
            depth_probs
            * torch.arange(
                1, self.config.max_recurrent_steps + 1, device=x.device, dtype=depth_probs.dtype
            )
        ).sum(dim=-1)

        if adaptive_inference and not self.training:
            # Tokens can be assigned different depths. Frozen token states remain available
            # as context for later tokens; a future optimized kernel can skip inactive queries.
            selected_depth = depth_probs.argmax(dim=-1) + 1
            if attention_mask is not None:
                valid_tokens = attention_mask.to(device=x.device, dtype=torch.bool)
                # Padding is not a reasoning token and must not increase the loop count.
                selected_depth = torch.where(
                    valid_tokens, selected_depth, torch.ones_like(selected_depth)
                )
                max_depth = int(selected_depth.masked_select(valid_tokens).max().item())
            else:
                max_depth = int(selected_depth.max().item())
            for depth in range(1, max_depth + 1):
                active = selected_depth >= depth
                x = self.recurrent_block.forward_active(
                    x, active, attention_mask=attention_mask
                )
                if return_trace:
                    active_valid = (
                        active
                        if attention_mask is None
                        else active & attention_mask.to(device=x.device, dtype=torch.bool)
                    )
                    recurrent_activation_stats.append({
                        "block": f"recurrent-{depth}",
                        "active_tokens": int(active_valid.sum().item()),
                        "active_fraction": float(
                            (
                                active_valid.sum().float()
                                / (
                                    attention_mask.to(device=x.device, dtype=torch.bool).sum().clamp_min(1)
                                    if attention_mask is not None
                                    else active_valid.numel()
                                )
                            ).cpu()
                        ),
                        **self._activation_summary(x, attention_mask),
                    })
        else:
            states = []
            current = x
            for _ in range(self.config.max_recurrent_steps):
                if self.training and self.config.gradient_checkpointing:
                    current = checkpoint(
                        self.recurrent_block, current, attention_mask, use_reentrant=False
                    )
                else:
                    current = self.recurrent_block(current, attention_mask=attention_mask)
                states.append(current)
                if return_trace:
                    recurrent_activation_stats.append({
                        "block": f"recurrent-{len(states)}",
                        **self._activation_summary(current, attention_mask),
                    })
            stacked = torch.stack(states, dim=-2)
            x = (stacked * depth_probs.unsqueeze(-1)).sum(dim=-2)
            selected_depth = depth_probs.argmax(dim=-1) + 1

        x = self.final_norm(x)
        logits = None
        if return_logits:
            logits_hidden = x[:, -logits_to_keep:, :] if logits_to_keep else x
            logits = self._vocabulary_logits(logits_hidden)
        loss = None
        lm_loss = None
        depth_cost = x.new_zeros((), dtype=torch.float32)
        correct_tokens = 0
        supervised_tokens = 0
        if labels is not None:
            # Next-token prediction. Ignore image prefix and any masked label positions.
            shift_labels = labels[:, 1:].contiguous()
            if not bool(shift_labels.ne(-100).any()):
                raise ValueError("labels contain no supervised next-token targets")
            if return_logits:
                shift_logits = logits[:, :-1, :].contiguous().float()
                lm_loss = F.cross_entropy(
                    shift_logits.reshape(-1, self.config.vocab_size),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
            else:
                lm_loss = self._chunked_lm_loss(x[:, :-1, :], shift_labels)
                if not self.training:
                    correct_tokens, supervised_tokens = self._chunked_lm_accuracy(
                        x[:, :-1, :], shift_labels
                    )
            if return_logits:
                supervised_tokens = int(shift_labels.ne(-100).sum().item())
            if self.config.depth_cost_weight:
                if attention_mask is not None:
                    active = attention_mask.to(device=expected_depth.device, dtype=torch.bool)
                    valid_depth = expected_depth.masked_select(active)
                    depth_cost = (
                        valid_depth.mean() if valid_depth.numel()
                        else expected_depth.new_zeros((), dtype=torch.float32)
                    )
                else:
                    depth_cost = expected_depth.mean()
                loss = lm_loss + self.config.depth_cost_weight * depth_cost
            else:
                loss = lm_loss

        result: dict[str, Any] = {
            "logits": logits,
            "loss": loss,
            "lm_loss": lm_loss,
            "depth_cost": depth_cost,
            "correct_tokens": correct_tokens,
            "supervised_tokens": supervised_tokens,
        }
        if return_trace:
            valid_tokens = (
                attention_mask.to(device=x.device, dtype=torch.bool)
                if attention_mask is not None
                else torch.ones(selected_depth.shape, device=x.device, dtype=torch.bool)
            )
            valid_depths = selected_depth[valid_tokens]
            depth_histogram = {
                str(depth): (
                    float((valid_depths == depth).float().mean().cpu())
                    if valid_depths.numel() else 0.0
                )
                for depth in range(1, self.config.max_recurrent_steps + 1)
            }
            result["trace"] = {
                "depth_probabilities": depth_probs.detach(),
                "selected_depth": selected_depth.detach(),
                "mean_expected_depth": float(
                    (
                        expected_depth.detach().float().masked_select(
                            attention_mask.to(device=x.device, dtype=torch.bool)
                        ).mean()
                        if attention_mask is not None and bool(
                            attention_mask.to(device=x.device, dtype=torch.bool).any()
                        )
                        else expected_depth.detach().float().mean()
                    ).cpu()
                ),
                "depth_histogram": depth_histogram,
                "unique_layer_activation_stats": layer_activation_stats,
                "recurrent_activation_stats": recurrent_activation_stats,
                "layers": len(self.layers),
                "recurrent_block_calls_in_dense_training": (
                    self.config.max_recurrent_steps if not adaptive_inference else None
                ),
                "image_prefix_tokens": prefix_len,
            }
        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        attention_mask: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
        image_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample output with sparse recurrent depth; KV-cache optimization is not implemented."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if temperature < 0:
            raise ValueError("temperature must be >= 0; use 0 for greedy decoding")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        self.eval()
        image_prefix_len = int(image_features.shape[1]) if image_features is not None else 0
        text_context_budget = self.config.max_position_embeddings - image_prefix_len
        if text_context_budget < 1:
            raise ValueError("image features leave no room for text tokens in max_seq_len")
        for _ in range(max_new_tokens):
            cropped = input_ids[:, -text_context_budget:]
            cropped_mask = (
                attention_mask[:, -text_context_budget:] if attention_mask is not None else None
            )
            logits = self(
                cropped,
                attention_mask=cropped_mask,
                image_features=image_features,
                image_attention_mask=image_attention_mask,
                adaptive_inference=True,
                logits_to_keep=1,
            )["logits"][:, -1, :].float()
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                if top_p < 1:
                    sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
                    cumulative = sorted_probs.cumsum(-1)
                    remove = cumulative - sorted_probs >= top_p
                    sorted_probs = sorted_probs.masked_fill(remove, 0)
                    sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True).clamp_min(1e-12)
                    probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
                next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)), dim=-1
                )
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return input_ids


    @torch.no_grad()
    def generate_with_trace(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 32,
        eos_token_id: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        attention_mask: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
        image_attention_mask: torch.Tensor | None = None,
        trace_token_limit: int = 128,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Generate output and record the model's actual compute path at each decoding step.

        The JSON-serializable trace contains selected recurrence depths, router probabilities
        for the token being generated, per-layer activation summaries and timings. It does not
        store natural-language chain-of-thought or full hidden-state tensors.
        """
        if not 1 <= max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be in [1, 1024] for traced generation")
        if temperature < 0:
            raise ValueError("temperature must be >= 0; use 0 for greedy decoding")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if trace_token_limit < 1:
            raise ValueError("trace_token_limit must be positive")
        if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must be a non-empty [batch, sequence] tensor")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")

        self.eval()
        input_device = input_ids.device
        original_prompt_tokens = int(input_ids.shape[1])
        image_prefix_len = int(image_features.shape[1]) if image_features is not None else 0
        text_context_budget = self.config.max_position_embeddings - image_prefix_len
        if text_context_budget < 1:
            raise ValueError("image features leave no room for text tokens in max_seq_len")
        trace_steps: list[dict[str, Any]] = []
        overall_started = time.perf_counter()

        for generation_index in range(max_new_tokens):
            step_started = time.perf_counter()
            cropped = input_ids[:, -text_context_budget:]
            cropped_mask = (
                attention_mask[:, -text_context_budget:] if attention_mask is not None else None
            )
            output = self(
                cropped,
                attention_mask=cropped_mask,
                image_features=image_features,
                image_attention_mask=image_attention_mask,
                adaptive_inference=True,
                return_trace=True,
                logits_to_keep=1,
            )
            logits = output["logits"][:, -1, :].float()
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                if top_p < 1:
                    sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
                    cumulative = sorted_probs.cumsum(-1)
                    remove = cumulative - sorted_probs >= top_p
                    sorted_probs = sorted_probs.masked_fill(remove, 0)
                    sorted_probs = sorted_probs / sorted_probs.sum(-1, keepdim=True).clamp_min(1e-12)
                    probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
                next_token = torch.multinomial(probs, num_samples=1)

            model_trace = output["trace"]
            selected_depth = model_trace["selected_depth"]
            depth_probs = model_trace["depth_probabilities"]
            if input_device.type == "cuda":
                torch.cuda.synchronize(input_device)
            step_elapsed_ms = (time.perf_counter() - step_started) * 1000.0
            trace_steps.append({
                "generation_index": generation_index,
                "input_text_tokens": int(cropped.shape[1]),
                "image_prefix_tokens": int(model_trace["image_prefix_tokens"]),
                "next_token_ids": next_token.detach().cpu().tolist(),
                "router_probabilities_for_next_token": (
                    depth_probs[:, -1, :].detach().float().cpu().tolist()
                ),
                "selected_depth_for_context_tokens": (
                    selected_depth[:, -trace_token_limit:].detach().cpu().tolist()
                ),
                "mean_expected_depth": model_trace["mean_expected_depth"],
                "depth_histogram": model_trace["depth_histogram"],
                "unique_layer_activation_stats": model_trace["unique_layer_activation_stats"],
                "recurrent_activation_stats": model_trace["recurrent_activation_stats"],
                "forward_elapsed_ms": step_elapsed_ms,
            })

            input_ids = torch.cat((input_ids, next_token), dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)), dim=-1
                )
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break

        if input_device.type == "cuda":
            torch.cuda.synchronize(input_device)
        trace = {
            "format": "mylm-omni-inference-trace-v1",
            "model_config": self.config.to_dict(),
            "parameter_count": self.num_parameters(),
            "prompt_tokens": original_prompt_tokens,
            "generated_tokens": int(input_ids.shape[1] - original_prompt_tokens),
            "total_elapsed_ms": (time.perf_counter() - overall_started) * 1000.0,
            "trace_steps": trace_steps,
            "interpretation": {
                "router_probabilities_for_next_token": (
                    "Softmax probabilities for recurrence depths 1..max_recurrent_steps at the "
                    "last visible context token. The selected depth is argmax(probabilities)+1."
                ),
                "activation_statistics": (
                    "Scalar summaries per layer/recurrence step; these are not explanations of "
                    "semantic reasoning and must not be interpreted as causal evidence by themselves."
                ),
                "timings": (
                    "Per-step wall time includes trace collection and device synchronization; "
                    "use untraced generate() for serving latency benchmarks."
                ),
            },
        }
        return input_ids, trace
