"""Offline-only mean-pooled embeddings for memory retrieval.

Use a sentence-embedding checkpoint downloaded and reviewed separately. This class always sets
local_files_only=True and never fetches a model from the network. A generic causal/base language
model is not automatically a good sentence embedding model.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch


class LocalTransformerEmbedder:
    """Produce one normalized vector from a local Hugging Face encoder checkpoint."""

    def __init__(
        self,
        local_model_path: str | Path,
        *,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        self.model_path = Path(local_model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"local embedding model not found at {self.model_path}; "
                "download and verify it before enabling offline memory embeddings"
            )
        if max_length < 8:
            raise ValueError("max_length must be >= 8")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_length = max_length

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False, use_fast=True
        )
        self.model = AutoModel.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False
        ).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model_id = str(self.model_path)

    @torch.inference_mode()
    def __call__(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("cannot embed empty text")
        batch = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        outputs = self.model(**batch, return_dict=True)
        hidden = outputs.last_hidden_state.float()
        mask = batch.get("attention_mask")
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        vector = pooled[0]
        norm = torch.linalg.vector_norm(vector)
        if not bool(torch.isfinite(norm)) or float(norm) <= 1e-12:
            raise ValueError("local embedding model produced a zero or non-finite embedding")
        vector = vector / norm
        result = vector.cpu().tolist()
        if not all(math.isfinite(float(value)) for value in result):
            raise ValueError("local embedding model produced NaN or infinite values")
        return [float(value) for value in result]
