"""Deterministic token dataset utilities for small-batch local training."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, Sampler


class JsonlTokenDataset(Dataset):
    """Index a tokenized JSONL file without holding all token arrays in RAM.

    Each row must contain input_ids (integer token IDs) and may contain labels. If labels
    are omitted, all non-padding tokens are trained as next-token language-model targets.
    The file is opened lazily per process so DataLoader workers do not share a file handle.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.offsets: list[int] = []
        with open(self.path, "rb") as stream:
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(position)
        if not self.offsets:
            raise ValueError(f"token dataset is empty: {self.path}")
        self._handle = None

    def __len__(self) -> int:
        return len(self.offsets)

    def _open(self):
        if self._handle is None or self._handle.closed:
            self._handle = open(self.path, "rb")
        return self._handle

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        handle = self._open()
        handle.seek(self.offsets[index])
        row = json.loads(handle.readline())
        input_ids = row.get("input_ids")
        labels = row.get("labels", input_ids)
        if not isinstance(input_ids, list) or not isinstance(labels, list):
            raise ValueError(f"line at dataset index {index} must have input_ids/labels lists")
        if len(input_ids) < 2 or len(labels) != len(input_ids):
            raise ValueError(f"dataset index {index} has invalid sequence/label length")
        if not all(isinstance(token, int) and token >= 0 for token in input_ids):
            raise ValueError(f"dataset index {index} contains invalid token IDs")
        if not all(isinstance(token, int) and (token >= 0 or token == -100) for token in labels):
            raise ValueError(f"dataset index {index} contains invalid label IDs")
        if not any(token != -100 for token in labels[1:]):
            raise ValueError(f"dataset index {index} has no supervised next-token targets")
        result = {"input_ids": input_ids, "labels": labels}
        feature_path = row.get("image_features_path")
        if feature_path:
            # Precomputed features are loaded from safetensors, never pickle.
            from safetensors.torch import load_file

            path = Path(feature_path)
            if not path.is_absolute():
                path = Path(self.path).parent / path
            if not path.is_file():
                raise FileNotFoundError(f"image feature file not found: {path}")
            feature_data = load_file(str(path), device="cpu")
            if "features" not in feature_data:
                raise ValueError(f"feature file lacks a 'features' tensor: {path}")
            result["image_features"] = feature_data["features"]
        return result


class TokenBatchCollator:
    """Right-pad variable-length token rows and mask padding labels with -100."""

    def __init__(self, pad_token_id: int, max_length: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, rows: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        lengths = [min(len(row["input_ids"]), self.max_length) for row in rows]
        batch_len = max(lengths)
        input_ids = torch.full(
            (len(rows), batch_len), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((len(rows), batch_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), batch_len), dtype=torch.long)
        for index, row in enumerate(rows):
            n = lengths[index]
            input_ids[index, :n] = torch.tensor(row["input_ids"][:n], dtype=torch.long)
            labels[index, :n] = torch.tensor(row["labels"][:n], dtype=torch.long)
            attention_mask[index, :n] = 1
            labels[index, :n][input_ids[index, :n] == self.pad_token_id] = -100
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        feature_rows = [row.get("image_features") for row in rows]
        if any(feature is not None for feature in feature_rows):
            feature_size = next(feature.shape[-1] for feature in feature_rows if feature is not None)
            if any(feature is not None and feature.shape[-1] != feature_size for feature in feature_rows):
                raise ValueError("image feature sizes differ within the batch")
            max_patches = max(
                (int(feature.shape[0]) for feature in feature_rows if feature is not None), default=0
            )
            dtype = next(feature.dtype for feature in feature_rows if feature is not None)
            image_features = torch.zeros(
                (len(rows), max_patches, feature_size), dtype=dtype
            )
            image_mask = torch.zeros((len(rows), max_patches), dtype=torch.long)
            for index, feature in enumerate(feature_rows):
                if feature is None:
                    continue
                if feature.ndim != 2:
                    raise ValueError("each image feature tensor must have shape [patches, features]")
                count = feature.shape[0]
                image_features[index, :count] = feature
                image_mask[index, :count] = 1
            batch["image_features"] = image_features
            batch["image_attention_mask"] = image_mask
        return batch


class EpochRandomSampler(Sampler[int]):
    """Deterministic per-epoch permutation that can be regenerated after a restart."""

    def __init__(self, dataset: Dataset, *, seed: int, epoch: int = 0) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.dataset), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.dataset)
