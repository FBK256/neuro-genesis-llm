import json

import torch

from mylm_omni.training.data import JsonlTokenDataset, TokenBatchCollator


def test_jsonl_dataset_and_collator_mask_padding(tmp_path):
    path = tmp_path / "tokens.jsonl"
    path.write_text(
        json.dumps({"input_ids": [1, 2, 3]}) + "\n" +
        json.dumps({"input_ids": [4, 5]}) + "\n",
        encoding="utf-8",
    )
    dataset = JsonlTokenDataset(path)
    batch = TokenBatchCollator(pad_token_id=0, max_length=8)([dataset[0], dataset[1]])
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"][1, 2].item() == -100
    assert torch.equal(batch["input_ids"][0], torch.tensor([1, 2, 3]))
