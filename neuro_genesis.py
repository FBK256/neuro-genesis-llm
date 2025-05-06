import os
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, processors
from safetensors.torch import save_model, load_model
from deepspeed import init_distributed, initialize
from tensorboardX import SummaryWriter
from pathlib import Path
from typing import List, Dict, Optional
from collections import deque, defaultdict

# 設定クラス
class NeuroGenesisConfig(PretrainedConfig):
    model_type = "neuro-genesis"
    
    def __init__(
        self,
        vocab_size=65536,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=4096,
        embedding_dim=512,
        adaptive_embedding_ratio=0.25,
        dynamic_bins=4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.embedding_dim = embedding_dim
        self.adaptive_embedding_ratio = adaptive_embedding_ratio
        self.dynamic_bins = dynamic_bins

# モデルアーキテクチャ
class NeuroGenesisModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 埋め込み層
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # トランスフォーマーレイヤー
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                activation=F.gelu,
                batch_first=True
            ) for _ in range(config.num_hidden_layers)
        ])
        
        # 出力層
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.state_memory = deque(maxlen=5)
        self.search_client = None

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, debug: bool = False) -> torch.Tensor:
        # 埋め込み処理
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(seq_len, device=input_ids.device).expand(batch_size, -1)
        x = self.embedding(input_ids) + self.position_embed(positions)
        
        # トランスフォーマー処理
        for layer in self.layers:
            x = layer(x, src_mask=attention_mask)
            if debug:
                print(f"Layer Output - Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
        
        # 状態記憶
        self.state_memory.append(x.mean(dim=1).detach())
        
        return self.lm_head(x)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config = self.config.to_dict()
        config["architecture"] = "NeuroGenesisModel"
        self.config.save_pretrained(save_directory)
        save_model(self.state_dict(), f"{save_directory}/model.safetensors")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config = NeuroGenesisConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        state_dict = load_model(f"{pretrained_model_name_or_path}/model.safetensors")
        model.load_state_dict(state_dict)
        return model

# トークナイザー
class NeuroTokenizer:
    def __init__(self, config_path: str):
        self.tokenizer = Tokenizer(models.BPE())
        self.special_tokens = [
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            "<|im_start|>", "<|im_end|>", "<math>", "</math>", "<code>", "</code>"
        ]
        self._load_config(config_path)
        self._setup_processor()

    def _load_config(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)

    def _setup_processor(self):
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
        )

    def train(self, dataset_paths: List[str]):
        texts = []
        for path in dataset_paths:
            with open(path) as f:
                data = json.load(f)
                texts.extend([item["text"] for item in data])
        
        tmp_file = Path("corpus.txt")
        tmp_file.write_text("\n".join(texts))
        
        trainer = trainers.BpeTrainer(
            vocab_size=self.config["vocab_size"],
            min_frequency=self.config["min_frequency"],
            special_tokens=self.special_tokens,
            show_progress=True
        )
        
        self.tokenizer.train(files=[str(tmp_file)], trainer=trainer)
        self._save_tokenizer()

    def _save_tokenizer(self):
        self.tokenizer.save("neuro_tokenizer.json")
        meta = {
            "model_type": "NeuroGenesis",
            "special_tokens": {tok: self.tokenizer.token_to_id(tok) for tok in self.special_tokens}
        }
        with open("tokenizer_config.json", "w") as f:
            json.dump(meta, f)

# データセット
class PretrainDataset(Dataset):
    def __init__(self, tokenizer, dataset_paths: List[str], max_length: int = 4096):
        self.data = []
        self.pad_id = tokenizer.token_to_id("[PAD]")
        
        for path in dataset_paths:
            with open(path) as f:
                data = json.load(f)
                tokenized = [tokenizer.encode(item["text"]).ids for item in data]
                self.data.extend([ids[:max_length] for ids in tokenized])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        padded = np.full(self.max_length, self.pad_id, dtype=np.int32)
        seq_len = min(len(tokens), self.max_length)
        padded[:seq_len] = tokens[:seq_len]
        return torch.tensor(padded)

# 事前学習
class NeuroPretrainer:
    def __init__(self, config_path: str):
        self.config = NeuroGenesisConfig.from_pretrained(config_path)
        self.model = NeuroGenesisModel(self.config)
        self.tokenizer = NeuroTokenizer("tokenizer_config.json")
        self.writer = SummaryWriter()
        
        self.ds_config = {
            "train_batch_size": 32,
            "gradient_accumulation_steps": 2,
            "optimizer": {
                "type": "FusedAdam",
                "params": {
                    "lr": 2e-4,
                    "betas": [0.9, 0.98]
                }
            },
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": 3}
        }

    def train(self, dataset_paths: List[str], epochs: int = 10):
        dataset = PretrainDataset(self.tokenizer, dataset_paths)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        engine, optimizer, _, _ = initialize(
            model=self.model,
            config_params=self.ds_config,
            model_parameters=self.model.parameters()
        )

        for epoch in range(epochs):
            total_loss = 0
            start_time = time.time()
            
            for i, batch in enumerate(loader):
                loss = engine(batch).loss
                engine.backward(loss)
                engine.step()
                
                total_loss += loss.item()
                self.writer.add_scalar("Loss/train", loss.item(), epoch * len(loader) + i)
                
                if i % 10 == 0:
                    self._print_progress(epoch, i, loss.item(), time.time() - start_time)
                    
                if i % 100 == 0:
                    self._save_checkpoint(engine, epoch, i)

            print(f"\nEpoch {epoch+1} | Avg Loss: {total_loss/len(loader):.4f}")

    def _print_progress(self, epoch: int, step: int, loss: float, elapsed: float):
        progress = step / len(self.loader) * 100
        print(f"\rEpoch {epoch+1} | Step {step} ({progress:.1f}%) | Loss: {loss:.4f} | Time: {elapsed:.1f}s", end="")

    def _save_checkpoint(self, engine, epoch: int, step: int):
        ckpt_dir = f"checkpoints/epoch{epoch}_step{step}"
        engine.save_checkpoint(ckpt_dir)
        self.model.save_pretrained(ckpt_dir)

# 実行
if __name__ == "__main__":
    # トークナイザー訓練
    tokenizer = NeuroTokenizer("config/tokenizer.json")
    tokenizer.train(["data/general.json", "data/code.json"])
    
    # 事前学習
    pretrainer = NeuroPretrainer("config/model.json")
    pretrainer.train(
        dataset_paths=["data/general.tok", "data/code.tok"],
        epochs=10
    )
    
    # 最終モデル保存
    pretrainer.model.save_pretrained("neuro_genesis_final")