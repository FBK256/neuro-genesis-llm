# Neuro Genesis LLM

![python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-%23EE4C2C?logo=pytorch)
![license](https://img.shields.io/badge/license-Apache%202.0-green)

**Neuro Genesis LLM** は、動的ビン分割による適応埋め込み (adaptive embedding) と  
DeepSpeed ZeRO-3 に最適化した大規模 Transformer 言語モデルの事前学習フレームワークです。  
同梱の **NeuroTokenizer** が BPE トークナイザーの学習／保存を自動化し、  
`neuro_genesis.py` だけで *Tokenizer ▶ Pre-training ▶ Inference* まで完結します。

---

## ✨ 主な特徴

| 機能 | 概要 |
|------|------|
| **Dynamic Bins & Adaptive Embedding** | 単語頻度に応じてベクトル次元を動的縮小し GPU メモリを圧縮 |
| **Multi-layer Transformer** | 8 層・8 ヘッド (デフォルト)。設定クラスで簡単変更 |
| **DeepSpeed ZeRO-3** | ギガバイト級モデルを 1 ～ 2 GPU で学習可能 |
| **Safetensors 保存** | `safetensors` 形式で高速かつ安全にチェックポイント出力 |
| **TensorBoard ロギング** | 損失や学習速度を即時可視化 |
| **簡易検索メモリ (state_memory)** | 推論時平均埋め込みを FIFO 保存し、外部検索との拡張容易 |

---

## 📦 インストール

```bash
git clone https://github.com/<your-account>/neuro-genesis-llm.git
cd neuro-genesis-llm
pip install -r requirements.txt

requirements.txt 例:
torch>=2.0
deepspeed
tokenizers
safetensors
tensorboardX

🚀 クイックスタート
1. トークナイザー学習
from neuro_genesis import NeuroTokenizer
tok = NeuroTokenizer("config/tokenizer.json")
tok.train(["data/general.json", "data/code.json"])

2. 事前学習
from neuro_genesis import NeuroPretrainer
trainer = NeuroPretrainer("config/model.json")
trainer.train(
    dataset_paths=["data/general.tok", "data/code.tok"],
    epochs=10
)

3. 推論
import torch
from neuro_genesis import NeuroGenesisModel, NeuroTokenizer

tok = NeuroTokenizer("tokenizer_config.json")
model = NeuroGenesisModel.from_pretrained("neuro_genesis_final").eval()

prompt = "Q: 富士山の標高は？\nA:"
ids = tok.tokenizer.encode(prompt).ids
with torch.no_grad():
    logits = model(torch.tensor([ids]))
print(tok.tokenizer.decode(torch.argmax(logits, dim=-1)[0].tolist()))

🗂 ディレクトリ構成 (推奨)
neuro-genesis-llm/
├── neuro_genesis.py          # メイン実装
├── config/
│   ├── model.json            # モデルハイパーパラメータ
│   └── tokenizer.json        # トークナイザー設定
├── data/                     # 入力データ
├── checkpoints/              # DeepSpeed チェックポイント
├── requirements.txt
├── LICENSE
└── README.md
