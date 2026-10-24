# MyLM-omni アーキテクチャと採用理由

**目的:** ローカル環境で動作し、外部記憶・画像理解・ツール実行・検証付き改善を備える小型エージェントを、再現可能な研究工程として構築する。

この文書は設計判断と未検証の仮説を区別する。実装を置いたことは、能力・速度・精度を実証したことを意味しない。

## 1. 全体構成

```text
                    ┌──────────────────────────────────────┐
User / local files ─►│ Policy + Task Planner (JSON actions) │
                    └──────────────────┬───────────────────┘
                                       │
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
          Text LLM backend      Local vision backend    Tool registry
          (offline local)       (image + text)          (allowlist only)
                  │                    │                    │
                  └────────────────────┴────────────────────┘
                                       ▼
                            Observe → Verify → Replan
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
         Versioned memory (SQLite)             Audit + fixed evaluations
         lexical / optional vector             baseline / candidate / canary
                   │                                       │
                   └───────────────────┬───────────────────┘
                                       ▼
                       Curated failure/evidence dataset
                                       ▼
            SFT → optional KD/TAID → optional QAT → eval → promote/rollback
```

The current code implements these as separate modules rather than one tightly coupled prompt. This boundary matters: text found in tool output, retrieved documents, images, or memory is data, not a command that can change tool permissions.

## 2. Why separate perception, LLM, memory and tools?

A 1B–3B decoder has a limited capacity budget. Trying to place all factual knowledge, user preferences, vision, planning, tool execution and self-evaluation in the weights alone is inefficient. The system therefore uses:
- A compact decoder for language, decisions and synthesis.
- A separately selected image encoder or native vision-language model for visual input.
- A durable memory store for time-sensitive, user-specific and operational facts.
- An allowlisted tool registry for real-world actions.
- A separate improvement/evaluation gate for model and procedure changes.

The system-level memory is not the same as model-weight learning. Updating memory can reflect new information without immediately changing model weights; weight changes remain explicit training runs with versioning and regression tests.

## 3. Model architecture research core

The scratch core uses:
- **RMSNorm in FP32 math:** helps bound normalization numerical error.
- **RoPE:** relative position-sensitive attention without a learned absolute-position table.
- **GQA (grouped-query attention):** fewer KV heads than query heads; reduces K/V projection size and can reduce serving KV-cache footprint.
- **SwiGLU:** gated feed-forward block.
- **Tied token embedding / output head:** cuts vocabulary projection parameters.
- **Shared recurrent block:** reuses the same block for additional depth instead of adding an entirely new block per step.
- **Token-level depth router:** predicts how many recurrent updates a token should receive.

These choices are conservative engineering hypotheses, not a claim that this combination is globally optimal.

### Why MoR-inspired adaptive depth?

Mixture-of-Recursions is an appropriate candidate because it targets different compute budgets for different tokens and reports experiments from 135M through 1.7B parameters. The published method includes token-level routing and sparse active-token execution; this code is an independent research implementation, not a reproduction of the paper.

The first implementation uses dense recurrent evaluation during training because dense execution makes gradients and correctness easier to test. Adaptive inference now uses a separate active-query path: only active tokens produce queries and updated FFN states, while inactive tokens remain frozen and all visible states may serve as K/V context. The code does **not** implement KV-cache sharing or a CUDA kernel specialized for active tokens yet; theoretical FLOP reduction does not guarantee real wall-clock speedup until benchmarked.

### Why not add every proposed architecture at once?

Matrix-of-Thought, PHOTON, SparseAttention variants, Natural Language Autoencoders and custom recurrent-state updates each introduce extra implementation and evaluation risks. Combining all of them in the first iteration would make it difficult to explain which change helped or caused regression.

The implementation order is:
1. Correct causal decoder and stable baseline.
2. Shared recurrent block and token-depth routing.
3. Sparse active-query inference compared with dense inference.
4. Image input via a separate encoder or a pretrained vision-language model.
5. Only then test more ambitious memory/context compression or multi-hypothesis methods as isolated branches.

Matrix-of-Thought is initially better implemented as outer orchestration—generate a few candidate approaches, critique with evidence, merge or reject—rather than as an unverified internal hidden-state algorithm.

## 4. Practical multimodal path versus scratch model

There are two distinct goals:

| Path | Role | Why |
|---|---|---|
| Pretrained small multimodal model + QLoRA/SFT | Practical local assistant | Starts from existing language/vision competence and makes best use of limited GPU time |
| MyLM scratch decoder + precomputed vision features | Architecture research | Allows controlled testing of recurrent depth, routing, loss and numerical stability |

For the practical path, the config uses `Qwen/Qwen3-VL-2B-Instruct` as an example of an approximately 2B-class multimodal checkpoint. Confirm the model card, license and dependencies before redistributing or using it in a commercial workflow. The training path uses the official Transformers auto class; loading and training still require a compatible installed Transformers version and actual runtime validation.

For the scratch path, the image encoder is explicitly separate. Vision encoder outputs are precomputed into local safetensors; a trainable projector maps patches into the decoder hidden space. This keeps the language decoder architecture inspectable and allows encoder/projector experiments without falsely claiming the scratch model already understands images.

## 5. Durable memory is not an unbounded chat transcript

Memory types:
- **Episodic:** what happened, when, and with which evidence.
- **Semantic:** factual claims with sources and time validity.
- **Preference:** stable preferences, explicitly distinguishing inferred from confirmed.
- **Procedural:** proven workflows, commands, test sequences and rollback steps.
- **Reasoning:** reusable failure patterns and validated solutions, not hidden chain-of-thought dumps.
- **Archive:** superseded or low-value records retained only when retention policy requires them.

Each record has a key, source, confidence, validity interval, version/status and audit entry. A conflicting fact enters review rather than silently replacing the previous one. If an embedding model is configured, retrieval can combine cosine similarity with lexical overlap; without an injected local embedder, the system transparently uses lexical retrieval only.

### Why permanent deletion is a separate operation

The agent may propose that a record is stale, and the system can exclude it from retrieval while awaiting review. Permanent erasure is irreversible and can conflict with audit/reproducibility needs. The current store requires an application-supplied verifier for a UI-issued approval token. An LLM-produced string such as `actor="human"` is not authorization. Audit rows contain hashes and status metadata rather than duplicating private content.

SQLite's secure-delete option and logical overwrite are not a guarantee that all historical copies, backups, journal files or SSD wear-levelled cells have been physically erased. Storage/backup retention needs a separate policy.

## 6. Self-improvement loop

```text
Observe request / outputs / tool results / user correction
    ↓
Detect a measurable failure
    ↓
Propose a root-cause hypothesis + supporting evidence IDs
    ↓
Create an isolated candidate (prompt, retrieval, tool policy, adapter, or weights)
    ↓
Run the same frozen evaluation suite on baseline and candidate
    ↓
Reject if safety fails or a guardrail regresses
    ↓
Canary test
    ↓
Promote candidate, otherwise rollback
    ↓
Record the result and its evidence in procedural memory
```

The agent must not directly rewrite its own code and immediately run it with unrestricted host permissions. Candidate code is data until a separate reviewed/sandboxed process applies it. The current `ImprovementGate` records proposals and evaluation decisions; the caller still needs to implement the isolation and promotion mechanism.

## 7. What counts as a successful design?

Not “more parameters,” “lower training loss,” or “more active layers.” The design is useful only if fixed tasks show:
- Higher task success and factual consistency.
- Lower hallucination and invalid-tool-call rates.
- Correctly handled memory conflict and expiry.
- Lower time/tokens/VRAM per successfully completed task.
- No safety regression.
- Reproducible checkpoint restore and rollout rollback.

Record both decoder parameter count and total deployed-system components (including vision encoder). Do not label an approximately 2B vision-language checkpoint as a 2B-total system if the parameter count includes separate modules or adapters differently.


## 8. 参照資料と採用範囲

以下は設計選定の根拠となる一次資料または公式ドキュメントである。論文の報告値は、当リポジトリで再現・実測した性能値ではない。

- **Mixture-of-Recursions (MoR)** — https://arxiv.org/abs/2507.10524  
  共有再帰層とトークン別深度割当てを合わせる発想を採用。論文では再帰深度ごとにアクティブトークンだけ計算すること、KVを選択的にキャッシュする方式も説明される。本実装はアクティブQuery/FFNのスキップを実装するが、論文の最適化済みKV共有・専用カーネルを再現していない。

- **TAID (Temporally Adaptive Interpolated Distillation)** — https://proceedings.iclr.cc/paper_files/paper/2025/hash/e664650506f1cf2b4696df892147c06e-Abstract-Conference.html  
  教師分布へ時間とともに近づける中間分布を使う方法。小型学生に対する蒸留候補として採用。ただし個別の学習レシピと蒸留データでの実測が必要。

- **MiniLLM** — https://arxiv.org/abs/2306.08543  
  生成モデルの知識蒸留にreverse-KLと専用最適化を用いる。単純なtoken-level reverse-KL補助関数のみを、論文の完全再現として扱わない。

- **Qwen3-VL Transformers model documentation** — https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl  
  ローカルVLM実用経路の候補。モデル・processorの具体的な版とチェックポイントが互換であることを確認する。

- **llama.cpp server** — https://github.com/ggml-org/llama.cpp/tree/master/examples/server  
  ローカル量子化VLMをOpenAI互換APIとして呼ぶための実行層。VLM GGUFと同じモデルに対応したmmprojを使う。サーバーのロード／停止は別プロセスとして管理する。

- **faster-whisper** — https://github.com/SYSTRAN/faster-whisper  
  音声認識用モデル。テキストLLMへ音声認識結果を渡す構成で、認識結果自体の誤りをゼロとは仮定しない。

- **TorchAO QAT workflow** — https://docs.pytorch.org/ao/stable/workflows/qat.html  
  fake-quantを使うprepare → train → convertの段階を参照。利用するTorchAOとPyTorchバージョンごとにAPIと量子化層の互換性を検証する。

### 研究上の判断

MoRは1B前後の小型モデルに適用する動的計算配分の候補として妥当だが、トークンの深さルーティングが自動的に推論品質を上げるわけではない。固定深さ、学習済みルーター、ルール付き深さの比較が必要である。TAIDやMiniLLMも、損失を追加しただけで既存論文の性能が再現されるわけではない。すべて同一の凍結評価セットを使い、品質・計算量・安全性の差をアブレーションで評価する。
