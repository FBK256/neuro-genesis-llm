# MyLM-omni 学習手法と実行レシピ

## 1. まず選ぶべき学習経路

### 実用モデルを作る経路（推奨）

RTX 3090 24GBでは、既存の小型マルチモーダルモデルに対するQLoRA/SFTから始める。これは新しい1B–3B基盤モデルをゼロから同じ品質で作れるという意味ではない。基盤モデルが既に獲得した言語・画像能力を維持し、固有の仕事・出力形式・ツール手順を学習する経路である。

Dataset line examples:

```jsonl
{"instruction":"この文章を箇条書きで要約してください。","output":"- ..."}
{"instruction":"画像の表を読み、合計値を検算してください。","output":"合計は...","image_path":"..."}
```

For the dedicated Qwen3-VL path, use a separate manifest schema:

```jsonl
{"image_path":"images/sha256.png","instruction":"図表の軸・単位・主な傾向を説明してください。","output":"...","license":"...","source_dataset":"..."}
```

The path must point to a local image. Remote image URLs are refused by the training code. Keep a license/provenance record and exclude held-out evaluation rows from the training file.

Example run on the training machine:

```bash
pip install -e ".[multimodal,dev]"
python -m mylm_omni.training.train_vlm_sft \
  --base-model Qwen/Qwen3-VL-2B-Instruct \
  --train-jsonl data/multimodal/train.jsonl \
  --eval-jsonl data/multimodal/validation.jsonl \
  --output-dir runs/mylm-omni-vlm-2b \
  --max-length 1024 --max-steps 2000 \
  --gradient-accumulation 8 --save-steps 100 --wandb
```

This is a first-run configuration, not a guarantee of fitting every image or every 2B-class model in 24GB. Image resolution, vision token count, attention implementation, optimizer state, LoRA target modules and CUDA/kernel versions determine actual memory. Start with a small smoke test and lower image resolution/context before a long run. The script's default model is an illustrative base checkpoint and must be downloaded/verified in advance. `trust_remote_code=False` is intentional.

### Architecture research path (custom scratch model)

The custom MyLM model exists for controlled experiments. A scratch JSONL example:

```jsonl
{"input_ids":[1,24,53,81,2],"labels":[-100,24,53,81,2]}
```

The JSON tokenizer must be trained first on curated text. Example:

```bash
python scripts/train_tokenizer.py \
  --input-jsonl data/text/curated.jsonl \
  --output-dir tokenizer/mylm-omni \
  --vocab-size 262144
```

The target vocabulary is 262,144. The tokenizer script errors if the corpus produces fewer tokens; add representative text or explicitly pass `--allow-smaller-vocab` for a pilot and copy the actual manifest size into the model config. The positional limit `max_position_embeddings=262144` is separate from training length, which defaults to 1,024. This setting alone does not imply long-context quality.

Then tokenize with a tokenizer-specific preprocessing adapter. The current `prepare_text_dataset.py` creates `input_ids` JSONL for full-sequence language modelling; if training the custom model, ensure `model-config.vocab_size` exactly equals the trained tokenizer size.

Smoke-test on the small pilot config first:

```bash
python scripts/train_scratch.py \
  --model-config configs/scratch-pilot.json \
  --train-jsonl data/text/tokens.jsonl \
  --eval-jsonl data/text/frozen-eval.jsonl \
  --max-steps 20 --max-seq-len 256 \
  --checkpoint-steps 10 --output-dir runs/mylm-smoke
```

The RTX 3090 scratch baseline is `configs/scratch-300m.json`, approximately 293M parameters. It factorizes the 256K vocabulary to 384 dimensions rather than allocating a 262,144 × 2,048 table.

```bash
python scripts/estimate_model.py --model-config configs/scratch-300m.json --activation-gb 8
python scripts/train_scratch.py \
  --model-config configs/scratch-300m.json \
  --train-jsonl data/text/tokens.jsonl \
  --eval-jsonl data/text/frozen-eval.jsonl \
  --max-steps 2000 --batch-size 1 --gradient-accumulation 16 \
  --max-seq-len 1024 --checkpoint-steps 100 \
  --output-dir runs/mylm-scratch-300m --resume never
```

This model is **not checkpoint-compatible** with the previous 1.66B architecture; use a new output directory and `--resume never`. Training uses activation checkpointing, BF16 autocast when supported, fused CUDA AdamW when available, gradient clipping and non-finite loss/gradient checks. Chunked vocabulary loss avoids retaining the full `[batch, sequence, 262144]` logits tensor; its output projection is recomputed during backward, trading some throughput for reduced peak memory.

For lower optimizer memory, install `pip install -e ".[low-memory,dev]"` and use `--optimizer adamw8bit --parameter-dtype bf16`. Start at sequence length 1,024 and batch size 1; increase only after a complete GPU step test. 256K is not a realistic initial training length on 24GB VRAM. Use a length curriculum (1K → 2K → 4K → 8K → longer) and fixed long-context evaluations. Keep 15–25% VRAM headroom and treat the first real step as the final fit test.

## 2. Dataset building and provenance

1. List each dataset ID, split, current license, snapshot/revision, text/image fields, data-source URL and transformation.
2. Refuse unknown or non-approved licenses by default. A dataset's public availability does not mean its redistribution/training terms are open.
3. Deduplicate by normalized text hash, image SHA-256 and near-duplicate screening.
4. Filter malformed records, repeated boilerplate, corrupted images, personal data and unsafe/out-of-scope text according to policy.
5. Build train/validation/test splits *before* generating teacher traces. The fixed test set must not be fed to the improvement loop.
6. Record provenance and filter counts, not just a final row count.

Text download example (check the current model/data card before running):

```bash
python scripts/download_text_datasets.py \
  --dataset DATASET_ID:train \
  --text-fields text --workers 2
```

Image-text download example:

```bash
python scripts/download_multimodal_datasets.py \
  --dataset DATASET_ID:train --image-field image \
  --text-fields caption text --workers 2
```

Replace `DATASET_ID` with a personally reviewed dataset. The registry file is an example, not an approval list. For very large corpora, cap parallel downloads and tokenization workers to avoid thrashing a 1TB SSD or exhausting RAM.

## 3. Learning schedule

| Stage | Objective | Promotion gate |
|---|---|---|
| 0. Data quality | License, deduplication, tokenization, modality checks | Provenance completeness and leakage scan |
| 1. SFT / continued pretraining | Stable next-token and instruction loss | Frozen validation quality, no NaN/Inf |
| 2. Tool / reasoning examples | Valid structured calls, verification and recovery | Tool completion, argument validity, policy tests |
| 3. Distillation | Student learns a fixed teacher's useful distribution | SFT baseline comparison and calibration |
| 4. Multimodal tuning | Ground answers in image evidence | VQA/OCR/chart benchmarks + text regression |
| 5. Optional QAT | Improve quantized-model quality | Pre/post quantization quality and latency |
| 6. Deployment | Safetensors/export and local serving | Cold restart, offline operation and rollback tests |

Use the following objective only as a bookkeeping template; ablate each term independently:

`L = λ_{SFT} L_{SFT} + λ_{KD} L_{KD} + λ_{TAID} L_{TAID} + λ_{depth} L_{depth}`

Do not automatically turn every coefficient on at once. A low training loss can coexist with poor reasoning, weak Japanese naturalness, wrong tool arguments, visual hallucinations or catastrophic regression.

## 4. Distillation

### Teacher trace data

The generator records teacher identity/revision, tokenizer identity/revision, prompt IDs, answer IDs, token-level top-k token IDs/logits, tail log-sum-exp, temperature, modality, task type, source and license.

A teacher record must preserve **per output token** logits. A single top-k row for the whole answer is invalid. Student and teacher must have identical token semantics and compatible vocabulary ordering—not just equal vocabulary lengths.

### Sparse top-k + tail

Top-k plus one aggregated tail-probability bucket preserves the total teacher mass outside top-k but loses distinctions among individual tail tokens. This reduces storage, not information loss. Compare its results against full-logit distillation on a small, representative set.

```bash
python scripts/generate_teacher_traces.py \
  --teacher-model TEACHER_ID \
  --teacher-revision IMMUTABLE_COMMIT \
  --input-jsonl data/instruction/train.jsonl \
  --output-jsonl data/teacher/train-traces.jsonl \
  --top-k 32 --temperature 1.0 --max-length 1024
```

### TAID

TAID's intermediate target distribution is formed from a mixture of the student's detached logits and teacher logits; the interpolation coefficient moves toward the teacher distribution according to a temporal schedule. The provided `train_taid.py` uses dense teacher logits and requires compatible tokenizers/output vocabularies. Its implementation should be compared with the official reference recipe; it is not an assertion of numerical parity with every published setting.

```bash
python scripts/train_taid.py \
  --student-model STUDENT_ID --teacher-model TEACHER_ID \
  --train-jsonl data/instruction/train.jsonl \
  --eval-jsonl data/instruction/validation.jsonl \
  --output-dir runs/mylm-taid \
  --max-steps 1000 --gradient-accumulation 8 --qlora
```

### MiniLLM distinction

MiniLLM is a generative distillation method built around reverse-KL optimization, including student-generated/on-policy sequence behavior. The current token-level `token_reverse_kl_loss` is an ablation helper, **not a complete MiniLLM implementation**. Do not label a run using that loss alone as “MiniLLM reproduction.”

## 5. QAT and quantization

QAT is a late-stage optimization; first obtain a stable, evaluated BF16/FP16 or full-precision checkpoint. Run QAT on a separate branch/output folder so the base checkpoint stays immutable.

```bash
python scripts/train_qat_scratch.py \
  --base-checkpoint runs/mylm-scratch \
  --train-jsonl data/text/tokens.jsonl \
  --eval-jsonl data/text/frozen-eval.jsonl \
  --output-dir runs/mylm-qat-int4 \
  --max-steps 500 --checkpoint-steps 100 --group-size 32
```

The custom TorchAO QAT script is a research scaffold. It must pass unit tests and an end-to-end export/load test against the specific installed TorchAO version before being used for a valuable training run. Do not assume arbitrary Qwen3-VL modules can be quantized by the custom scratch QAT script.

## 6. Hardware-specific starting points

- **RTX 3090 24GB, 64GB RAM:** use QLoRA/SFT on a pretrained 0.6B–1.7B model for practical capability. For scratch research start with `scratch-300m.json`, sequence length 1,024, micro-batch 1 and chunked loss. Do not begin at 256K sequence length.
- **A100 40GB:** longer sequences/larger accumulations after measuring actual memory; keep periodic external checkpoints.
- **A100 80GB / RTX PRO 6000 96GB:** custom scratch architecture and denser-teacher experiments are more plausible; validate throughput and dataset I/O, not only allocated VRAM.
- **Colab:** assume the runtime may terminate unexpectedly. Checkpoint every 100 optimizer steps, persist resumable state to durable storage and verify the checkpoint after writing.

The provided config and max-step values are defaults for a pilot. Once tokens/sec is measured over a representative window, estimate ETA as remaining optimizer steps × measured seconds/step plus evaluation/checkpoint overhead.
