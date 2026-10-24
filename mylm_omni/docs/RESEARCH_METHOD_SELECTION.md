# 追加提案手法の採用・保留基準

この文書は、参考提案の中から限られたGPU時間・ローカル完結・安全な自律実行という制約に合うものを選別する。提案文の記載値は、当リポジトリで実測した性能ではない。

## 結論

| アイデア | 判断 | 理由・実装範囲 |
|---|---|---|
| llama.cpp の量子化VLM | 採用 | ループバック限定のOpenAI互換バックエンドと起動スクリプトを実装。モデルは別途ダウンロードし、対応mmprojを組み合わせる |
| faster-whisper tiny系による音声入力 | 採用 | 音声をローカルASRで文字列化し、認識信頼度・セグメント情報とともにプランナーへ渡す。ASR出力は不確実なデータとして扱う |
| JSON一操作ずつのエージェントループ | 採用 | 登録済みツールのallowlist、引数検証、履歴圧縮、反復制限、最大ステップ数、検証失敗時の再計画を実装 |
| ファイル作成の個別承認 | 採用 | 既定で書き込みを登録せず、明示フラグ＋操作ごとの確認が必要。新規ファイルのみ作成し、既存ファイルを上書きしない |
| ルールベースの推論予算 | 採用（ベースライン） | reflex / deliberative / deep モードを実装。ただし学習済みDifficulty Headではない。固定評価で比較する基準 |
| SQLiteの長期記憶と意味検索 | 採用 | ソース・有効期間・競合レビュー・バージョンを保存。埋め込みモデルはローカルパスから任意に読み込み、未設定時は語彙検索と明示する |
| 失敗からの改善仮説 | 採用（提案まで） | 失敗イベント・根拠ID・テスト計画を永続化。候補は隔離評価・安全テスト・カナリアを通るまで本番に適用しない |
| 成功・失敗修復軌跡からSFTデータを作る | 採用 | scripts/build_agent_sft_dataset.py でサニタイズ済み・出典付き・検証済み軌跡だけを変換。未修正の失敗行動は教師ターゲットにしない |
| Qwen3-VL のQLoRA/SFT | 実用経路として採用 | train_vlm_sft.py はローカル画像マニフェストを扱う。ただし重み・processor・Transformers版・VRAMの互換性を実機で確認する |
| MoR（共有再帰層＋トークン別深度） | 研究経路として採用 | スクラッチモデル側で再帰ブロック、深さルーター、アクティブQueryの推論経路を実装。専用CUDAカーネルや論文のKV最適化は未実装 |
| TAID | 採用候補 | train_taid.py で実験できる。教師・学生のtokenizer/revisionを固定し、通常SFTと同一評価セットで比較する |
| Top-k＋tail蒸留 | 採用候補 | train_sparse_distill.py を実装。Top-k外の各トークン分布は失われるので、dense teacher logitsとの比較を必須にする |
| MiniLLM | 部分採用 | reverse-KL関連の補助関数はあるが、完全なMiniLLM再現ではない。論文相当の目的・生成経路を検証するまでは名前を付けて主張しない |
| TorchAO QAT | 研究用に採用 | カスタムMyLMスクラッチモデル向けの別ステージとして実装。任意のQwen3-VLにそのまま適用するものではない |
| SigLIP＋小型decoder＋PixelShuffle | 将来比較候補 | 視覚エンコーダ分離・投影層の実験として比較価値はある。ただし現在の実用VLM経路と混ぜず、別モデル構成としてパラメータ・画質・学習時間を測る |
| 3BのDifficulty / Value / GUI Pointer head | 保留 | データ・目的関数・キャリブレーション・GUI環境が必要。手書きルールや未学習ヘッドを成功確率として扱うことはしない |
| Hybrid Mamba-2＋Attention＋MoE Top-2 | 保留 | アーキテクチャ変更が多く、限られた学習時間では効果の原因を識別しにくい。まず単一変更のベースラインを完成させる |
| GRPO / RLVR | 保留 | タスク成功を独立判定できる安全な環境、報酬ハックテスト、固定評価、ロールバックが必要。評価ハーネスが固まる前に導入しない |
| 自動のGUI / シェル / ブラウザー操作 | 現在は不採用 | 初期ツールはローカル読取・計算・個別承認付きファイル作成に限定。GUIやコマンド実行を後から入れる場合は別プロセス・OSサンドボックスで隔離する |
| OpenAIクラウドAPI版 | ローカル経路では不採用 | 入力・画像が外部へ送信され得るため、ローカルにデータを留める要件と異なる。必要な場合も別プロファイル・別ポリシーとして明示する |

## 実用システムと研究モデルを分ける

### A. 実用ローカルエージェント

- 推論：llama.cppの量子化済みVLMを127.0.0.1だけにバインド
- 音声：ローカルfaster-whisperで文字起こし
- 制御：JSONで次アクションを1つだけ出力
- ツール：フォルダ内の一覧・テキスト検索・読取・制限付き計算。ファイル作成は個別承認
- 記憶：SQLiteで出典・期間・競合・削除提案を管理
- 改善：失敗記録 → 仮説 → 固定評価 → カナリア → 昇格またはロールバック

これは学習済み3Bモデルの新規開発ではなく、既存VLMを使ったエージェントシステムである。

### B. 独自アーキテクチャ研究

- MyLMスクラッチdecoderと共有再帰ブロックを使い、固定深さ対MoR風深さ配分を比較する
- train_scratch.py はアーキテクチャの実験台。既存ベースモデルの言語能力は引き継がない
- train_vlm_sft.py、train_taid.py、train_sparse_distill.py、train_qat_scratch.py は独立の実験経路であり、すべてを同時に重ねない
- 評価データは学習・改善提案から隔離し、同じモデル・tokenizer・データrevisionをログに残す

## テスト時計算量と「自信度」

モデル自身が出した確信度やverifierの自己判定を、そのまま校正済み成功確率とはみなさない。現在は、ツール結果に照らした別パスの検証を行い、JSON不正・根拠不足なら成功扱いしない。後にconfidence headを試す場合は、独立の正誤ラベルからcalibration curve、Brier score、ECE、リスク別の誤受理率を評価し、信頼できない場合は推論予算を増やすか人間確認へ戻す。

## ローカルサーバー

公式 llama.cpp ドキュメントでは、対応モデルのマルチモーダル入力とOpenAI互換のチャットエンドポイントが説明されている。実際のモデル／mmprojの対応は版によって異なるため、起動後にスモークテストを実施する。

- [llama.cpp multimodal documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md)
- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Mixture-of-Recursions paper](https://arxiv.org/abs/2507.10524)
- [TAID paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e664650506f1cf2b4696df892147c06e-Abstract-Conference.html)
- [MiniLLM paper](https://arxiv.org/abs/2306.08543)
- [TorchAO QAT workflow](https://docs.pytorch.org/ao/stable/workflows/qat.html)

これらの論文や公式マニュアルは設計根拠であり、本リポジトリでの性能再現を意味しない。


### RTX 3090 scratch baseline

- `configs/scratch-300m.json`: around 293M parameters, 256K vocabulary with factorized 384-dimensional tied embeddings, hidden size 1,024, 16 layers, GQA 16Q/4KV.
- Memory: activation checkpointing, chunked vocabulary loss, micro-batch 1.
- Speed: fused AdamW when supported, BF16 autocast, implicit causal mask for right-padded text and throughput based on the actual non-padding token count.
- Stability: warmup/cosine schedule, gradient norm clipping at 1.0, stop on non-finite loss/gradients, periodic checkpoints.
- The 256K position limit is not the initial training sequence length. The old 1.66B model's checkpoints are architecture-incompatible.
