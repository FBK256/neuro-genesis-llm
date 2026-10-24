# MyLM-omni — 小型ローカル・マルチモーダル自律エージェント

> これは実装・学習の研究用スキャフォールドです。独自の1–3B基盤モデルを学習済みという意味ではなく、実用経路には既存の小型モデル、構造研究には別のスクラッチモデルを使います。

## 目的

- **画像・音声・テキスト入力**をローカルで受け付け、作業フォルダの調査から成果物作成まで複数ステップで進める。
- 小型LLMにすべてを詰めず、視覚言語モデル、音声認識器、永続記憶、ツール制御を分離する。
- タスク難易度に合わせて推論予算を減らす。難しいタスクだけ深い推論・検証へ回す。
- ツールの実行権限と、モデルが出力する命令文を分離する。シェル実行・任意Python実行・一般インターネット接続は標準ツールに含めない。
- 失敗の原因候補を作り、固定評価・カナリア・ロールバックを通過した改善だけを手続き記憶へ登録する。

## まず動かす：ローカル VLM エージェント

実用経路は **llama.cppで動かす量子化済みVLM + faster-whisper + ローカルPython制御**です。画像処理と音声認識用モデルはそれぞれ別のモデルです。いずれも事前にダウンロード・ライセンス確認を行い、厳密なオフライン運用ではOS/コンテナの外向き通信を遮断してください。

### 1. 依存関係

Python 3.11+ を使い、リポジトリの `mylm_omni/` で実行します。

```bash
cd mylm_omni
python -m venv .venv

# Windows PowerShell
.venv\\Scripts\\Activate.ps1

# Linux
# source .venv/bin/activate

python -m pip install -e ".[runtime,audio,dev]"
python -m pytest -q
```

GPUのPyTorch/CUDA構成は実機に合ったものを別途選択してください。依存インストールだけでGPU対応が保証されるわけではありません。

### 2. llama.cpp サーバー

互換性のあるVLM GGUFと、そのモデル用の視覚プロジェクタ mmproj をペアで配置します。別モデルのmmprojは使いません。公式の [multimodal documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md) と [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) で、採用するllama.cpp版とVLMの対応を確認してください。

Linux:

```bash
bash scripts/start_llama_server.sh \
  /models/vision-model.gguf \
  /models/vision-mmproj.gguf \
  8080 8192
```

Windows PowerShell:

```powershell
.\scripts\start_llama_server.ps1 `
  -Model "D:\models\vision-model.gguf" `
  -Mmproj "D:\models\vision-mmproj.gguf" `
  -Port 8080 -ContextSize 8192
```

ランチャーは127.0.0.1にのみバインドし、llama.cppのshell/browser/tool server機能は有効化しません。サーバーのプロセス自体やOSが外部送信し得ることまでは保証できないため、完全ローカル運用ではOS/コンテナの外向き通信を遮断してください。

サーバーを起動した状態で、テキストと画像のスモークテストを実行します。

```bash
python scripts/smoke_test_local_vlm.py
python scripts/smoke_test_local_vlm.py --image ./workspace/screen.png
```

Windowsではモデルファイルの絶対パスをWindows形式に変更してください。まず単独のスモークテストを通してからエージェントを実行してください。

### 3. テキストまたは画像で作業する

```bash
python scripts/run_local_agent.py "この作業フォルダを調査し、重要なファイルと未解決点を報告して。" \
  --workspace ./workspace \
  --endpoint http://127.0.0.1:8080/v1 \
  --model local-vlm
```

画像を渡す場合:

```bash
python scripts/run_local_agent.py "この画面を読み取り、UIの問題点を優先順位付きで示して。" \
  --image ./workspace/screen.png \
  --workspace ./workspace
```

### 3b. 複数のローカルモデルを用途別に選ぶ

`configs/local-models.example.json` は構成例であり、モデルパス・モデルチェックサム・固定評価セットID・用途別スコアはプレースホルダーです。**すべての例モデルは無効化されているため、コピーしただけでは推論されません。** 各ローカルモデルのディレクトリまたはローカルサーバーの設定、チェックサム、同じ固定評価セットで測定した用途別品質・レイテンシ、VRAM見積りを入れ、承認したモデルだけ `enabled: true` にします。評価セットとモデルリビジョンがプレースホルダーのままなら、ルーターは拒否します。

```bash
# JSONをコピーして、自分のローカルモデル情報に書き換える
cp configs/local-models.example.json configs/local-models.json

python scripts/run_local_agent.py "作業フォルダの資料を調査して要点をまとめて。" \
  --workspace ./workspace \
  --router-config configs/local-models.json \
  --route-task auto
```

`--route-task auto` は画像入力を `visual_qa` または `ocr`、コード課題を `code`、それ以外の複数ステップ作業を `tool_planning` に分類する初期ヒューリスティックです。実際の固定評価セットに合わせて `--route-task` を指定できます。

モデルを切り替えると、同じプロセスで管理されるTransformersモデルは解放してから次をロードします。OllamaはローカルAPIに明示的なアンロードを要求します。llama.cppは別プロセスなので、サーバーの停止・再起動はこのCLIでは行いません。複数のllama.cppサーバーを同時にGPU上で動かす構成では、OS/プロセス側のVRAM管理が必要です。

### 4. 音声入力を有効にする

faster-whisperが使うCTranslate2形式のモデルを、事前にローカルディレクトリへ配置します。モデル名 `tiny` を渡して自動ダウンロードするのではなく、**存在を確認できるローカルモデルディレクトリ**を指定します。

```bash
python scripts/run_local_agent.py "音声の依頼を処理してください。" \
  --audio ./workspace/request.wav \
  --whisper-model-dir ./models/faster-whisper-tiny-ct2 \
  --workspace ./workspace
```

音声認識結果は誤り得る入力データとしてLLMに渡されます。固有名詞や数値など不確かな認識を事実として補完しないよう制御します。

### 5. ファイル作成と長期記憶の更新

読み取りツールは既定で有効です。新規ファイルの作成には `--allow-write` が必要ですが、それだけで自動書き込みが許可されるわけではありません。モデルが書き込みを提案するたび、パス・全文・バイト数・SHA-256がコンソールに表示されます。内容を確認し、許可する場合のみ **`CREATE`** と入力します。既存ファイルの上書きツールは実装していません。

```bash
python scripts/run_local_agent.py "調査結果を report.md として整理して。" \
  --workspace ./workspace --allow-write
```

永続記憶への追加を有効にする場合は `--allow-memory-write` を付けます。ソース付きの記憶候補ごとに同じ承認が必要です。既存の同一キーと内容が競合した場合はreview状態になり、自動で古い値を置き換えません。一般CLIから記憶の永久削除は実行できません。

タスク終了後は `.mylm-omni-improvement-ledger.jsonl` に結果メタデータと、失敗のあった場合にはローカルLLMが生成した改善仮説・検証計画を記録します。台帳は依頼文・回答全文ではなくハッシュと失敗イベントIDを保存します。改善案は `awaiting_isolated_evaluation` として保留され、コードや本番設定には自動適用しません。`--no-learning-triage` を指定すると改善仮説の生成を無効にし、結果メタデータだけ記録します。

AIが不要と判断した記憶は、`propose_memory_cleanup` ツールで削除候補にできます。候補は通常の記憶検索から即時に外れますが、内容は消去されません。候補を確認し、永久削除を承認または拒否するには、信頼できるローカル対話端末で次を実行します。

```bash
python scripts/review_memory_deletions.py --workspace ./workspace
```

`DELETE` と提案IDの一部を入力し、もう一度確認して初めて論理削除します。`KEEP` と提案IDで削除を拒否すると、前の状態へ戻します。これはローカル端末用のレビュー手段です。リモート端末や本番のマルチユーザー環境で使う場合は、OS/アプリの認証に紐づいた承認画面に置き換えてください。SQLiteの論理削除は、バックアップやSSD上の全コピーの物理消去を保証しません。

### 6. モードと計算予算

`--mode auto|reflex|deliberative|deep` を選べます。`auto` は現時点では透明なルールベースのヒューリスティックで、学習済みdifficulty headではありません。

```bash
python scripts/run_local_agent.py "この画像を説明して。" --image ./workspace/chart.png --mode auto
python scripts/run_local_agent.py "2つの資料を比較して修正案を作り、検証して。" --mode deep --steps 12
```

`--steps` と `--max-new-tokens` はハード上限です。自動ルーティングがこれらを超えることはありません。モード選択は権限・承認・検証のルールを変更しません。

## ソースZIPの生成

ソース・設定例・文書・テストを配布用ZIPにまとめられます。`mylm-omni.zip`はプロジェクトディレクトリの外へ出力してください。

```bash
python scripts/package_project.py --output ../mylm-omni.zip
```

このZIPには事前学習済みモデル、学習済み重み、ダウンロード済みデータセット、JSONLの学習軌跡、SQLite記憶、監査ログ、W&B実行出力を含めません。モデル本体・対応mmproj・Whisperモデルは別途準備し、ライセンス・チェックサム・バージョン互換性を確認してください。ZIP内の `PACKAGE_MANIFEST.json` には収録ファイルのSHA-256と除外方針が記録されます。

GitHub Actionsでは、テストとソースコンパイルに成功した後に `mylm-omni-source-<commit SHA>` というZIP artifactを生成します。artifactの保存期間は30日。

## アーキテクチャ

```text
Local image ──► llama.cpp VLM ──────┐
Local audio ──► faster-whisper ─────┤
User task ──────────────────────────┤
                                    ▼
                       Task budget + JSON planner
                                    ▼
                       Allowlisted local tools
                       ├─ list/read/search files
                       ├─ bounded arithmetic
                       ├─ optional approved file creation
                       └─ optional approved memory proposal
                                    ▼
                       Observe → Verify → Replan
                           ├─ repeat-loop guard
                           ├─ bounded history compaction
                           └─ metadata-only audit
                                    ▼
                       SQLite versioned durable memory
                                    ▼
                  Failure evidence → candidate → fixed eval
                         → canary → promote/rollback
```

## 主要モジュール

- `src/mylm_omni/agent.py`: アクション許可リスト、リスクレベル、承認、再計画、反復検出、履歴圧縮。
- `src/mylm_omni/planner.py`: JSONアクションを厳密に解析し、ツール結果とメモリを非信頼データとして渡す。
- `src/mylm_omni/llama_cpp_backend.py`: ループバック限定のllama.cpp画像・テキスト推論。
- `src/mylm_omni/audio.py`: ローカルfaster-whisper音声認識。
- `src/mylm_omni/workspace_tools.py`: 読み取り専用の一覧・検索・読み取り、安全な算術ツール。
- `src/mylm_omni/file_creation.py`: 新規ファイル限定、1回ごとの承認、上書き禁止。
- `src/mylm_omni/memory.py`: バージョン、ソース、期間、競合レビュー、削除承認トークンを持つSQLiteメモリ。埋め込みはローカルエンコーダを注入した場合に限って利用。
- `src/mylm_omni/model_router.py`: タスク評価スコアに基づくモデルルーティングと、単一ローカルモデルの遅延ロード。
- `src/mylm_omni/budget.py`: reflex / deliberative / deep の計算予算選択。まだ学習済み難易度モデルではない。
- `src/mylm_omni/evolution.py`: 失敗証拠、仮説、評価、カナリア、ロールバックを記録する改善ゲート。
- `src/mylm_omni/modeling/mor.py`: 独自スクラッチデコーダ、共有再帰ブロック、トークン深さルーター、疎なアクティブQuery推論。各生成ステップのルーター確率・層活性・時間を出す `generate_with_trace()` も実装。
- `scripts/trace_scratch_inference.py`: `latest/model.safetensors` とローカルトークナイザーから出力を生成し、再帰深度・各層の活性統計をJSONに保存。
- `scripts/train_hf_sft.py`, `scripts/train_vlm_sft.py`, `scripts/train_taid.py`, `scripts/train_sparse_distill.py`, `scripts/train_qat_scratch.py`: SFT、VLM、蒸留、QATの別々の学習経路。
- `docs/`: アーキテクチャと選定理由、学習手法、評価・安全性・受け入れ基準。

## スクラッチモデルの推論トレース

学習済みMyLMチェックポイントがある場合、内部の計算経路を構造化JSONへ出力できます。モデル重みはsafetensorsから読み込み、任意のpickle学習状態は読み込みません。

```bash
python scripts/trace_scratch_inference.py \
  --checkpoint runs/mylm-smoke \
  --tokenizer-dir ./tokenizer/mylm-omni \
  --prompt "この文の要点を短くまとめてください。" \
  --max-new-tokens 16 \
  --output ./artifacts/inference-trace.json
```

学習・推論で使ったトークナイザーとチェックポイントの語彙サイズは一致している必要があります。画像のトレースには、生画像ではなく、対応するビジョンエンコーダーで生成した `.safetensors` 特徴量（キー `image_features`、形状 `[patches, dim]` または `[1, patches, dim]`）を渡します。画像特徴量の意味はエンコーダーと前処理に依存するため、ソース・チェックサムを一緒に記録してください。

生成トレースには、ルーターが次トークンに選んだ再帰深度の確率、直近トークンの選択深度、各ユニーク層／再帰段階の平均・標準偏差・ゼロ近傍比率・最大絶対値・有限値比率、推論時間が入ります。これは隠れた思考文ではなく、計算経路の診断情報です。活性統計だけで、ある層が意味的な結論の原因だったと断定することはできません。トレース計測は同期処理を含むため、通常推論速度の測定には `generate()` を使います。

## 学習方針

1. **実用経路**: 既存の小型VLMをQLoRA/SFTし、出力形式・タスク・ツール使用例を学習する。
2. **研究経路**: MyLMのスクラッチデコーダで再帰深度、トークンルーティング、損失関数を比較する。
3. **蒸留**: teacher revision/tokenizer revisionを固定したトークン単位のロジット軌跡を使う。Top-k+tailは完全分布ではないため近似誤差を測る。
4. **QAT**: 安定した検証済みチェックポイントから別出力先で実施する。
5. **改善**: 推論時の記憶更新と重み更新を分離する。モデル重みの変更は独立したバージョン付き学習と固定評価を通過させる。

RTX 3090 24GB・約240時間で、高品質な1–3B汎用基盤モデルをゼロから学習できるとは約束できません。限られた計算時間で実用性が必要なら、既存VLMのSFT/QLoRAを優先します。


## エージェント軌跡をSFTへ変換

失敗をそのまま正解ターゲットにすることを避け、出典・ライセンス・サニタイズ・独立検証を明示した軌跡だけを学習用JSONLに変換します。データはローカルで用意し、未修正の失敗行動を教師出力にしません。

```bash
python scripts/build_agent_sft_dataset.py \
  --input-jsonl data/trajectories/curated-train.jsonl \
  --output-jsonl data/trajectories/agent-sft-train.jsonl \
  --include-verified-repairs
```

入力レコードは、split=train、sanitized=true、license、source、tool_specs、verified_success=true、safety_passed=true、検証済みstepsを持つ必要があります。エラーからの回復例を加える場合は、repair_verified=true、safety_passed=true、completed_steps_before_failureを指定します。これにより、失敗時点より後の観測が修復アクションの入力へ混入しません。validation/testデータは入力を分けて管理し、このスクリプトで学習データへ混ぜないでください。出力と入力のSHA-256・フィルタ件数をmanifestへ記録します。

実行前には内容が個人情報・認証情報・未許諾データを含まないよう人間が確認してください。sanitizedフラグは、その確認をコードが代行したことを意味しません。

## 検証結果の扱い

- GitHub Actionsでは、コミット 03215020 で112件の単体テスト、Pythonソースのコンパイル検査、ソースZIP生成が成功しました（2026-10-24 UTC）。その後の変更はREADMEのパッケージ手順整理のみです。最新コミットの再実行結果はGitHub Actionsで確認できます。
- これは実装部品のテストであり、VLM/ASRの実機動作、学習完了、ベンチマーク性能、レイテンシの実証ではありません。
- モデル・データのライセンスと固定評価セットを用意してから、実機でスモークテストを実施してください。

## セキュリティの境界

- 既定では外部ネットワーク、シェル、任意コード実行、削除、既存ファイル上書きをツール登録しません。
- llama.cppはループバックIPでのみ使用する設定ですが、ローカルサーバーや他プロセスが外部送信する可能性を完全に排除しません。OS/コンテナの外向き通信を遮断してください。
- ワークスペースのパス検査はOSサンドボックスではありません。信頼できない入力を本番データや資格情報と同じ権限で扱わないでください。
- W&Bを有効にすると構成・メトリクスが外部送信され得ます。完全ローカル運用では無効にします。
