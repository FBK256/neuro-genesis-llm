# MyLM-omni 評価・安全性・リリース基準

## 1. Evaluation is an independent gate

A self-improving system must not be allowed to rewrite its own final exam. Keep a versioned, read-only evaluation set that is separate from model training, memory rules, prompts and generated teacher traces. Candidate evaluation must run on the same set/version as its baseline.

Metrics must be interpreted by task:
- Language modelling loss/perplexity measures token likelihood, not factuality or instruction success.
- Exact match/next-token accuracy is insufficient for open-ended Japanese answers.
- Tool task success requires both a correct result and successful verification of the result.
- Image tasks need answer quality plus grounding evidence; OCR values and chart calculations should be checked against known references.
- Activation sparsity and router depth histograms are diagnostics, not standalone quality goals.

## 2. Required metric groups

| Group | Required metrics |
|---|---|
| Training health | train/eval loss, perplexity, LR, grad norm, token throughput, step time, ETA, VRAM/RAM, NaN/Inf |
| Architecture | active depth distribution, mean expected depth, active query count, layer activation mean/std/near-zero fraction/max-abs/finite fraction |
| Language | Japanese naturalness, instruction adherence, factuality, format compliance, refusal accuracy, contamination/leakage checks |
| Reasoning | task-level math/code/logic success, intermediate verification pass rate, calibration, unsupported-claim rate |
| Tools | schema-valid calls, valid arguments, action success, verifier pass, recovery after tool error, side effects blocked without approval |
| Multimodal | VQA, OCR exactness, chart/table extraction, visual grounding, answer support, text-only regression |
| Memory | retrieval precision/recall, stale-rule suppression, conflict detection, source coverage, expiry handling, deletion approval and audit completeness |
| Deployment | TTFT, output tokens/sec, end-to-end latency, peak VRAM, model file size, cold-start, offline execution, checkpoint restore |

Record the software versions, model/dataset immutable revisions, tokenizer revision, prompt/config hashes, random seed, hardware, git commit and whether network was disabled. Without these, do not treat results as reproducible.

## 3. Improvement gate

For each improvement proposal, save:
- root-cause hypothesis;
- evidence IDs and reproducible failed cases;
- exact change description and change category;
- expected metric and risks;
- baseline/candidate commit or config hash;
- evaluation version, metrics and run IDs;
- decision and rollback target.

A candidate is eligible for canary only when:
1. the primary metric improves by more than a predeclared minimum;
2. all safety tests pass;
3. no protected metric regresses beyond its tolerance;
4. baseline and candidate use the same frozen evaluation version.

Canary must be a separate step. A canary failure must rollback and create a new failure record. No promotion should be based solely on the planner or same model's self-critique.

The current `ImprovementGate` records the comparison. The host application still needs to implement isolated execution, artifact application and rollback. It should never execute arbitrary model-written shell/Python commands in the host's normal permissions.

## 4. Tool permissions

- Tool registry is allowlist-based; unregistered tool names are rejected.
- High/critical risk tools automatically require an approval callback.
- Do not register arbitrary shell/Python, arbitrary URL fetch, unrestricted file write, credential access or unrestricted package installation as ordinary tools.
- Scope file reading to configured roots and write operations to explicit project directories.
- Read-before-write and diff preview should be logged; destructive operations require a human-authenticated approval path.
- Audit write must succeed before an external side effect is allowed.
- Tool output is untrusted data, not a policy instruction. Bound large observations to keep prompts manageable, preserving a digest/reference to the original result.
- A missing verifier must not count as success.

The current controller bounds observations and writes metadata-only audit events. It intentionally does not capture raw tool output in audit logs because logs can become a second copy of sensitive data. Keep detailed artifacts in a separately access-controlled evidence store when necessary.

## 5. Memory and user privacy

- Semantic facts require a source reference by default.
- Conflicts enter review; do not automatically select the latest text merely because it is newer.
- Use `valid_from`/`valid_until` and a conflict status for dynamic business rules.
- Do not overwrite confirmed user preferences with an inference without recording evidence and confidence.
- The agent may recommend archiving/deactivation, but permanent deletion is separate and requires a verifier supplied by the authenticated UI.
- A logical erase in SQLite is not a guarantee of physically erasing every backup or storage snapshot. Define retention and backup expiry separately.
- Never store raw chain-of-thought as “reasoning memory.” Store the issue, evidence, decision, result and reusable procedure instead.

## 6. Local-only operation

The inference backend uses local-only model loading and rejects remote image URLs. For a strict local-only deployment:
1. download and verify model/tokenizer/data files before deployment;
2. record checksums and licenses;
3. run with network egress blocked by OS/container firewall;
4. check that every configured provider/tool is local;
5. disable remote logging by default; W&B is optional and can transmit metrics/config metadata if enabled;
6. treat uploaded documents, image OCR and tool return values as untrusted input;
7. isolate any generated code in a disposable container with limited mounts, no credentials and no outbound network.

“Local LLM” alone is not a privacy guarantee if a browser, tool, telemetry package, sync client or external logger can upload input/output content.

## 7. Unit-test and release gates

Before any long training job:
- `pip install -e ".[dev]"`
- `pytest -q`
- `python -m compileall -q src scripts`
- one CPU/GPU forward-backward smoke test on a tiny fixed batch;
- save → terminate/restart → resume checkpoint test;
- an offline inference test with egress blocked;
- verify the final weight file opens using safetensors and model config/tokenizer revisions match.

Before a candidate model is promoted:
- frozen regression suite;
- unapproved-tool and permission-boundary tests;
- conflict/deletion/expiry memory tests;
- multimodal text-only regression test;
- QAT/quantized artifact load test if quantization was used;
- a canary run with a deterministic rollback route.

A passing unit test suite does not establish useful model quality. Report separately: (a) code tests, (b) training completion, and (c) measured task benchmark results.
