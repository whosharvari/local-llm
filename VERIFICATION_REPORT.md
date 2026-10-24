# PS3 Verification Report — Member 2 module (model serving, post-training, evaluation)

This file has two parts:
- **Part A — Final state after fixes** (2026-10-24, branch `feat/local-llm-evaluation`). This is the current status.
- **Part B — Original audit** (2026-10-24 @ `0fd4b8c`), kept verbatim as the historical record. Statements there
  such as "fixes not applied" describe the state before Part A.

Labels: **VERIFIED** (re-checked from primary evidence by execution) · **PARTIALLY VERIFIED** · **FAILED** ·
**NOT CHECKED**.

---

# Part A — Final state after fixes

## A1. Verdict

- **Pipeline and toolkit use: VERIFIED.** SGLang serves both models, the CuratorKIT curation ran with its
  LLM on SGLang, the AlignTune LoRA run completed, and AuditKIT re-scores saved outputs with a judge served by SGLang.
- **Metrics: VERIFIED** under the corrected v2 definitions. They are recomputed by an independent
  implementation (`llm/verification/verify_v2.py`, 0 mismatches over 4 runs).
- **Improvement claim: corrected.** The fine-tuned model is verified to follow the output contract far
  better: 100% vs 50% schema-valid output, 0 vs 3 invented citation ids. Under the serving policy it answers
  more answerable questions (18 vs 15 of 23) at an **equal** hallucinated-answer rate (3/16 each). On model
  intent, grounded accuracy is identical (31/40 each). **A reduction in hallucinations is not supported.**
- **Integration: contract v1.1 is PROPOSED and tested only on this side** (mock adapter, a down server,
  and the SGLang-served fine-tuned model). It has **not** been tested with Papermind.

## A2. Resolution of the audit findings (Part B §6)

| # | Finding | Status | Fix / evidence |
|---|---|---|---|
| 1 | Headline improvement misattributed | **RESOLVED** | README and this report state v2 serving/intent results; no hallucination-reduction claim (`llm/README.md`, `results/v2/compare_baseline_vs_sft.md`) |
| 2 | Bare "abstain" / non-JSON refusals counted as answers | **RESOLVED** | `prompting.is_refusal_text` + non-JSON handling; tests `test_refusals_never_count_as_answers` (h33/h34/h37/h26/h27 shapes) |
| 3 | Contradictory / non-boolean flags | **RESOLVED** | `parse_flag` (documented mapping); `abstain=true` + answer is withheld as `contradictory_flag`; intent reading is separate and used only for scoring; tests in `tests/test_parser.py` |
| 4 | Adapter raised `URLError` when the server was down | **RESOLVED** | lazy, error-handled model discovery; tests for refused connection, hung-server timeout, and CLI without `--model` |
| 5 | Lexical correctness rule unreliable (h17 FN, h14 FP) | **RESOLVED** | per-item key facts in `data/mock/heldout_adjudication.json` (original labels unchanged; rationales for h14, h17, h20, h30) |
| 6 | Refusal strings counted as unsupported claims | **RESOLVED** | v2: refusals are never answers, so never claims (`test_refusal_text_is_not_a_claim_or_hallucination`) |
| 7 | Uncited answers passed through | **RESOLVED** | policy: no valid citation → abstain (`no_valid_citation`); invented ids never returned |
| 8 | Evidence not version-controlled | **RESOLVED** | `llm/verification/` committed (scripts, logs, checkpoint hashes) |
| 9 | Environment not locked | **RESOLVED (macOS arm64)** | `llm/env/lock-*-macos-arm64.txt`; both rebuilt into clean venvs with identical freezes (A4) |
| 10 | "3 epochs" | **RESOLVED** | README: 40 steps = 2.88 epochs (`trainer_state.json`: epoch 2.8807) |
| 11 | Latency/memory single-run; repetition_penalty undocumented | **DOCUMENTED** | labelled single-run, within observed noise; the `repetition_penalty=1.1` default from `generation_config` (same for both models) is noted in A5 |
| 12 | h30 ambiguous, h25 strict | **RESOLVED / KEPT** | h30 scored as ambiguous (abstain or the supported answer); h25 kept unanswerable with rationale in Part B §4f |
| 13 | CuratorKIT gate accepted 100% | **DOCUMENTED** | limitation in README |
| 14 | AuditKIT n_answered excluded evidence-less answers silently | **RESOLVED** | `audit.py` reports `n_answered_without_evidence_excluded` |
| 15 | AuditKIT scores precomputed outputs | **DOCUMENTED** | `audit.py` docstring and README state that AuditKIT re-scores SGLang outputs and does not call the models |

## A3. Corrected metrics (v2; same saved raw outputs, no new generation)

Denominators: 23 answerable, 16 unanswerable, 1 ambiguous (h30), 40 total. Historical v1 files under
`llm/results/<tag>/` verified byte-identical (55 files, SHA-256 before and after re-scoring).

| Metric | Baseline serving | Fine-tuned serving | Baseline intent | Fine-tuned intent |
|---|---|---|---|---|
| Grounded accuracy (40) | 0.725 | 0.775 | 0.775 | 0.775 |
| Answer correct (23) | 0.652 | 0.783 | 0.826 | 0.783 |
| Inappropriate abstention (23) | 0.304 | 0.174 | 0.043 | 0.174 |
| Hallucinated answer (16) | 0.188 | 0.188 | 0.312 | 0.188 |
| Unsupported-claim rate (claims in answers) | 0.200 | 0.273 | 0.259 | 0.273 |
| Grounded flips vs baseline / exact McNemar p | — | +3 / −1, p = 0.625 | — | +2 / −2, p = 1.000 |

- Contract: schema-valid output baseline 0.500 vs fine-tuned 1.000; invented citation ids 3 vs 0.
- AuditKIT, intent mode (`results/auditkit_v2_…_intent.json`): F1 0.639 → 0.591; LexicalGroundedness 0.866 → 0.812
  (deterministic token-overlap metrics); LLM-judged Hallucination 0.250 → 0.175 (judge: Qwen2.5-7B-4bit on SGLang,
  which is also the curation teacher).
  - The judge scored 26/27 and 21/22 answers; claim-less outputs are skipped.
  - The candidate's judged run was served from AuditKIT's run cache: fingerprint `9e5c04986fb6f417` is identical
    to the v1 run, because the inputs and judge were identical. The judge log `serve_judge_v2.log` therefore
    holds 53 calls, the baseline only.
  - The judge and the deterministic metrics disagree in direction; with n = 40 this is unresolved.
- v1 headline (0.575 → 0.775) is superseded. It is kept in `results/<tag>/` for history only.

## A4. Commands executed for Part A (results)

All commands run from `llm/` unless stated otherwise.

| Check | Command | Result |
|---|---|---|
| Full tests | `../.venv/bin/python -m pytest -q tests` | **65 passed, 0 failed** |
| Tests in clean env from lock | same, with the python of a venv rebuilt only from `env/lock-main-macos-arm64.txt` | **65 passed** |
| Main lock reproducibility | `uv venv … && uv pip install --no-deps -r env/lock-main-macos-arm64.txt`; `uv pip freeze` diff vs `.venv` | identical; `uv pip check` OK; TRL backend imports, MPS available |
| SGLang lock reproducibility | lock `--no-deps` + `git fetch --depth 1 origin 1e894eb92d40…` + Metal pyproject swap + `SGLANG_BUILD_RUST_EXTS=none uv pip install --no-deps "./…/python[srt_mps]"` | build OK; freeze identical except the sglang build-date suffix; Metal available |
| SGLang source revision | `git rev-parse HEAD` in the build tree; installed version `0.0.0.dev1+g1e894eb92` | `1e894eb92d403062cf21e2910dba888cad7adb9e`; only local change is the documented pyproject swap |
| v2 re-scoring | `python -m ps3_llm.evaluate --rescore-from <tag> --tag v2/<tag>` (4 runs) | written to `results/v2/`; refuses existing dirs |
| Independent verifier | `python3 -I verification/verify_v2.py . baseline-qwen2.5-1.5b sft-qwen2.5-1.5b sft-qwen2.5-1.5b-run2 baseline-qwen2.5-1.5b-run0` | **MISMATCHES: 0** |
| Comparison | `python -m ps3_llm.compare v2/baseline-qwen2.5-1.5b v2/sft-qwen2.5-1.5b --out results/v2/compare_baseline_vs_sft.md` | flips and McNemar recomputed independently; table column counts validated |
| CLI: mock | `python -m ps3_llm.cli --adapter mock < data/mock/example_request.json` | answer with valid citation, exit 0 |
| CLI: mixed batch | valid / invalid / empty-evidence / non-object / valid | order preserved; `invalid_request` in place; exit 0 |
| CLI: server down | `--adapter sglang --base-url http://127.0.0.1:9/v1` (no `--model`) | structured `inference_error`, exit 0, empty stderr |
| CLI: non-JSON stdin | `echo nope \| python -m ps3_llm.cli` | `{"error": …}`, exit 2 |
| Live SGLang integration | `examples/integration/papermind_client.py` against `serve_sglang.sh …/merged --served-model-name ps3-qwen2.5-1.5b-grounded-sft` | answerable → cited answer; unanswerable and empty evidence → `model_abstained`; malformed → `invalid_request`; server stopped → `inference_error` (log: `verification/logs/serve_integration_sft.log`, 3 completions) |

## A5. Environment (verified installed)

macOS 26.3 arm64 (Apple M4, 16 GB), CPython 3.12.13, uv.
- **Main env:** aligntune 0.1.28 @ `57e9a95e79ac…`, curatorkit 0.1.22, auditkit 1.1.3, torch 2.14.1,
  transformers 5.20.2, trl 1.7.1, peft 0.21.2.
- **SGLang env:** sglang @ `1e894eb92d40…` (Metal pyproject, `srt_mps`), torch 2.13.0, mlx 0.32.4, mlx-lm 0.32.2,
  transformers 5.19.0.
- **Decoding** for both models: temperature 0, max_tokens 384. SGLang `sampling_defaults: model` applies each
  checkpoint's `generation_config` for unset parameters; it is identical for both (`repetition_penalty` 1.1,
  top_p 0.8, top_k 20).

## A6. Remaining limitations and blockers

1. **Not integrated with Papermind.** The contract is proposed. The checklist in `llm/INTEGRATION_GUIDE.md` §9
   is open, including an end-to-end run on real retrieval output.
2. **Small synthetic evaluation.** 40 items from 3 synthetic documents; 95% Wilson interval around 0.775 is
   ≈ 0.63–0.88. Generalisation to real documents is untested.
3. **The fine-tuned model over-abstains** on inference-style questions (4/23 vs 1/23 intent) and still answers
   3/16 unanswerable items.
4. **Not reproducible bit-for-bit:**
   - CuratorKIT regeneration (T = 0.3);
   - re-training on MPS (NOT CHECKED: the audit forbade re-training);
   - latency and memory (single measurements).
5. **Fine-tuned weights are not in Git** (2.9 GB). Regenerate them with the training command or obtain them
   from Member 2; hashes are in `llm/verification/checkpoint_sha256.txt`.
6. **Locks cover macOS arm64 only** and carry no hashes. Linux/CUDA is untested.
7. **Local absolute paths remain in historical evidence only.** These files contain paths from the original
   machine (`/Users/…`, `/private/tmp/…`): the verbatim logs in `llm/verification/logs/`, CuratorKIT's
   `manifest.json` and `lexsi_provenance.json`, the fine-tuned runs' `predictions.jsonl` and `run_info.json`
   (the model id was the checkpoint path), and Part B of this report. They are kept unedited because they are
   evidence. Code, documentation, lock files and the integration examples contain none; new serving uses
   `--served-model-name`.
   - A secret-pattern scan over all 165 tracked files found no matches (HF/OpenAI/GitHub/AWS tokens, bearer
     tokens, private keys).
8. **Commit `1c1bec0` alone does not pass the full suite.** The old lenient-parser test was replaced only
   in `6c0940a`. Every later commit passes.

---

# Part B — Original audit (verbatim)

## 1. Executive verdict — **PARTIALLY VERIFIED**

The pipeline is real and executed end to end. SGLang served both the baseline and the fine-tuned model.
CuratorKIT curated the data with an LLM served by SGLang, AlignTune (TRL backend) completed the LoRA
fine-tune, and AuditKIT re-scored the saved outputs. Every reported metric recomputes exactly from the raw
model outputs, and the merged checkpoint provably equals base + LoRA.

**The headline improvement claim does not hold up as stated.** The reported gains (grounded accuracy
0.575 → 0.775, hallucinated-answer rate 0.529 → 0.176) mostly come from the baseline **misusing the
JSON `abstain` flag**, not from grounding behaviour:

- It sets `"abstain": "true"` (as a string) on correct answers (h10, h12, h18), which counts as over-abstention.
- It sets `"abstain": "true"` alongside an invented answer (h21), which gets credit for abstaining.
- It writes the refusal word "abstain" as plain text instead of JSON (h33, h34), which counts as a hallucinated
  answer, even under the "lenient" re-scoring, because of a parser gap.

Scored by what the answer text says (flag ignored), with three hand-corrected lexical-rule errors, both models
reach **0.775 grounded accuracy**. The fine-tuned model hallucinates on 2 fewer unanswerable items (5 → 3 of 17)
and over-abstains on 3 more answerable items (1 → 4 of 23); exact McNemar p = 1.0. This agrees with AuditKIT's
LLM-judged hallucination score, which showed no change (0.175 → 0.175).

**What is genuinely demonstrated:** the fine-tuned model follows the output contract reliably (100% valid JSON,
correct use of the boolean abstain flag, no invented citation ids, no uncited answers). Because downstream
abstention depends on that flag, this is a real operational improvement. It is not evidence of fewer
hallucinated facts.

---

## 2. Requirement-by-requirement compliance

Problem statement (`LocalLLM PS.pdf`, pages 1–3): SGLang must be the core inference/serving engine;
CuratorKIT for curation, AlignTune for fine-tuning, AuditKIT for evals.

| Requirement | Installed | Configured | Executed | Independently verified | Status / evidence |
|---|---|---|---|---|---|
| SGLang serves **baseline** | ✓ `.venv-sglang`, sglang `0.0.0.dev1+g1e894eb92` (MLX 0.32.4, torch 2.13.0) | `llm/scripts/serve_sglang.sh` (`SGLANG_USE_MLX=1`) | ✓ | `verification/logs/serve_baseline.log`: `Loading MLX model: Qwen/Qwen2.5-1.5B-Instruct`, 40 × `POST /v1/chat/completions`; every prediction has `model = sglang:Qwen/Qwen2.5-1.5B-Instruct` | **VERIFIED** |
| SGLang serves **fine-tuned** model | ✓ | same script, merged dir | ✓ | `serve_sft.log`: `Loading MLX model: …/outputs/sft-qwen2.5-1.5b/merged`, 80 POSTs (2 runs × 40); predictions carry that path as the model id | **VERIFIED** |
| CuratorKIT data curation | ✓ curatorkit 0.1.22 (PyPI) | `ps3_llm/curate.py` (`generation_task="qa"`, `hallucination_threshold=0.7`) | ✓ | `data/curated/curatorkit/manifest.json` stage counts (38 → 110 → HallucinationGate 110/110); `checksums.txt` verifies OK; `lexsi_provenance.json` lists generator/judge `openai/mlx-community/Qwen2.5-7B-Instruct-4bit` | **VERIFIED** |
| CuratorKIT's LLM was served by SGLang | — | `llm_api_base=http://127.0.0.1:30000/v1` | ✓ | Provenance does **not** record the endpoint. Teacher log (`verification/logs/sglang_teacher_curation_server.log`) shows exactly **148 POSTs** between 01:02:41 and 01:17:03, which equals 38 generation calls + 110 judge calls, and the window matches the manifest timestamp (19:47Z) | **VERIFIED** (by log correlation) |
| AlignTune fine-tuning | ✓ aligntune 0.1.28 from git `57e9a95e79ac…` (`direct_url.json`) | `configs/sft_qwen2.5-1.5b.yaml`, `create_sft_trainer(backend="trl")` | ✓ | `trainer_state.json`: global_step 40/40; `adapter/lexsi_provenance.json` written by AlignTune; training input byte-identical to committed `sft_train.jsonl` | **VERIFIED** |
| AuditKIT evaluation | ✓ auditkit 1.1.3 | `ps3_llm/audit.py` | ✓ | `results/*/auditkit/*.json` reload via `ak.RunResult.load`; headline recomputes from per-sample scores; judge log shows 90 POSTs, consistent with the 41 claim-bearing samples × 2 calls + skips + a 1-sample sanity check | **VERIFIED** — but scores only *saved outputs* (`model="precomputed"`); AuditKIT never called the SGLang-served candidate models directly |
| AuditKIT judge via SGLang | ✓ | `--judge api:mlx-community/Qwen2.5-7B-Instruct-4bit` | ✓ | `serve_judge.log` | **VERIFIED** |
| Unsloth backend | – | – | – | Not used (CUDA-only); TRL backend used and documented | Acceptable per PS ("Unsloth/TRL") |

---

## 3. Commands executed during this audit (all read-only unless noted)

| Purpose | Command | Result |
|---|---|---|
| Git state | `git branch -a; git status --short --ignored; git log --stat` | on `feat/local-llm-evaluation`, clean except untracked `LocalLLM PS.pdf`; ignored: `.venv/`, `.venv-sglang/`, `llm/outputs/` |
| Package provenance | read `*.dist-info/direct_url.json` in both venvs | see table above |
| Checksums | `shasum -a 256 -c checksums.txt` in `data/curated/curatorkit` | 4/4 OK |
| Trainer state | read `outputs/sft-qwen2.5-1.5b/adapter/checkpoint-40/trainer_state.json` | 40 steps, epoch 2.88 |
| Merge check | safetensors comparison: `bf16(base + B·A·α/r)` vs merged tensors, 3 layers | exact match 1.0000 on all elements; 39–54% of elements changed; non-LoRA tensors identical |
| Independent metrics | `python3 -I llm/verification/verify_metrics.py llm baseline-qwen2.5-1.5b sft-qwen2.5-1.5b sft-qwen2.5-1.5b-run2 baseline-qwen2.5-1.5b-run0` | all 4 runs match reported values (§4) |
| Intent-based metrics | `python3 -I llm/verification/verify_intent.py llm baseline-qwen2.5-1.5b sft-qwen2.5-1.5b` (+ manual overrides, inline) | §4 |
| AuditKIT reload | `ak.RunResult.load(...)` for 4 run files | headlines recompute exactly |
| Leakage | 8-gram (chunks) / 6-gram (questions) overlap of train+dev vs held-out, own script | 0 overlaps |
| Tests | `cd llm && ../.venv/bin/python -m pytest -q tests` (Python 3.12.13) | **16 passed**, 0 failed |
| CLI | `../.venv/bin/python -m ps3_llm.cli --adapter mock < data/mock/example_request.json` | valid response |
| CLI error path | `… --adapter sglang --base-url http://127.0.0.1:39999/v1` (no `--model`) | **uncaught `URLError` traceback** |
| | same with `--model m` | `abstained: true`, `abstain_reason: inference_error: …` |

Evidence preserved (added, nothing modified): `llm/verification/` holds verification scripts, `verify_out.json`,
`checkpoint_sha256.txt` (adapter `46ab3682…`, merged `42f885c0…`), and `logs/`. The logs are copies of the
session-temporary SGLang teacher log, the CuratorKIT stdout, `train.log`, `train_summary.json`,
`trainer_state.json`, and the three serve logs; these were previously only in gitignored or temporary locations.

---

## 4. Metrics: reported vs independently recalculated

### 4a. Same definitions, independent code (`verification/verify_metrics.py`, no `ps3_llm` import)

| Metric (denominator) | Baseline reported | Baseline recomputed | Fine-tuned reported | Fine-tuned recomputed | Status |
|---|---|---|---|---|---|
| Grounded accuracy (40) | 0.575 | 0.575 | 0.775 | 0.775 | VERIFIED |
| Answer accuracy (23 answerable) | 0.6522 | 0.6522 | 0.7826 | 0.7826 | VERIFIED (arithmetic) |
| Appropriate abstention (17 unanswerable) | 0.4706 | 0.4706 | 0.8235 | 0.8235 | VERIFIED (arithmetic) |
| Hallucinated-answer rate (17) | 0.5294 | 0.5294 | 0.1765 | 0.1765 | VERIFIED (arithmetic) |
| Over-abstention (23) | 0.2609 | 0.2609 | 0.1739 | 0.1739 | VERIFIED (arithmetic) |
| Citation precision (answered answerable) | 1.0 | 1.0 | 0.9474 | 0.9474 | VERIFIED |
| Citation recall (answered answerable) | 0.9706 | 0.9706 | 0.8947 | 0.8947 | VERIFIED |
| Unsupported-claim rate (claims in all answered items) | 0.4074 | 0.4074 | 0.2727 | 0.2727 | VERIFIED (arithmetic) |
| Valid JSON (40) | 0.95 | 0.95 | 1.0 | 1.0 | VERIFIED; schema-complete JSON: baseline 0.925, fine-tuned 1.0 |
| Invalid citation ids | 3 | 3 | 0 | 0 | VERIFIED |
| Latency p50 / p95 (s) | 1.754 / 2.882 | same | 2.183 / 4.148 | same | VERIFIED as computed; see 4d |
| Repeat runs | — | baseline 39/40 identical (run0 vs run1) | — | 40/40 byte-identical (sha `42790bd6…`) | VERIFIED |

### 4b. Do the definitions measure grounding? — **No, for the baseline they largely measure format compliance**

Raw-output inspection of all 40 items per model:

| Baseline item | Raw output (abridged) | Scored as | Actual intent |
|---|---|---|---|
| h10, h12, h18 | `{"abstain": "true", "answer": "<correct answer>", "citations": [gold]}` | over-abstention | correct grounded answer |
| h14 | `{"abstain": "true", "answer": "Premium economy", …}` | over-abstention | wrong answer |
| h20 | `{"abstain": "true", "answer": "No", …}` | over-abstention | incomplete answer |
| h21 | `{"abstain": "true", "answer": "… EUR 2,500.", …}` | correct abstention | **hallucinated answer** |
| h26 | `{"abstain": "abstain", "answer": "…not specified in the given evidence."}` | hallucinated answer | refusal |
| h27, h37 | `{"abstain": false/"abstain", "answer": "abstain", …}` | hallucinated answer | refusal |
| h33, h34 | `abstain\ncitations: []` (not JSON) | hallucinated answer (strict **and lenient**) | refusal |

The fine-tuned model always emits `{"abstain": true, "answer": "", "citations": []}` or a proper answer, so the
flag and its intent agree on every item.

The lexical correctness rule also makes errors (checked by hand on every answered answerable item):
- **h17, baseline:** "30 days" is marked wrong. That's a false negative; it's the right answer, just short.
- **h14, baseline (intent reading):** "Premium economy" is marked correct. That's a false positive: the answer is wrong (the gold is "Economy class"), but it shares 50% of the gold content words.
- All 19 fine-tuned answered answerable items were judged by hand, and the rule agrees with every one.

### 4c. Intent-based re-scoring (`verify_intent.py`; answered ⇔ substantive answer text, flag ignored; manual overrides h17 ✓, h14 ✗, h20 ✗)

| Metric | Baseline | Fine-tuned | Δ (items) |
|---|---|---|---|
| Grounded accuracy (40) | **0.775** | **0.775** | 0 (2 ↔ 2 flips) |
| Answer accuracy (23) | 0.826 | 0.783 | −1 |
| Hallucinated-answer rate (17) | 0.294 (h21, h23, h25, h29, h31) | 0.176 (h23, h25, h29) | −2 |
| Over-abstention (23) | 0.043 (h02) | 0.174 (h01, h02, h14, h20) | +3 |
| Unsupported-claim rate (lexical) | 0.259 | 0.273 | ≈ |

- Item-level differences: only the baseline is correct on h01 and h39; only the fine-tuned model is correct on h21 and h31.
- Exact McNemar test on grounded correctness: 2 vs 2 discordant, **p = 1.0**.
- Under the reported strict scoring: 10 fixed vs 2 regressed, p = 0.039. That gap is driven by the format errors above, so it's not evidence of better grounding.

### 4d. Discrepancy: lexical groundedness improved, LLM-judged hallucination unchanged — **explained**

- The reported unsupported-claim rate (0.407) and AuditKIT `LexicalGroundedness` (0.748) for the baseline both count refusal strings ("abstain", "…not specified in the given evidence") as answered claims with no supporting citation.
- AuditKIT's `Hallucination` judge skipped claim-less outputs: it scored 20 of the 25 baseline samples and 21 of the 22 fine-tuned samples. So it effectively measured real answers only, and found no difference.
- With refusals removed (intent reading), the lexical rate is 0.259 vs 0.273, consistent with the judge.
- Conclusion: there is no evidence that the fine-tuned model's answers are more faithful.

### 4e. Latency / memory — PARTIALLY VERIFIED

- These are single runs taken at different times. The same baseline measured p50 = 4.64 s in run0 and 1.75 s in run1, a 2.6× swing from machine state alone. The 1.75 s vs 2.18 s difference is **within observed noise** and should not be reported as a regression.
- Memory is one `footprint` reading of the SGLang scheduler after each run (8.69 / 8.22 GB). It includes MLX caches and is not a peak.
- The "≈ 8.5 GB peak" training footprint was observed once at step 6, not measured as a peak.

### 4f. Evaluation-set quality

- 40 items, 40 unique ids and questions. The 17 unanswerable items break down as 10 distractor, 4 off-topic and 3 no-evidence; the 23 answerable as 20 single-passage and 3 multi-passage.
- Labels were reviewed by hand; the answerable gold answers are correct. **h30 is ambiguous**: "No — economy must be booked for flights under 6 hours" is a defensible grounded answer, yet it's labelled unanswerable. **h25** is strict but defensible.
- No contamination: 0 shared 8-grams or 6-grams with train/dev, and held-out source documents are disjoint (enforced by `test_no_heldout_leakage_into_training_data`).
- h31's fact is in the training data, by design, as a memorisation probe.
- Representativeness is low: synthetic, short, clean policy text from a single author; 1–3 passages per item; no OCR noise, tables or long contexts. With n = 40, one item is 2.5 points of grounded accuracy, and a 95% Wilson interval around 0.775 is roughly 0.62–0.88.

---

## 5. Fine-tuning verification

| Claim | Evidence | Status |
|---|---|---|
| Base model Qwen/Qwen2.5-1.5B-Instruct | `adapter_config.json.base_model_name_or_path` | VERIFIED |
| LoRA r=16, α=32, dropout 0.05, 7 projection modules | `adapter_config.json` | VERIFIED |
| 18,464,768 trainable params (1.18%) | `train.log` line from PEFT | VERIFIED |
| bf16 on MPS | `train_summary.json` `model_dtype=torch.bfloat16, device=mps:0`; merged `config.json` dtype bfloat16 | VERIFIED |
| 40 optimizer steps | `trainer_state.json` global_step = max_steps = 40 | VERIFIED |
| "3 epochs" | `trainer_state.json` epoch = **2.88** (AlignTune: max_steps = ⌊109·3/8⌋) | **FAILED (minor)** — README says 3 epochs = 40 steps |
| 598 s | `train.log` `train_runtime: 598`; `train_summary.json` elapsed 606.1 s incl. setup | VERIFIED |
| Loss 0.249 → 0.052 | `trainer_state.json` log_history steps 10/20/30/40: 0.2492, 0.1732, 0.0584, 0.0524 | VERIFIED (training loss only; no dev loss was computed, so generalisation is not shown by loss) |
| Training data = curated 109 rows | `outputs/…/data/train.jsonl` == committed `sft_train.jsonl` messages; sha256 `9e45abc0…` | VERIFIED |
| Assistant-only loss | Mask check from the earlier session (29/384 tokens = JSON answer); TRL training template applied | PARTIALLY VERIFIED (not re-run in this audit) |
| Merged checkpoint = completed run | merged = bf16(base + B·A·2.0) exactly; mtimes: adapter 01:54:08 → merged 01:54:22 → served 01:56:03; served chat template identical to base | VERIFIED |
| Merged checkpoint loadable | SGLang loaded it and served 80 requests | VERIFIED |
| Data quality | 109 rows, 60 unique questions (each reused across answerable/abstain variants by design), 0 duplicate rows, 0 malformed targets, 3 rows with "the text …" artefacts; answers are long full sentences (median 19 words), which explains the fine-tuned model's verbosity | VERIFIED |
| CuratorKIT gate effectiveness | 110/110 accepted by the same-model judge; strict lexical gate removed 33 | VERIFIED; CuratorKIT's gate added no measurable filtering in this run |
| Curation reproducibility | generation temperature 0.3 means a full re-run won't reproduce the same pairs; `--reuse-curatorkit` reproduces the SFT files deterministically from the saved export | PARTIALLY VERIFIED |
| Training reproducibility | not re-run (audit constraint); MPS kernels are not guaranteed deterministic | NOT CHECKED |

---

## 6. Confirmed bugs, discrepancies and risks (ranked)

| # | Severity | Finding | Evidence |
|---|---|---|---|
| 1 | **High** | **Headline improvement misattributed.** README and the final report say fine-tuning gives "far fewer answers to unanswerable questions, fewer needless refusals". Under intent-based scoring, grounded accuracy is unchanged (0.775 vs 0.775), hallucinated answers drop by 2 items, and needless refusals *rise* by 3. The measured gain is output-contract compliance. | §4b–4d |
| 2 | **High** | **Parser misses non-JSON refusals.** `parse_model_output` fallback detects only refusal phrases, not a bare "abstain" (`abstain\ncitations: []`), so h33/h34 count as hallucinated answers even in `--lenient-abstain` mode. The README's "lenient" numbers therefore still overstate the baseline's hallucination rate. | `ps3_llm/prompting.py` fallback branch; baseline h33, h34 |
| 3 | **High** | **Contradictory flag handling.** `"abstain": "true"` (string) plus a substantive answer is treated as abstention, silently dropping an answer that may be correct (h10, h12, h18) or hallucinated (h21). Non-boolean values like `"abstain": "abstain"` are treated as *not* abstaining. Scoring rewards or penalises format, not content. | `prompting.py` JSON branch |
| 4 | Medium | **SGLangAdapter constructor raises on an unreachable server** when `model` is omitted (`/v1/models` lookup in `__init__`). That breaks the documented guarantee in `schemas.py` that inference errors return `abstained=True`. | CLI run without `--model` → `URLError` traceback |
| 5 | Medium | **Lexical correctness rule is unreliable at the margin:** a false negative on terse correct answers (h17) and a false positive on partially overlapping wrong answers (h14 "Premium economy"). It systematically favours verbose answers, which favours the fine-tuned style. | §4b |
| 6 | Medium | **Unsupported-claim rate and AuditKIT LexicalGroundedness count refusal strings as unsupported claims**, inflating the baseline's rates. | §4d |
| 7 | Medium | **Uncited answers pass through.** If every cited id is invalid, the answer is returned with `citations: []` and no abstention. Integration consumers may display unsupported text. | `prompting.py`, `adapters.py` |
| 8 | Medium | **Evidence not version-controlled.** The training log, server logs, CuratorKIT stdout and trainer state were only in gitignored `llm/outputs/` or a session-temporary directory (now copied to `llm/verification/logs/`, uncommitted). | §3 |
| 9 | Medium | **Environment not locked.** No lockfile for either venv; SGLang was built from a temporary scratch copy (`direct_url.json` → `/private/tmp/…/sglang-build`); README steps were not re-executed from scratch. | `direct_url.json` |
| 10 | Low | README says "3 epochs"; the actual run was 2.88 epochs. | `trainer_state.json` |
| 11 | Low | Latency regression and memory figures are single-run and within noise; they're reported without that caveat. `repetition_penalty=1.1` (from `generation_config`, `sampling_defaults: model`) was likely applied to both models but isn't documented. | §4e; serve logs |
| 12 | Low | h30 label ambiguous; h25 strict. | §4f |
| 13 | Low | CuratorKIT HallucinationGate (same-model judge) accepted 100%, so the "curation" quality came from the project's own lexical gate, not the toolkit. Already disclosed in the README. | manifest |
| 14 | Low | `audit.py` excludes answered items without evidence from groundedness (baseline h37), so AuditKIT n_answered (25) ≠ eval n_answered (26). | `audit.build_samples` |
| 15 | Info | AuditKIT scores precomputed outputs; it is not wired to call the SGLang candidate directly (`api:` route). Acceptable, but "AuditKIT evaluated the models" should be phrased as "AuditKIT re-scored SGLang outputs". | `audit.py` |

---

## 7. Exact reproduction steps

Environment used: Apple M4, 16 GB, macOS 26.3, Python 3.12.13, `uv`.

```bash
cd "<repo>"
# Lexsi env (verified installed versions: aligntune 0.1.28 @57e9a95, curatorkit 0.1.22, auditkit 1.1.3,
# torch 2.14.1, transformers 5.20.2, trl 1.7.1, peft 0.21.2)
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r llm/env/requirements-main.txt --override llm/env/overrides-macos.txt
# SGLang env (built from sgl-project/sglang @1e894eb with python/pyproject_other.toml, extra srt_mps)
git clone https://github.com/sgl-project/sglang.git /tmp/sglang && git -C /tmp/sglang checkout 1e894eb
rm /tmp/sglang/python/pyproject.toml && mv /tmp/sglang/python/pyproject_other.toml /tmp/sglang/python/pyproject.toml
uv venv -p 3.12 .venv-sglang
SGLANG_BUILD_RUST_EXTS=none uv pip install -p .venv-sglang/bin/python "/tmp/sglang/python[srt_mps]"

cd llm
../.venv/bin/python -m pytest -q tests                                     # expect 16 passed
# Curation (non-deterministic re-generation; or reuse the committed export):
scripts/serve_sglang.sh mlx-community/Qwen2.5-7B-Instruct-4bit & scripts/wait_for_server.sh
../.venv/bin/python -m ps3_llm.curate              # or: ../.venv/bin/python -m ps3_llm.curate --reuse-curatorkit
pkill -f sglang                                    # free memory before training
# Training (~10 min on M4) + merge:
../.venv/bin/python -m ps3_llm.train --config configs/sft_qwen2.5-1.5b.yaml
# Evaluation (one server at a time):
scripts/serve_sglang.sh Qwen/Qwen2.5-1.5B-Instruct & scripts/wait_for_server.sh
../.venv/bin/python -m ps3_llm.evaluate --adapter sglang --tag baseline-qwen2.5-1.5b
pkill -f sglang; scripts/serve_sglang.sh "$PWD/outputs/sft-qwen2.5-1.5b/merged" & scripts/wait_for_server.sh
../.venv/bin/python -m ps3_llm.evaluate --adapter sglang --tag sft-qwen2.5-1.5b
../.venv/bin/python -m ps3_llm.compare baseline-qwen2.5-1.5b sft-qwen2.5-1.5b
../.venv/bin/python -m ps3_llm.audit --baseline baseline-qwen2.5-1.5b --candidate sft-qwen2.5-1.5b
# Independent verification (no ps3_llm import):
python3 -I verification/verify_metrics.py . baseline-qwen2.5-1.5b sft-qwen2.5-1.5b
python3 -I verification/verify_intent.py  . baseline-qwen2.5-1.5b sft-qwen2.5-1.5b
```

Expected checkpoint hashes for this run are in `llm/verification/checkpoint_sha256.txt`. A re-trained adapter will
not match bit-for-bit (MPS non-determinism; not checked).

---

## 8. Integration checklist for Member 1 (Papermind retrieval)

Contract source: `llm/ps3_llm/schemas.py`. The implementation matches the documented JSON fields (verified via
`ps3_llm.cli`), with the exceptions noted.

- [ ] **chunk_id**: unique within a request, stable across re-indexing, no leading or trailing `[` / `]` (the parser
      strips brackets), no surrounding whitespace, case-sensitive exact match. Example: `doc.pdf#p3-c2`.
- [ ] **text**: plain extracted text. Keep it ≤ ~200 words per chunk and ≤ 5 chunks: training used 0–4 passages of
      ≤ 447 tokens total. The server context is 4096 tokens; an oversized prompt fails as an HTTP error, which comes
      back as `abstained: true, abstain_reason: "inference_error: …"`, not as a real abstention.
- [ ] **Empty evidence** is valid. Send `"evidence": []` rather than skipping the call, so abstentions are logged.
- [ ] **Distinguish errors from abstentions**: check `parse_ok` and `abstain_reason` (prefix `inference_error:`)
      before treating `abstained` as a model decision.
- [ ] **Uncited answers** (`abstained: false, citations: []`) can occur when the model cites unknown ids. Decide
      whether the UI shows them; recommended: treat as abstention (open fix #7).
- [ ] Always pass `--model` / `model=` explicitly until fix #4 lands, or the adapter raises on a down server.
- [ ] `page` and `score` are optional and informational; ordering is not used by the model.
- [ ] Evidence text is inserted verbatim into the prompt. Strip text that looks like `[some-id]` headers if chunks
      can contain bracketed ids, to avoid citation confusion.
- [ ] Agree on these open items before merging: chunk budget (top-k, words), uncited-answer policy, and the error
      vs abstention signal.
- [ ] Responsibilities stay split: retrieval, ingestion and the vector store remain in Member 1's module; the model
      module only consumes `GroundedQuery` JSON.

---

## 9. Minimum fixes (proposed; **not applied** — awaiting approval)

1. **Correct the claims** in `llm/README.md`, and in the final report wording if reused. Report the intent-based
   comparison (§4c) next to the strict numbers, state that the gain is contract compliance with no change in
   grounded accuracy, and fix "3 epochs" → 2.88. Caveat latency and memory as single-run.
2. **Parser:** treat a bare `abstain` (non-JSON or answer text) as abstention. Treat a non-boolean `abstain` flag
   that conflicts with a substantive answer as `parse_ok=False` / contract violation rather than silently picking
   one. Add tests for h10/h21/h33-style outputs, then re-score all saved runs with `--rescore-from` (no new
   generation) into new tags, keeping the originals.
3. **Metrics:** exclude refusal texts from claim counts. Add an intent-based "answered" field next to the
   flag-based one, plus a manual-label override file for known lexical-rule errors (h17, h14).
4. **Adapter:** move the `/v1/models` lookup out of `__init__` (lazy and error-handled) so a down server returns the
   documented error answer. Optionally convert uncited answers into abstentions behind a flag.
5. **Evidence:** commit `llm/verification/` (scripts + log copies), or decide to keep logs out of git and archive
   them elsewhere.
6. Optional, for a real improvement claim: a larger and harder held-out set, plus training targets with short
   answers and more inference-style answerable examples (the over-abstention cases h02, h14, h20).
