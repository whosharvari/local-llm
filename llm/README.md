# PS3 model module — SGLang serving, post-training and evaluation (Member 2)

Grounded question answering: given a question plus retrieved evidence passages, the model returns a JSON
answer with passage citations, or abstains when the evidence does not contain the answer.

Retrieval, PDF ingestion and the vector store belong to Member 1 (Papermind) and are **not** in this
module. This module consumes request JSON and returns answer JSON — see
[`INTEGRATION_GUIDE.md`](INTEGRATION_GUIDE.md) (proposed contract v1.1) and [`ps3_llm/schemas.py`](ps3_llm/schemas.py).
Independent audit and its follow-up: [`../VERIFICATION_REPORT.md`](../VERIFICATION_REPORT.md).

## Pipeline

| Stage | Tool | Entry point | Output |
|---|---|---|---|
| Serving / inference | **SGLang** (MLX runtime on Apple Silicon) | `scripts/serve_sglang.sh` | OpenAI-compatible `/v1` |
| 1. Data curation | **CuratorKIT** `qa` generation + `HallucinationGate`, teacher served by SGLang | `python -m ps3_llm.curate` | `data/curated/` |
| 2. Fine-tuning | **AlignTune** `create_sft_trainer`, TRL backend, LoRA, Apple MPS | `python -m ps3_llm.train` | `outputs/<run>/adapter`, `merged/` (not in Git) |
| 3. Evaluation | deterministic v2 metrics + **AuditKIT** re-scoring (judge served by SGLang) | `ps3_llm.evaluate`, `ps3_llm.compare`, `ps3_llm.audit` | `results/` |

```
ps3_llm/
  schemas.py    contract (request validation, GroundedAnswer, reason codes)
  prompting.py  system prompt, evidence rendering, SFT target format, output parser (serving policy + intent reading)
  adapters.py   MockAdapter (deterministic, no model) and SGLangAdapter (OpenAI-compatible client, never raises)
  cli.py        JSON-in / JSON-out entry point for integration
  curate.py     CuratorKIT stage + strict grounding gate + grounded SFT example builder
  train.py      AlignTune LoRA SFT + LoRA merge for SGLang
  metrics.py    v2 metrics (adjudicated correctness; serving and intent decisions)
  evaluate.py   run an adapter over the held-out set, or re-score saved raw outputs (--rescore-from)
  compare.py    baseline-vs-candidate report (both decisions, item flips, exact McNemar)
  audit.py      AuditKIT re-scoring of saved outputs (+ optional SGLang-served judge)
data/mock/      synthetic "Halden Group" policy documents (train_docs.json and heldout_docs.json are disjoint),
                held-out set heldout_eval.jsonl (original labels) + heldout_adjudication.json (correctness criteria)
data/curated/   CuratorKIT outputs (manifest, provenance, checksums) and the SFT train/dev files
configs/        training configs
env/            dependency manifests and verified macOS arm64 lock files
results/        v1 historical runs (unchanged) and results/v2/ (corrected re-scoring of the same raw outputs)
verification/   independent verifier scripts (no ps3_llm import) and preserved logs
examples/integration/  standalone JSON client + real request/response examples
```

## Environment

Tested on Apple M4, 16 GB unified memory, macOS 26.3, CPython 3.12.13, `uv`. No CUDA GPU, so SGLang runs
on its MLX backend (`SGLANG_USE_MLX=1`) and AlignTune uses the TRL backend on MPS (Unsloth requires CUDA).
SGLang and the Lexsi toolkits live in **separate** virtualenvs (torch 2.13.0 vs 2.14.1).

| Component | Version (verified installed) |
|---|---|
| sglang | built from upstream commit `1e894eb92d403062cf21e2910dba888cad7adb9e` (`0.0.0.dev1+g1e894eb92`), extra `srt_mps`, mlx 0.32.4, torch 2.13.0 |
| aligntune | 0.1.28 from git `57e9a95e79ace15014beede175c4c3d72405f3b3` (PyPI 0.1.17 predates the API used) |
| curatorkit / auditkit | 0.1.22 / 1.1.3 (PyPI) |
| torch / transformers / trl / peft | 2.14.1 / 5.20.2 / 1.7.1 / 0.21.2 |

### Clean install on Apple Silicon (verified: both rebuilt into fresh venvs, freezes identical to the working envs)

Run from the repository root.

```bash
# Main env: the lock is the complete closure -> --no-deps (re-resolving would pull vllm via trl[vllm] and fail on macOS)
uv venv -p 3.12 .venv
uv pip install --no-deps -p .venv/bin/python -r llm/env/lock-main-macos-arm64.txt

# SGLang env: pinned deps first, then sglang itself from the exact upstream commit with the Metal pyproject
uv venv -p 3.12 .venv-sglang
uv pip install --no-deps -p .venv-sglang/bin/python -r llm/env/lock-sglang-macos-arm64.txt
mkdir sglang-src && git -C sglang-src init -q && git -C sglang-src remote add origin https://github.com/sgl-project/sglang.git
git -C sglang-src fetch --depth 1 origin 1e894eb92d403062cf21e2910dba888cad7adb9e && git -C sglang-src checkout FETCH_HEAD
rm sglang-src/python/pyproject.toml && mv sglang-src/python/pyproject_other.toml sglang-src/python/pyproject.toml
SGLANG_BUILD_RUST_EXTS=none uv pip install --no-deps -p .venv-sglang/bin/python "./sglang-src/python[srt_mps]"

# verify
(cd llm && ../.venv/bin/python -m pytest -q tests)                       # expect 93 passed
.venv-sglang/bin/python -c "import sglang, mlx.core as mx; print(mx.metal.is_available())"   # True
```

Not reproducible from a lock: the locks pin versions but carry no hashes; PyPI wheels could in principle be
yanked. Other platforms (Linux/CUDA) are untested: use `llm/env/requirements-main.txt` with
`--override llm/env/overrides-macos.txt` removed as appropriate, and SGLang's standard CUDA install.

## Reproduce (all commands from `llm/`)

```bash
../.venv/bin/python -m pytest -q tests                                   # 93 tests, no model server needed

# 1. curation (~15 min on M4; regeneration is sampled at T=0.3, so pairs will differ from the committed ones)
scripts/serve_sglang.sh mlx-community/Qwen2.5-7B-Instruct-4bit & scripts/wait_for_server.sh
../.venv/bin/python -m ps3_llm.curate                    # full run, or:
../.venv/bin/python -m ps3_llm.curate --reuse-curatorkit # deterministic rebuild of the SFT files from the committed export
pkill -f sglang                                          # free memory before training

# 2. fine-tune + merge (~10 min on M4): 40 optimizer steps = 2.88 epochs (AlignTune: max_steps = floor(109*3/8))
../.venv/bin/python -m ps3_llm.train --config configs/sft_qwen2.5-1.5b.yaml

# 3. evaluate through SGLang (one server at a time; each new --tag must be a new directory)
scripts/serve_sglang.sh Qwen/Qwen2.5-1.5B-Instruct & scripts/wait_for_server.sh
../.venv/bin/python -m ps3_llm.evaluate --adapter sglang --tag v2/baseline-new
pkill -f sglang
scripts/serve_sglang.sh "$PWD/outputs/sft-qwen2.5-1.5b/merged" --served-model-name ps3-qwen2.5-1.5b-grounded-sft & scripts/wait_for_server.sh
../.venv/bin/python -m ps3_llm.evaluate --adapter sglang --tag v2/sft-new

# 4. reports
../.venv/bin/python -m ps3_llm.compare v2/baseline-new v2/sft-new
../.venv/bin/python -m ps3_llm.audit --baseline v2/baseline-new --candidate v2/sft-new --mode intent \
    [--judge api:mlx-community/Qwen2.5-7B-Instruct-4bit]   # judge needs the 7B served by SGLang

# re-score the committed historical outputs (no model) and verify independently
../.venv/bin/python -m ps3_llm.evaluate --rescore-from baseline-qwen2.5-1.5b --tag v2/my-rescore   # new dir only
python3 -I verification/verify_v2.py . baseline-qwen2.5-1.5b sft-qwen2.5-1.5b                     # expect MISMATCHES: 0
```

Model weights: the baseline downloads from the Hugging Face Hub on first serve; the fine-tuned
`outputs/sft-qwen2.5-1.5b/merged` (2.9 GB) is gitignored — regenerate with step 2 or copy it from Member 2.
SHA-256 of the recorded run's weights: `verification/checkpoint_sha256.txt` (re-training on MPS is not
guaranteed bit-identical).

## Metrics (v2)

Defined in [`ps3_llm/metrics.py`](ps3_llm/metrics.py). Deterministic; no judge model.

- **Correctness** is decided per item by adjudicated key facts in `data/mock/heldout_adjudication.json`
  (every required fact present, no rejected phrase). The original labels in `heldout_eval.jsonl` are unchanged;
  four reviewed items (h14, h17, h20, h30) carry written rationales there.
- Each item is scored under two decisions from the same raw text:
  - **serving**: what the API returns (flag respected; refusal text, flag/answer contradictions, non-JSON
    and answers with no valid citation all become abstentions);
  - **intent**: whether the answer text is substantive, ignoring the abstain flag. Used to compare models
    that differ in how well they use the flag.
- Reported separately: answer correct / wrong / inappropriate abstention (23 answerable), correct abstention /
  hallucinated answer (16 unanswerable), citation correctness (answered answerable), unsupported-claim rate
  (claims in answers; refusals are never claims), JSON / schema validity (40), grounded accuracy (40; and 39
  excluding the ambiguous h30).

The v1 metrics that produced `results/<tag>/summary.json` were superseded after the audit, which found that
they mostly measured the baseline's misuse of the abstain flag. v1 files are kept unchanged as history.

## Results (2026-10-24; v2 re-scoring of the saved SGLang outputs — no new generation)

Held-out set: 40 synthetic items over 3 documents never used in curation or training (leakage test enforced):
23 answerable (3 multi-passage), 16 unanswerable (9 distractor, 4 off-topic, 3 no-evidence), 1 ambiguous (h30).
Both models were served by SGLang with identical decoding (temperature 0, max 384 tokens, concurrency 4,
same server defaults). Source: [`results/v2/compare_baseline_vs_sft.md`](results/v2/compare_baseline_vs_sft.md);
independently recomputed by `verification/verify_v2.py` (0 mismatches).

### Curation (CuratorKIT)
38 training chunks (8 documents) → CuratorKIT `qa` (3 questions/chunk; generator and judge
`mlx-community/Qwen2.5-7B-Instruct-4bit` on SGLang; `hallucination_threshold=0.7`) → 110 generated, **110
accepted, 0 rejected** by CuratorKIT's HallucinationGate. Many accepted answers contained claims not in the
source, so a deterministic strict-grounding gate (`curate.strict_ground`) followed: 60 kept, 17 trimmed,
33 rejected → **77 QA pairs** → 109 train / 30 dev SFT examples (split by document).

### Fine-tuning (AlignTune, TRL backend, LoRA, MPS)
Qwen2.5-1.5B-Instruct; LoRA r=16, α=32, dropout 0.05 on all attention and MLP projections (18,464,768
trainable params, 1.18%); bf16; micro-batch 1 × grad-accum 8; lr 2e-4; assistant-only loss; gradient
checkpointing. **40 optimizer steps = 2.88 epochs, 598 s train runtime; training loss 0.249 (step 10) →
0.052 (step 40).** Falling training loss on 109 examples says nothing about generalisation by itself; no
dev-loss was computed. Process footprint ≈ 8.5 GB (single observation).

### Baseline vs fine-tuned

| Metric (denominator) | Baseline serving | Fine-tuned serving | Baseline intent | Fine-tuned intent |
|---|---|---|---|---|
| Grounded accuracy (40) | 0.725 | 0.775 | 0.775 | 0.775 |
| Answer correct (23) | 0.652 (15) | 0.783 (18) | 0.826 (19) | 0.783 (18) |
| Wrong answer (23) | 0.043 (1) | 0.043 (1) | 0.130 (3) | 0.043 (1) |
| Inappropriate abstention (23) | 0.304 (7) | 0.174 (4) | 0.043 (1) | 0.174 (4) |
| Correct abstention (16) | 0.812 (13) | 0.812 (13) | 0.688 (11) | 0.812 (13) |
| **Hallucinated answer (16)** | **0.188 (3)** | **0.188 (3)** | 0.312 (5) | 0.188 (3) |
| Cites ≥1 gold passage (answered answerable) | 1.000 | 0.947 | 1.000 | 0.947 |
| Unsupported-claim rate (claims in answers) | 0.200 | 0.273 | 0.259 | 0.273 |
| Item flips vs baseline (grounded) | — | 3 better, 1 worse, McNemar p = 0.625 | — | 2 better, 2 worse, p = 1.000 |

| Output contract (40) | Baseline | Fine-tuned |
|---|---|---|
| Valid JSON | 0.950 | 1.000 |
| Schema-valid JSON (3 keys, boolean flag) | 0.500 | 1.000 |
| Invented citation ids | 3 | 0 |
| Latency p50 / p95, single run (s) | 1.75 / 2.88 | 2.18 / 4.15 |
| SGLang scheduler memory, single reading | 8.7 GB | 8.2 GB |

**What the evidence supports**
- The fine-tuned model **follows the output contract reliably** (100% schema-valid; the baseline used a
  non-boolean or contradictory abstain flag on 19 of 40 items, emitted non-JSON on 2, and invented 3
  citation ids).
- Under the serving policy that Papermind will actually receive, the fine-tuned model answers more
  answerable questions (18 vs 15 correct) **with the same hallucinated-answer rate (3/16)**. The three
  serving gains (h10, h12, h18) are items where the baseline wrote a correct answer but set `abstain: "true"`.
- On model intent (flag ignored), grounded accuracy is **identical (31/40)**: the fine-tuned model gives 2
  fewer hallucinated answers (h21, h31) but 3 more needless refusals and loses h39 on citation.
- **Not supported:** that fine-tuning reduces hallucinated or unsupported claims. The lexical
  unsupported-claim rate is not lower. AuditKIT (intent mode): F1 0.639 → 0.591, LexicalGroundedness
  0.866 → 0.812, LLM-judged Hallucination 0.250 → 0.175 (judge: the same 7B model that generated the
  training data). The deterministic metrics and the judge disagree; n = 40 cannot resolve that.
- Repeat runs: fine-tuned outputs byte-identical (40/40); baseline differed on 1/40 item.
- Latency and memory are single measurements taken at different times; the same baseline measured
  p50 = 4.64 s in an earlier run, so the latency difference is within observed noise.

**Error analysis (fine-tuned, intent)**: answers 3 distractor questions with a related but non-answering
fact (h23, h25, h29); refuses answers that need a small inference (h02 band lookup, h14 "4 hours < 6 hours",
h20 exception) and h01; multi-passage questions are weak (h39 wrong citation; h40 picks the high-cost
hotel cap for a standard city, as does the baseline). Memorisation probe h31 (fact present in training,
unrelated evidence): the fine-tuned model abstains, the baseline invents "20 days".

## Limitations

- **Small synthetic evaluation.** 40 hand-labelled items from three synthetic documents; one item = 2.5
  points; a 95% Wilson interval around 0.775 is ≈ 0.63–0.88. Limited evidence of generalisation; real
  Papermind documents are untested.
- **No model selection sweep; dev split unused.** Only Qwen2.5-1.5B-Instruct was trained.
- **Teacher = judge.** CuratorKIT generation and judging and the AuditKIT judge all use the same 7B model.
- **Training answers are long** (median 19 words), so the fine-tuned model is verbose (lower token F1).
- **Apple-specific.** bf16 load, TRL gradient checkpointing and an MPS allocator cap were needed on 16 GB;
  CUDA paths (Unsloth, vLLM) untested. Locks have no hashes.
- **Toolkit issues found:** CuratorKIT's same-model gate accepted 110/110; AlignTune 0.1.28 stringifies
  list-valued conversational `prompt` columns (worked around by passing `messages`) and does not apply its
  `use_gradient_checkpointing` option (TRL flag passed instead).
- Three training answers still contain "the text says…" phrasing missed by the boilerplate filter.
- Raw logs in `verification/logs/` are verbatim and contain local absolute paths of the original machine.
