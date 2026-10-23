# PS3 model module — SGLang serving, post-training and evaluation (Member 2)

Grounded answers with fewer hallucinations: given a question plus retrieved evidence passages, the model
returns a JSON answer with passage citations, or abstains when the evidence does not contain the answer.

Retrieval, PDF ingestion and the vector store belong to Member 1 and are **not** in this module. This module
consumes `GroundedQuery` JSON and returns `GroundedAnswer` JSON; see the contract in
[`ps3_llm/schemas.py`](ps3_llm/schemas.py).

## Pipeline

| Stage | Tool | Entry point | Output |
|---|---|---|---|
| Serving / inference | **SGLang** (MLX runtime on Apple Silicon) | `scripts/serve_sglang.sh` | OpenAI-compatible `/v1` |
| 1. Data curation | **CuratorKIT** `qa` generation + `HallucinationGate`, teacher served by SGLang | `python -m ps3_llm.curate` | `data/curated/` |
| 2. Fine-tuning | **AlignTune** `create_sft_trainer`, TRL backend, LoRA, Apple MPS | `python -m ps3_llm.train` | `outputs/<run>/adapter`, `merged/` |
| 3. Evaluation | own deterministic metrics + **AuditKIT** re-scoring and `compare` | `python -m ps3_llm.evaluate`, `ps3_llm.audit`, `ps3_llm.compare` | `results/` |

```
ps3_llm/
  schemas.py    contract: EvidenceChunk, GroundedQuery, GroundedAnswer, EvalExample
  prompting.py  system prompt, evidence rendering, JSON target format, robust output parser
  adapters.py   MockAdapter (deterministic, no model) and SGLangAdapter (OpenAI-compatible client)
  cli.py        JSON-in / JSON-out entry point for integration
  curate.py     CuratorKIT stage + strict grounding gate + grounded SFT example builder
  train.py      AlignTune LoRA SFT + LoRA merge for SGLang
  metrics.py    groundedness / citation / abstention / latency metrics
  evaluate.py   runs an adapter over the held-out set, writes results/<tag>/
  audit.py      AuditKIT re-scoring of saved runs + ak.compare
  compare.py    baseline-vs-candidate report
data/mock/      synthetic "Halden Group" policy documents: train_docs.json and heldout_docs.json are disjoint
data/curated/   CuratorKIT outputs (manifest, provenance, checksums) and the SFT train/dev files
configs/        training configs
results/        per-run predictions, per-example scores, summaries, error buckets, AuditKIT runs
```

## Environment

Tested on Apple M4, 16 GB unified memory, macOS 26.3, Python 3.12, `uv`. There is no CUDA GPU, so:
SGLang runs on its MLX backend (`SGLANG_USE_MLX=1`), and AlignTune uses the TRL backend on MPS (Unsloth
requires CUDA). SGLang and the Lexsi toolkits are installed in **separate** virtualenvs because they pin
different torch versions (2.13 vs 2.14).

```bash
# SGLang (Apple Silicon / Metal). Built from source at commit 1e894eb with the srt_mps extra.
git clone https://github.com/sgl-project/sglang.git && git -C sglang checkout 1e894eb
rm sglang/python/pyproject.toml && mv sglang/python/pyproject_other.toml sglang/python/pyproject.toml
uv venv -p 3.12 .venv-sglang
SGLANG_BUILD_RUST_EXTS=none uv pip install -p .venv-sglang/bin/python "./sglang/python[srt_mps]"

# Lexsi toolkits + this module (AlignTune from source: the PyPI 0.1.17 release predates the API used here).
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python -r llm/env/requirements-main.txt --override llm/env/overrides-macos.txt
```

`llm/env/overrides-macos.txt` drops vLLM (pulled in by `trl[vllm]`, cannot run on macOS) and requires
torch>=2.11 (torchao). On Linux/CUDA, install SGLang normally (`pip install sglang`) and drop the override.

## Reproduce

All commands are run from `llm/`.

```bash
# 0. tests (no model needed)
../.venv/bin/python -m pytest -q tests

# 1. curation: serve the teacher with SGLang, then run CuratorKIT (~15 min on the M4)
scripts/serve_sglang.sh mlx-community/Qwen2.5-7B-Instruct-4bit &   # wait: scripts/wait_for_server.sh
../.venv/bin/python -m ps3_llm.curate                                  # full run
../.venv/bin/python -m ps3_llm.curate --reuse-curatorkit               # rebuild SFT data from saved CuratorKIT export
# stop the teacher before training (only one model fits alongside training on 16 GB)

# 2. fine-tune (AlignTune TRL LoRA on MPS) and merge the adapter for SGLang
../.venv/bin/python -m ps3_llm.train --config configs/sft_qwen2.5-1.5b.yaml

# 3. evaluate baseline and fine-tuned model through SGLang, same held-out set and decoding
scripts/serve_sglang.sh Qwen/Qwen2.5-1.5B-Instruct &
../.venv/bin/python -m ps3_llm.evaluate --adapter sglang --tag baseline-qwen2.5-1.5b
# stop it, then serve the merged fine-tuned checkpoint
scripts/serve_sglang.sh outputs/sft-qwen2.5-1.5b/merged &
../.venv/bin/python -m ps3_llm.evaluate --adapter sglang --tag sft-qwen2.5-1.5b

# 4. reports
../.venv/bin/python -m ps3_llm.compare baseline-qwen2.5-1.5b sft-qwen2.5-1.5b
../.venv/bin/python -m ps3_llm.audit --baseline baseline-qwen2.5-1.5b --candidate sft-qwen2.5-1.5b

# integration smoke test (mock adapter, no server)
../.venv/bin/python -m ps3_llm.cli --adapter mock < data/mock/example_request.json
```

## Metrics

Defined in `ps3_llm/metrics.py`; all deterministic, so baseline-vs-fine-tuned differences are not confounded
by a judge model.

- **Grounded accuracy (headline):** answerable item → correct answer *and* at least one gold citation;
  unanswerable item → abstained.
- **Hallucinated answer rate:** share of unanswerable items (distractor evidence, off-topic evidence, or no
  evidence) that were answered anyway.
- **Unsupported claim rate:** answer split into sentences/clauses; a claim is supported only if all its
  numbers and ≥60% of its content words occur in the passages it cites.
- Also: answer accuracy and token F1, over-abstention, citation precision/recall, invented numbers,
  invalid citation ids, JSON validity, p50/p95 latency, throughput, SGLang memory footprint.

Held-out set: 40 hand-labelled items over 3 documents that are never used for curation or training
(`tests/test_core.py::test_no_heldout_leakage_into_training_data` enforces this). 23 answerable (incl. 3
multi-passage), 17 unanswerable (10 distractor, 4 off-topic, 3 no-evidence).

## Results (measured 2026-10-24, Apple M4 16 GB)

All numbers are from runs in `results/`; nothing below is estimated.

### Curation (CuratorKIT)
38 training chunks (8 documents) → CuratorKIT `qa` (3 questions/chunk, teacher and judge
`mlx-community/Qwen2.5-7B-Instruct-4bit` on SGLang, `hallucination_threshold=0.7`) → 110 generated,
**110 accepted, 0 rejected** by CuratorKIT's HallucinationGate. Inspection showed many accepted answers
padded with claims not in the source (CuratorKIT's docs warn about same-model judge leniency), so a
deterministic strict-grounding gate (`curate.strict_ground`) was added: 60 kept unchanged, 17 trimmed,
33 rejected → **77 QA pairs** → 109 train / 30 dev SFT examples (split by document).

### Fine-tuning (AlignTune, TRL backend, LoRA, MPS)
Qwen2.5-1.5B-Instruct, LoRA r=16/α=32 on all attention+MLP projections (18.5M trainable params, 1.18%),
bf16, 3 epochs = 40 optimizer steps (micro-batch 1 × grad-accum 8), assistant-only loss, gradient
checkpointing. **Completed: 598 s, train loss 0.249 → 0.052** (`results/training_sft-qwen2.5-1.5b.json`).
Peak process footprint ≈ 8.5 GB.

### Baseline vs fine-tuned — same 40 held-out items, SGLang, temperature 0, max 384 tokens

| Metric | Baseline | Fine-tuned |
|---|---|---|
| **Grounded accuracy** (all 40) | 0.575 | **0.775** |
| Answer accuracy (23 answerable) | 0.652 | **0.783** |
| Appropriate abstention (17 unanswerable) | 0.471 | **0.824** |
| **Hallucinated answer rate** (unanswerable) | 0.529 | **0.176** |
| Over-abstention (answerable) | 0.261 | **0.174** |
| Unsupported claim rate (lexical, answered) | 0.407 | **0.273** |
| Answers without citations / invalid citation ids / invented numbers | 5 / 3 / 1 | 0 / 0 / 0 |
| Citation precision / recall | **1.000 / 0.971** | 0.947 / 0.895 |
| Answer token F1 (answerable) | **0.530** | 0.456 |
| Valid JSON | 0.95 | **1.00** |
| Latency p50 / p95 (s, concurrency 4) | **1.75 / 2.88** | 2.18 / 4.15 |
| SGLang scheduler memory | 8.7 GB | 8.2 GB |

Full report with per-example flips and error buckets: `results/compare_baseline-qwen2.5-1.5b_vs_sft-qwen2.5-1.5b.md`.

**Robustness checks**
- *Run-to-run noise:* the fine-tuned model produced byte-identical outputs on a repeat run (40/40); the
  baseline differed on 1/40 item between two runs (0.550 vs 0.575 grounded accuracy).
- *Parser sensitivity:* the baseline sometimes wrote `"answer": "abstain"` instead of setting the flag, which
  the strict parser counts as answering. Re-scoring both runs' saved outputs with a lenient parser
  (`--rescore-from … --lenient-abstain`) narrows but keeps the gap: grounded accuracy 0.650 → 0.775,
  hallucinated answer rate 0.353 → 0.176, unsupported claim rate 0.333 → 0.273.
- *AuditKIT* (`results/auditkit_*.json`): F1Score over all 40 (refusals scored) 0.482 → 0.591 (6 items
  wrong→right, 3 right→wrong); LexicalGroundedness of answered items 0.748 → 0.812. AuditKIT's LLM-judged
  `Hallucination` (judge: Qwen2.5-7B-4bit on SGLang) on answered items: **0.175 → 0.175 — no change.**

**Interpretation.** The measured gain is mainly in *deciding when to answer*: far fewer answers to
unanswerable questions, fewer needless refusals, always-valid JSON with citations. Whether the answers it
does give are more faithful is mixed: the lexical metric improves, the LLM judge sees no difference.
n=40 is small; no statistical significance is claimed.

**Error analysis (fine-tuned)**
- Still answers 3 distractor questions with a nearby, true-but-irrelevant fact (h23 panel chair → "must
  include a Finance member"; h29; h25 conflates "DPO assesses within 24 h" with "notify regulator within 24 h").
- Over-abstains when the answer needs a small inference (h02 EUR 10,000 falls in the 2,500–25,000 band; h14
  4-hour flight < 6 hours; h20 "not reimbursable unless…") — same failures as the baseline — plus a new
  regression on h01.
- Multi-passage questions are weak (answerable_multi 1/3): h40 picks the high-cost hotel cap for a standard
  city (both models); h39 answers correctly but cites the wrong passage.
- Token F1 drops because answers became full sentences while gold answers are short; answer accuracy rises.
- Memorisation probe h31 (annual-leave fact that *is* in training, asked with unrelated evidence): the
  fine-tuned model correctly abstains; the baseline invented "20 days".
- Latency rose ~25% p50 at similar output length; the cause was not investigated.

## Limitations

- **Mock data.** All documents are synthetic ("Halden Group"); real organisation documents arrive via
  Member 1's retrieval. The held-out set is 40 hand-labelled items — enough to show the direction, not to
  rank close variants.
- **No model selection sweep.** Only Qwen2.5-1.5B-Instruct was trained (largest size that trains comfortably
  on 16 GB MPS); the dev split (`data/curated/sft_dev.jsonl`) was built but not used for hyperparameter search.
- **Teacher = judge.** CuratorKIT generation and judging, and the AuditKIT hallucination judge, all use the
  same 7B model, which may share blind spots.
- **Apple-specific tweaks.** bf16 load, gradient checkpointing passed through to TRL, and the MPS allocator
  cap were needed on 16 GB; on a CUDA GPU the Unsloth backend and larger batches would apply.
- **Known toolkit issues found:** CuratorKIT's same-model judge accepted 110/110; AlignTune 0.1.28 stringifies
  list-valued conversational `prompt` columns (worked around by passing `messages`), and its
  `use_gradient_checkpointing` model option was not applied by the loader (passed TRL's flag instead).
- Three accepted training answers still contain "the text says…" phrasing that the boilerplate filter missed.
