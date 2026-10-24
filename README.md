# PS3 — Grounded document QA with a local LLM (Syntax Error 2026, Local LLM Track)

**Objective:** answer questions about a collection of 13 ML research papers **only from retrieved evidence**,
cite the passages used, and abstain when the evidence does not contain the answer. The goal is grounded
answers with fewer hallucinations, from a local model served by SGLang.

**Status (read first):**

- **Works end to end.** The pipeline runs locally: question → retrieval over the paper index → SGLang-served
  fine-tuned model → validated answer with citations, or a clear abstention (`llm/scripts/demo.sh`).
- **Tools.** SGLang, CuratorKIT, AlignTune and AuditKIT are each integrated and have been executed.
- **A hallucination reduction has NOT been demonstrated.** On the synthetic held-out set the fine-tuned model
  has the same hallucinated-answer rate as the baseline (3/16 vs 3/16). On the real-paper **dev** split
  (34 items, machine-drafted labels) it answers far more questions correctly, but hallucinates on 3/14
  abstain items, against 0/14 for the baseline, which abstains on almost everything.
- **The frozen real-paper test comparison is BLOCKED on human label review** (see
  [What is blocked](#what-is-blocked)).
- **Not retrained on real papers.** The real-paper fine-tune was not trained: its CuratorKIT run was broken
  by a GPU fault, and the outputs are preserved.

Full acceptance-criteria table and executed commands: [`VERIFICATION_REPORT.md` → Part C](VERIFICATION_REPORT.md).

## Repository layout

| Path | Owner | Contents |
|---|---|---|
| [`llm/`](llm/) | Member 2 | Model module: prompting, SGLang adapter, retrieval + end-to-end pipeline, CuratorKIT curation, AlignTune training, metrics, AuditKIT re-scoring, tests. Details: [`llm/README.md`](llm/README.md) |
| [`llm/INTEGRATION_GUIDE.md`](llm/INTEGRATION_GUIDE.md) | Member 2 | JSON request/response contract (v1.1) and abstain reasons |
| [`llm/data/real_corpus/`](llm/data/real_corpus/README.md) | Member 2 | Real-paper evaluation set: 106 items, split by document. **Labels are machine drafts, not human-verified** |
| [`llm/data/real_train/v1/`](llm/data/real_train/v1/README.md) | Member 2 | Leakage-free training seeds (62 grounded_qa rows, 122 chunks), refitted retrieval gate, failed CuratorKIT run (preserved) |
| [`VERIFICATION_REPORT.md`](VERIFICATION_REPORT.md) | — | Part C: final submission state. Parts A/B: earlier audit (historical) |
| `data/raw/`, `data/processed/chunks.json` | Member 1 | 13 papers (PDF) and their 2,349 text chunks |
| `data/datasets/grounded_qa/` | Member 1 | 139 labelled questions (no reference answers), 163 claim probes. Unmodified |
| `vector_store/` | Member 1 | FAISS IndexFlatIP (2,349 × 384, all-MiniLM-L6-v2), row-aligned metadata. Unmodified |

## Architecture and data flow

```
question ─► retrieval (ps3_llm.retrieval: vector_store/index.faiss, all-MiniLM-L6-v2, top-4, chunk/doc/page kept)
         ─► optional retrieval gate (refitted on train-split questions only)
         ─► prompt built from the retrieved passages only (ps3_llm.prompting; no labels on this path)
         ─► SGLang-served model (OpenAI-compatible /v1, localhost)
         ─► serving parser: JSON only, citations restricted to retrieved ids, refusal/contradiction -> abstain
         ─► number guard: answers stating numbers absent from the cited passages are withheld
         ─► {answer, citations with document + page} or a clear abstention; JSONL run log
```

| Tool | Role | Where |
|---|---|---|
| **SGLang** (MLX backend) | Serves the baseline, the fine-tuned model, the 7B teacher and the AuditKIT judge | `llm/scripts/serve_sglang.sh`, `wait_for_server.sh` |
| **CuratorKIT** | QA generation from training chunks + HallucinationGate (Stage 1 data; real-paper run failed, see above) | `llm/ps3_llm/curate.py`, `realtrain.py` |
| **AlignTune** | LoRA SFT of Qwen2.5-1.5B-Instruct (TRL backend, Apple MPS) | `llm/ps3_llm/train.py` |
| **AuditKIT** | Re-scores saved outputs: F1, LexicalGroundedness, LLM-judged Hallucination | `llm/ps3_llm/audit.py` |

Primary metrics are deterministic (`llm/ps3_llm/metrics.py` v2 plus `real_corpus report`); AuditKIT is a second view.

## Setup

- **Tested only on** Apple Silicon (M4, 16 GB, macOS 26.3, CPython 3.12) with `uv`. Linux/CUDA is untested.
- **Two virtualenvs** at the repository root:
  - `.venv`: CuratorKIT, AlignTune, AuditKIT, torch, sentence-transformers;
  - `.venv-sglang`: SGLang on MLX.
- **Install commands** (from the lock files in `llm/env/`):
  [`llm/README.md` → Clean install](llm/README.md#clean-install-on-apple-silicon-verified-both-rebuilt-into-fresh-venvs-freezes-identical-to-the-working-envs).
- **Weights.** The fine-tuned checkpoint `llm/outputs/sft-qwen2.5-1.5b/merged` (2.9 GB, gitignored) must be
  regenerated (`llm/README.md` → Reproduce) or copied from Member 2. Expected SHA-256:
  `42f885c0a302768bb9dbf76a54ec073e01e076d3abd6a7af2b7c7c6224e53764` (`llm/verification/checkpoint_sha256.txt`).
- **Downloads on first use:** the base model and the embedding model `sentence-transformers/all-MiniLM-L6-v2`
  come from the Hugging Face Hub.
- **Run one model server at a time.** On 16 GB, two 7–8B MLX servers ran out of Metal memory, and the MLX
  backend suffered GPU faults when other GPU processes ran alongside it.

## Run (all commands from `llm/`)

```bash
cd llm
../.venv/bin/python -m pytest -q tests                      # 129 passed, 1 skipped (no server needed)
python3 -I verification/verify_v2.py . baseline-qwen2.5-1.5b sft-qwen2.5-1.5b   # MISMATCHES: 0
../.venv/bin/python -m ps3_llm.real_corpus validate         # dataset schema/provenance/leakage: OK

# end-to-end demo: serves the checkpoint on 127.0.0.1:30000, asks 3 questions, logs records, stops the server
scripts/demo.sh "$PWD/outputs/sft-qwen2.5-1.5b/merged" results/demo/my_demo.jsonl

# ask one question (server already running, e.g. scripts/serve_sglang.sh <ckpt> --served-model-name ps3 &)
../.venv/bin/python -m ps3_llm.pipeline "How many layers does the BERT-base model have?" \
    --gate data/real_train/v1/retrieval_gate.json --log results/demo/ask.jsonl

# locked real-corpus evaluation of one model (serve -> evaluate -> report -> stop); new tag each time
scripts/run_real_eval.sh Qwen/Qwen2.5-1.5B-Instruct ps3-baseline real_corpus/<v>/dev/baseline \
    data/real_corpus/converted/dev_provisional.jsonl data/real_corpus/converted/dev_provisional_adjudication.json
```

- `evaluate` and `report` refuse to overwrite existing results.
- Committed runs under `llm/results/` are never modified.
- Curation, training and AuditKIT commands: [`llm/README.md`](llm/README.md).

## Results

### Synthetic held-out set (40 items; committed v2, `llm/results/v2/compare_baseline_vs_sft.md`)

Serving decision: hallucinated answers 3/16 → 3/16; answer correct 15/23 → 18/23; unsupported-claim rate
0.200 → 0.273; schema-valid output 50% → 100%; invented citation ids 3 → 0.

### Real-paper dev split (34 items, provisional labels, `llm/results/real_corpus/v0.1.0-draft/dev-provisional-labels-r2/`)

The same saved SGLang outputs are scored under three decisions. 95% Wilson intervals are in brackets.

| Serving decision | Baseline Qwen2.5-1.5B | Fine-tuned (Stage 1 SFT) |
|---|---|---|
| Answer correct (of 20) | 5 [0.11–0.47] | 15 [0.53–0.89] |
| False abstention (of 20) | 15 | 2 |
| Hallucinated answer (of 14 abstain items) | 0 [0.00–0.22] | 3 [0.08–0.48] |
| Abstention precision / recall | 14/29 / 14/14 | 11/13 / 11/14 |
| With number guard: correct / hallucinated | 5 / 0 | 11 / 2 |
| Schema-valid JSON (of 34) | 0.147 | 1.000 |
| AuditKIT F1 / LexicalGroundedness (intent) | 0.420 / 0.584 | 0.588 / 0.757 |

How to read this:

- The baseline's flawless hallucination score comes from abstaining on 29/34 items. Most of its
  abstentions are contract failures: 12 contradictory flags and 5 unparseable outputs.
- The fine-tune answers much more, and correctly more often, but it does hallucinate. Examples: "two GPUs"
  and "7.93%" on planted traps.
- The number guard removes one hallucination at the cost of 4 correct answers.
- The independent-judge (Llama-3.1-8B) AuditKIT run was cut short by a GPU fault and is **incomplete**
  (`llm/results/*serving.INCOMPLETE.md`).
- None of this is a test-set result.

## What is blocked

1. **Human review of the real-corpus labels → test freeze → final baseline-vs-candidate test comparison.**
   All 106 labels are machine drafts. No review was fabricated. Queue and commands:
   [`llm/data/real_corpus/REVIEW_QUEUE.md`](llm/data/real_corpus/REVIEW_QUEUE.md). Protocol fixed in advance:
   [`EVAL_PROTOCOL.md`](llm/data/real_corpus/EVAL_PROTOCOL.md). The test split has never been run by any model.
2. **Real-paper CuratorKIT data and AlignTune retraining.** The 2026-10-25 CuratorKIT run produced 234 pairs,
   but 227 judge calls failed after the SGLang server's Metal GPU fault. Outputs are preserved, and no model
   was trained on them ([`llm/data/real_train/v1/README.md`](llm/data/real_train/v1/README.md)).

## Limitations

- **No demonstrated hallucination reduction** (see Results). Small samples: 40 synthetic, 34 dev items.
- **Synthetic training.** The fine-tuned model was trained only on synthetic policy documents.
- **Lexical scoring.** Answer criteria are string matches with listed alternatives.
- **Retrieval gate.** It was refitted on 62 questions, only 8 of them negatives, and is opt-in.
- **Stage 1 parser limitation.** It misses some refusal phrasings under the *intent* decision.
- **Apple-Silicon only.** Latency and memory are single measurements, and the locks have no hashes.
- **Stability.** The MLX SGLang backend crashed repeatedly under long 7–8B workloads on this machine.

## Repository hygiene

- No credentials or tokens in tracked or new files (pattern scan for HF/OpenAI/GitHub/AWS tokens, bearer tokens, private keys).
- Servers bind to 127.0.0.1 only.
- **Local absolute paths** remain only in preserved evidence: verbatim logs, CuratorKIT manifests and
  provenance (including the failed run in `llm/data/real_train/v1/curatorkit/`), fine-tuned runs'
  `predictions.jsonl` and `run_info.json`, and `vector_store/grounding_calibration.json`.
- **Third-party papers.** `data/raw/` redistributes 13 papers whose licences have not been reviewed. Confirm
  redistribution terms before publishing. The chunk text contains the authors' published e-mail addresses.
