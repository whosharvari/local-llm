# PS3 — Grounded document QA with a local LLM (Syntax Error 2026, Local LLM Track)

**Objective:** answer questions about a document collection **only from retrieved evidence**, cite the
passages used, and abstain when the evidence does not contain the answer — i.e. grounded answers with fewer
hallucinations, served by a local model.

**Status (read first):** the serving, curation, fine-tuning and evaluation pipeline runs end to end on the
model side, and the committed v2 metrics re-compute exactly from the saved model outputs. **A measurable reduction in hallucinations has NOT
been demonstrated**: on the 40-item held-out set the fine-tuned model has the same hallucinated-answer rate as
the baseline (3/16) and a higher unsupported-claim rate (0.273 vs 0.200). See [Limitations](#limitations).

## Repository layout

| Path | Owner | Contents |
|---|---|---|
| [`llm/`](llm/) | Member 2 | Model module: prompting, SGLang adapter, CuratorKIT curation, AlignTune training, metrics, AuditKIT re-scoring, tests. Details: [`llm/README.md`](llm/README.md) |
| [`llm/INTEGRATION_GUIDE.md`](llm/INTEGRATION_GUIDE.md) | Member 2 | JSON request/response contract between retrieval and the model (v1.1, **proposed**) |
| [`VERIFICATION_REPORT.md`](VERIFICATION_REPORT.md) | — | Independent audit of the model module and its follow-up |
| `data/raw/` | Member 1 | 13 ML research papers (PDF) — the retrieval corpus |
| `data/processed/chunks.json` | Member 1 | 2,349 text chunks of those PDFs (`id`, `text`, `source_file`, `page_number`, `chunk_index`) |
| `data/datasets/grounded_qa/` | Member 1 | 139 labelled questions over the corpus (answerable / context-swap / out-of-corpus; no reference answers) and 163 claim probes |
| `vector_store/` | Member 1 | FAISS index (2,349 × 384-d, `all-MiniLM-L6-v2` per its calibration file), chunk metadata, retrieval-side abstention calibration |

The retrieval / ingestion **code** (Papermind) is not in this repository — only its artifacts above. The
model module has not yet been run on this corpus or connected to this index; it was trained and evaluated
on synthetic documents in `llm/data/mock/`.

## Pipeline and tool roles

```
question + retrieved passages ──► ps3_llm (prompt, SGLang adapter, parser) ──► {answer, citations, abstained}
                                          │
             SGLang  ── serves the baseline, the fine-tuned model, and the 7B teacher/judge (OpenAI-compatible /v1)
          CuratorKIT ── generates QA pairs from training chunks + HallucinationGate     (llm/ps3_llm/curate.py)
           AlignTune ── LoRA SFT of Qwen2.5-1.5B-Instruct on the curated data (TRL/MPS) (llm/ps3_llm/train.py)
            AuditKIT ── re-scores saved outputs: F1, LexicalGroundedness, LLM-judged Hallucination (llm/ps3_llm/audit.py)
```

Deterministic v2 metrics (`llm/ps3_llm/metrics.py`) are the primary evaluation; AuditKIT is a second view.

## Setup

Tested only on Apple Silicon (M4, 16 GB, macOS, CPython 3.12) with `uv`. Two virtualenvs at the repository
root: `.venv` (CuratorKIT, AlignTune, AuditKIT, torch) and `.venv-sglang` (SGLang on its MLX backend).
The exact install commands, from the lock files in `llm/env/`, are in
[`llm/README.md` → Clean install](llm/README.md#clean-install-on-apple-silicon-verified-both-rebuilt-into-fresh-venvs-freezes-identical-to-the-working-envs).
Linux/CUDA is untested.

The fine-tuned weights (`llm/outputs/sft-qwen2.5-1.5b/merged`, 2.9 GB) are gitignored: regenerate them with
the training command in `llm/README.md` or copy them from Member 2. Expected SHA-256:
`llm/verification/checkpoint_sha256.txt`.

## Run

All commands from `llm/`.

```bash
cd llm

# tests (no model server needed)
../.venv/bin/python -m pytest -q tests

# independent re-computation of the committed v2 metrics (stdlib only, no ps3_llm import)
python3 -I verification/verify_v2.py . baseline-qwen2.5-1.5b sft-qwen2.5-1.5b    # expect MISMATCHES: 0

# re-score committed raw outputs and compare them, writing only to a scratch directory
mkdir -p /tmp/ps3 && cp -R results/baseline-qwen2.5-1.5b results/sft-qwen2.5-1.5b /tmp/ps3/
../.venv/bin/python -m ps3_llm.evaluate --rescore-from baseline-qwen2.5-1.5b --tag base --results-dir /tmp/ps3
../.venv/bin/python -m ps3_llm.evaluate --rescore-from sft-qwen2.5-1.5b --tag sft --results-dir /tmp/ps3
../.venv/bin/python -m ps3_llm.compare base sft --results-dir /tmp/ps3        # report: /tmp/ps3/compare_base_vs_sft.md

# serve the fine-tuned model with SGLang and evaluate it live (one model at a time on 16 GB)
scripts/serve_sglang.sh "$PWD/outputs/sft-qwen2.5-1.5b/merged" --served-model-name ps3-qwen2.5-1.5b-grounded-sft &
scripts/wait_for_server.sh http://127.0.0.1:30000
../.venv/bin/python -m ps3_llm.evaluate --adapter sglang --tag sft-live --results-dir /tmp/ps3
echo '{"question": "...", "evidence": [{"chunk_id": "doc#c1", "text": "..."}]}' | ../.venv/bin/python -m ps3_llm.cli --adapter sglang
pkill -f sglang
```

`evaluate` refuses to write into an existing results directory, so committed runs under `llm/results/` are
never overwritten; use a new `--tag` or a scratch `--results-dir`. Curation, training and AuditKIT-with-judge
commands: [`llm/README.md` → Reproduce](llm/README.md#reproduce-all-commands-from-llm).

## Where things are

| What | Where |
|---|---|
| Curated training data (CuratorKIT export, strict-gate output, SFT train/dev) | `llm/data/curated/` |
| Synthetic train documents / 40-item held-out set + adjudication | `llm/data/mock/` |
| Training configs | `llm/configs/` |
| Checkpoints (not in Git) | `llm/outputs/<run>/adapter`, `llm/outputs/<run>/merged` |
| Metrics — current (v2) | `llm/results/v2/<tag>/summary.json`, report `llm/results/v2/compare_baseline_vs_sft.md` |
| Metrics — historical v1 (superseded, kept unchanged) | `llm/results/<tag>/` |
| AuditKIT runs | `llm/results/v2/<tag>/auditkit-intent/`, `llm/results/auditkit_*.json` |
| Verification scripts and preserved logs | `llm/verification/`, [`VERIFICATION_REPORT.md`](VERIFICATION_REPORT.md) |

## Limitations

- **No demonstrated hallucination reduction.** v2, serving policy, 40 items
  (`llm/results/v2/compare_baseline_vs_sft.md`): hallucinated answers 3/16 → 3/16; unsupported-claim rate
  0.200 → 0.273; grounded accuracy 0.725 → 0.775 (3 items better, 1 worse, McNemar p = 0.625). The gain is in
  output-contract compliance (schema-valid 50% → 100%, invented citation ids 3 → 0), not in grounding.
  AuditKIT's LLM-judged hallucination falls (0.250 → 0.175) but its judge is the same 7B model that generated
  the training data, and LexicalGroundedness falls (0.866 → 0.812).
- **Synthetic, small evaluation.** 40 hand-labelled items over 3 synthetic documents; the real paper corpus
  in `data/` has not been used for training or evaluation.
- **No end-to-end pipeline** from the PDFs through retrieval to the model in this repository; the
  integration contract is proposed and untested against Papermind.
- **CuratorKIT's HallucinationGate accepted 110/110 generated pairs**; the filtering that mattered was the
  added deterministic strict-grounding gate (`llm/data/curated/build_stats.json`).
- **Apple-Silicon only**, single latency/memory measurements, locks without hashes.

## Repository hygiene notes

- No credentials or tokens found in tracked files (pattern scan for HF/OpenAI/GitHub/AWS tokens, bearer
  tokens, private keys).
- Local absolute paths of the original machines remain only in preserved evidence (`llm/verification/logs/`,
  CuratorKIT provenance files, fine-tuned runs' `predictions.jsonl`/`run_info.json`) and in
  `vector_store/grounding_calibration.json` (`metadata.trained_on`). They are kept because they are
  provenance records.
- `data/raw/` redistributes 13 third-party papers. Their licences have not been reviewed here; the
  Transformer paper's own notice permits reproduction of its tables and figures only for journalistic or
  scholarly works. Confirm redistribution terms before publishing the repository, or replace the PDFs with
  download links. The chunk text contains the papers' published author e-mail addresses.
