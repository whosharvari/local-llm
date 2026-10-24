# Real-paper training data v1 — status: CURATION RUN FAILED (outputs preserved, not used for training)

Code: `ps3_llm/realtrain.py`. Only the 9 train-split papers in `../../real_corpus/split_manifest.json` may
contribute; dev/test papers are evaluation-only.

## Usable and verified

| File | Contents |
|---|---|
| `grounded_qa_train_clean.jsonl` | Leakage-free subset of Member 1's `data/datasets/grounded_qa/examples.jsonl`: 62 of its 104 train-split rows (31 answerable, 23 context-swap, 8 out-of-corpus). Every passage is checked against `chunks.json` by passage id, text and document. |
| `grounded_qa_removed.jsonl` | 77 excluded rows with reasons: the 42 train-split rows whose passages include dev/test documents (named passage and document per row), plus the 35 dev/test-split rows. The original file is unchanged. |
| `train_chunks.jsonl` | 122 deterministically selected chunks (seed 20261025) from the 9 train papers. They are non-adjacent and quality-filtered (no titles, reference lists or tables). 112 are for training; 10 come from `attention_is_all_you_need.pdf`, held out for validation loss. |
| `retrieval_gate.json` | Retrieval gate refitted on `grounded_qa_train_clean.jsonl` only. Cross-validated on that data: 94% of in-corpus questions kept, 50% of out-of-corpus questions abstained, with only 8 negatives. Dev check (evaluation only): 23/24 in-corpus kept, 6/6 out-of-corpus abstained. |

Command: `python -m ps3_llm.realtrain clean && python -m ps3_llm.realtrain chunks`.
Regression tests: `tests/test_realtrain.py`.

## Failed: CuratorKIT run of 2026-10-25 (`curatorkit/`, `curatorkit_run.json`)

**Settings.** Generator and judge were `mlx-community/Qwen2.5-7B-Instruct-4bit` served by SGLang (MLX).
`qa` generation, 2 questions per chunk, HallucinationGate threshold 0.7. Wall clock 2259.6 s.

**Stage counts** (`curatorkit/manifest.json`):

| Stage | In | Out | Rejected |
|---|---|---|---|
| JSONLReader | — | 122 | 0 |
| SchemaGate | 122 | 120 | 2 |
| ExactDeduplicator | 120 | 120 | 0 |
| QAGenerationTask | 120 | 234 | 0 |
| HallucinationGate | 234 | **7** | **227** |

**Rejection breakdown:**

- `judge_error:RuntimeError` × 227. The SGLang server crashed during the gate stage with a Metal GPU fault
  (`kIOGPUCommandBufferCallbackErrorInnocentVictim`, 02:26:52), and every later judge call failed.
  **These are integration failures, not quality judgements.**
- `encoding_error:null_byte_in_output` × 2.

The 234 generated pairs are preserved in `curatorkit/rejected.jsonl` (with source chunks) and
`curatorkit/sft_alpaca.jsonl` (7 pairs). They have not been re-judged, so no real-paper SFT data was built
(`realtrain build` was not run) and no model was trained on it.

**Earlier failures in the same session** (no outputs kept):

- Two attempts ran the teacher and an independent Llama-3.1-8B judge at the same time. Metal ran out of
  memory (`kIOGPUCommandBufferCallbackErrorOutOfMemory`).
- `--mem-fraction-static 0.3` gave the MLX backend a 250-token KV pool.
- Two runs were killed by GPU faults that coincided with other torch processes.

The working teacher settings were `CTX=3072 scripts/serve_sglang.sh mlx-community/Qwen2.5-7B-Instruct-4bit
--max-total-tokens 8192`, with nothing else using the GPU, but the GPU fault recurred after about 35 minutes.

## To finish (when a stable GPU session is available)

```bash
cd llm
mv data/real_train/v1/curatorkit data/real_train/v1/curatorkit_failed_20261025   # keep the failed run as evidence
CTX=3072 scripts/serve_sglang.sh mlx-community/Qwen2.5-7B-Instruct-4bit --max-total-tokens 8192 &   # nothing else on the GPU
scripts/wait_for_server.sh
../.venv/bin/python -m ps3_llm.realtrain curate --num-questions 2 --concurrency 3
pkill -f sglang
PORT=30000 scripts/serve_sglang.sh mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --max-total-tokens 8192 & scripts/wait_for_server.sh
../.venv/bin/python -m ps3_llm.realtrain judge                  # independent second HallucinationGate pass
pkill -f sglang
../.venv/bin/python -m ps3_llm.realtrain build                  # SFT train/validation + manifest + leakage check
../.venv/bin/python -m ps3_llm.train_check data --config configs/sft_real_v1_qwen2.5-1.5b.yaml
../.venv/bin/python -m ps3_llm.train --config configs/sft_real_v1_qwen2.5-1.5b.yaml
../.venv/bin/python -m ps3_llm.train_check weights --config configs/sft_real_v1_qwen2.5-1.5b.yaml
../.venv/bin/python -m ps3_llm.train_check valloss --config configs/sft_real_v1_qwen2.5-1.5b.yaml
```
