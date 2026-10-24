---
tags:
- curatorkit
configs:
- config_name: sft_alpaca
  default: true
  description: curatorkit sft_alpaca export, sha256 238c48b23abb36be513074bafdb0694403b3e0c451765bc350cdb652b2dca556
  data_files:
  - split: train
    path: sft_alpaca.jsonl
---

# Dataset Card: ps3_grounded_qa

## Summary

- **Run timestamp**: 2026-10-24T21:12:35.521536Z
- **Wall clock time**: 2259.6s
- **Pipeline config hash**: `5fdf6bf2a5a0cbb4`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 122 | 122 | — | 0 | 122 |
| SchemaGate | 122 | 120 | 0 | 2 | 120 |
| ExactDeduplicator | 120 | 120 | — | — | 120 |
| TextCleaner | 120 | 120 | — | — | 120 |
| QAGenerationTask | 120 | 234 | — | 0 | 234 |
| HallucinationGate | 234 | 7 | 0 | 227 | 7 |
| AlpacaExporter | — | 7 | — | — | 7 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 122/0 passed (N/A), 0 rejected
- **SchemaGate**: 120/122 passed (98.4%), 2 rejected
- **QAGenerationTask**: 234/120 passed (195.0%), 0 rejected
- **HallucinationGate**: 7/234 passed (3.0%), 227 rejected


**Rejection reasons:**
  - `judge_error:RuntimeError`: 227
  - `encoding_error:null_byte_in_output`: 2

## Deduplication

  - exact_duplicates_removed: 0

## Known Limitations

- No MinHash deduplication applied.
- Table extraction requires `extract_tables: true` on the PDF reader and MinerU with GPU support. Disabled by default.
- Token counting uses a whitespace tokenizer by default. For exact LLM context window
  enforcement, enable `use_tiktoken: true` in the SchemaGate config.
- Both within-run (MinHash/Diversity) and cross-run (EmbeddingDeduplicator) deduplication
  are available. Cross-run dedup persists an embedding index across runs; use
  `embedding_reset_index: true` to start fresh.

## Reproduction

```bash
pip install curatorkit
curatorkit run pipeline.yaml --output-dir ./out/
```

## Tool Versions

- **curatorkit**: 0.1.22
- **python**: 3.12.13

## Configs

Load one with `datasets.load_dataset("<this folder or Hub repo>", "<config>")`.

| Config | Splits (rows) | Columns |
|--------|---------------|---------|
| `sft_alpaca` | train (7) | `instruction`: string<br>`input`: string (empty when there is no input)<br>`output`: string |
