# Dataset Card: ps3_grounded_qa

## Summary

- **Run timestamp**: 2026-10-23T19:47:03.279924Z
- **Wall clock time**: 918.6s
- **Pipeline config hash**: `b395a4e228f03189`

## Source Documents

  - Unknown

## Pipeline Stages

| Step | Input | Passed | Recovered | Rejected | Forward |
|------|-------|--------|-----------|----------|---------|
| JSONLReader | 38 | 38 | — | 0 | 38 |
| SchemaGate | 38 | 38 | 0 | 0 | 38 |
| ExactDeduplicator | 38 | 38 | — | — | 38 |
| TextCleaner | 38 | 38 | — | — | 38 |
| QAGenerationTask | 38 | 110 | — | 0 | 110 |
| HallucinationGate | 110 | 110 | 0 | 0 | 110 |
| AlpacaExporter | — | 110 | — | — | 110 |

## Gate Pass Rates and Rejection Breakdown

- **JSONLReader**: 38/0 passed (N/A), 0 rejected
- **SchemaGate**: 38/38 passed (100.0%), 0 rejected
- **QAGenerationTask**: 110/38 passed (289.5%), 0 rejected
- **HallucinationGate**: 110/110 passed (100.0%), 0 rejected


**Rejection reasons:**
  - None

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
