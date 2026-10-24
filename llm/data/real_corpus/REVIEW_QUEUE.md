# Human review queue — real-corpus evaluation set

**Status:** 0 of 106 items are human-reviewed. No reviews have been created on anyone's behalf. Freezing the
test split (`freeze --split test`) is blocked until every one of the 72 test items has a named human's decision.

The labels are machine drafts. These checks have been done, and none of them replaces human review:

- automated provenance checks (`validate`, `tests/test_real_corpus.py`);
- a criteria-sensitivity probe: changing a required number must make the reference answer fail;
- the drafter's own item-by-item read against the source text.

## How to review

```bash
cd llm
# interactive (shows the question, reference answer, criteria, support quotes and all 4 passages in full)
../.venv/bin/python -m ps3_llm.real_corpus review --split test --reviewer "Full Name"
# or offline: fill in review/review_test.csv (decision = accept | reject | needs_revision) while reading review/review_test.md
../.venv/bin/python -m ps3_llm.real_corpus import-reviews data/real_corpus/review/review_test.csv
../.venv/bin/python -m ps3_llm.real_corpus status
# when every test item has a decision:
../.venv/bin/python -m ps3_llm.real_corpus freeze --split test --name test-v1
```

- The sheets in `review/` carry each item's label hash.
- If a label changes after a sheet was exported, regenerate the sheet: `review-sheet --split test --out data/real_corpus/review/review_test`.
- A decision recorded against an old hash is reported as stale and does not count.

For every item, check that:

1. the reference answer is correct and complete against the paper;
2. each required fact is necessary and its alternatives are fair;
3. the GOLD passages support the answer, and no other passage answers it;
4. for abstain items, no passage answers the question.

## Items needing a specific decision

| Item | Decision needed |
|---|---|
| `rcd-resnet-02` | Criteria require "identity"; both models' dev answers ("bypasses some or all layers") omit it. Keep the requirement or relax to the skip fact alone? |
| `rcd-rag-01` | **Revised after observing dev model outputs** (the first criteria rejected the paper's own "parametric and non-parametric memory" wording). Confirm the revision is justified by the source, or reject the item. |
| `rct-gpt3-02` | Conceptual comparison; lexical criteria are loose (any answer mentioning weights and examples). Accept as is, tighten, or reject. |
| `rct-gpt3-15` | Answer read from a flattened Table 2.1 row (`175.0B 12288 3.2M 0.6 × 10−4`). Is the row unambiguous enough to count as answerable from the chunk? Criteria tightened on 2026-10-25 to require the exponent (before any model saw the test split). |
| `rct-gpt3-10` | Criteria tightened on 2026-10-25 from bare "4" to beam-width phrasings (before any model saw the test split). Confirm. |
| `rct-gpt3-25` | Answer (96 layers) is in the PDF but not in any chunk (table column lost in extraction); expected behaviour is abstention. Accept this definition of "answerable from supplied evidence"? |
| `rct-gpt3-23` | An answer stating only "93% English" is scored wrong because it does not state the share asked for (7%). Agree? |
| `rct-rlhf-04`, `rct-rlhf-06` | Lenient alternatives ("better", "equal"; "lose", "score"). Accept or tighten. |
| `rct-rlhf-15`, `rcd-rag-09` | A bare number ("16", "8") is one of the required facts; an answer with that number in another role would pass. Accept or tighten. |
| all 28 abstain-type items | Confirm that none of the 4 supplied passages answers the question. The drafter found none, but did not check independently. |
