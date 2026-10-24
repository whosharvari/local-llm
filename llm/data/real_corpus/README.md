# Real-corpus evaluation set (Stage 2) — `ps3-real-corpus-eval` v0.1.0-draft

Grounded-QA evaluation items built from the 13 ML papers in `../../../data/` (Member 1's corpus), for
measuring answer correctness, hallucinated answers, abstention and citation correctness of the PS3 model
on real documents. The synthetic benchmark in `../mock/` is separate and unchanged.

> **Status: 0 of 106 labels are human-verified.** Every reference answer, answer criterion and
> abstention label was drafted by a model (Claude, via Claude Code, 2026-10-24), and every item has
> annotation status `machine_draft`. These labels are **not ground truth**. Any metric computed on them
> is provisional, and `ps3_llm.real_corpus report` labels it that way. Stage 2 is complete only when a
> named human has reviewed every test item and the test split has been frozen (see [Review](#human-review-workflow)).

## Contents

| File | What |
|---|---|
| `drafts.jsonl` | Label source, edited by hand: question, category, reference answer, required facts, verbatim support quotes, rationale |
| `items.jsonl` | Built by `build`: drafts plus exact evidence text, page numbers, gold citations and a label SHA-256. Regenerated deterministically; tests fail if it is stale |
| `schema.json` | JSON Schema of an item and of a review record |
| `split_manifest.json` | Document → split assignment, seed, dataset version, SHA-256 of `chunks.json` and of Member 1's question file |
| `reviews.jsonl` | Append-only human review decisions (empty: no review has happened yet) |
| `converted/{dev,test}_provisional{.jsonl,_adjudication.json}` | The items in the existing `ps3_llm.evaluate` format (all non-rejected items, including machine drafts) |
| `frozen/` | Created by `freeze`: the human-verified test set used for final claims (does not exist yet) |

Code: [`ps3_llm/real_corpus.py`](../../ps3_llm/real_corpus.py). Tests: [`tests/test_real_corpus.py`](../../tests/test_real_corpus.py).

## Splits (by document)

| Split | Documents | Use |
|---|---|---|
| **test** | `gpt3_few_shot_learners.pdf`, `rlhf_human_preferences.pdf` | Final evaluation only. No training, prompt tuning or model selection. Look at model outputs only after the labels are frozen |
| **dev** | `rag_knowledge_intensive.pdf`, `resnet_deep_residual_learning.pdf` | Model selection and threshold tuning. No training examples |
| **train** | the other 9 papers | May contribute training examples in Stage 3, subject to the leakage check |

- The assignment is **inherited from Member 1's `grounded_qa` split** (the `source_document` of its test
  and dev answerable items). It is not drawn at random, so Member 1's train-split questions never concern
  an evaluation document. A test checks the two splits still agree.
- **Seed 20261024** controls only the order of evidence passages within an item. Distractor passages come
  from a deterministic lexical ranking.
- Every item's evidence and support passages come only from its own split's documents. The validator
  enforces this per item.

## Composition

| Split | answerable | numerical | unanswerable | context_swap | out_of_corpus | total (answer / abstain) |
|---|---|---|---|---|---|---|
| test | 15 | 30 | 10 | 10 | 7 | **72** (45 / 27) |
| dev | 8 | 12 | 4 | 4 | 6 | **34** (20 / 14) |

- **answerable / numerical**: the supplied evidence states the answer. In numerical items the key fact is
  a number. Two numerical items need two passages (`multi_passage`).
- **unanswerable**: evidence from the same paper, ranked by similarity to the question, but the paper (or
  the extracted text) never states the answer. Several are traps on purpose: the evidence holds a
  closely related number, e.g. CIFAR's "two GPUs" for an ImageNet question.
- **context_swap**: an answerable question from one paper paired with passages from the other paper of
  the same split.
- **out_of_corpus**: a topic absent from all 13 papers. The key term is checked absent across all 2,349 chunks.
- "Answerable" means **answerable from the supplied evidence**, not from the PDF. Example: `rct-gpt3-25`
  asks for GPT-3's layer count. The PDF's Table 2.1 states it, but text extraction dropped that column, so
  the expected behaviour is to abstain.
- **Question origin**: 22 questions are Member 1's `grounded_qa` test/dev questions, reused verbatim with
  newly drafted answers (`question_origin = grounded_qa:<id>`). 70 were drafted from the paper text.
  14 are context swaps of answerable items.
- Each item gives the model exactly 4 evidence passages.

## Item schema (summary — full definition in `schema.json`)

`id` (stable, `rct-*` test / `rcd-*` dev) · `split` · `category` · `answerable` · `expected_behavior`
(answer | abstain) · `question` · `reference_answer` (null when abstaining) · `required_facts` (answer
criteria: each inner list is one fact, met if any alternative occurs; this is the acceptable-variation
mechanism) · `reject` · `acceptable_variations` (examples for reviewers) · `support` (chunk id, document,
page, **verbatim quote**) · `gold_citations` · `evidence` (exact chunk text, document, page) ·
`source_document` · `swap_of` · `absent_terms` · `distractor_answers` · `rationale` · `provenance`
(question origin, `label_origin = machine_draft`, drafter, date, corpus hash) · `label_sha256`.

## How labels were produced, and what is checked automatically

The drafter read the full extracted text of the four dev/test papers (`data/processed/chunks.json`),
wrote questions, reference answers and answer criteria, and copied a supporting quote from the chunk
text. `build` refuses to write `items.jsonl` unless every item passes these checks, and the tests re-run them:

- Each support quote is a verbatim (whitespace-normalised) passage of the cited chunk, and its document
  and page match `chunks.json`.
- Each required fact is stated in the support quotes, and the reference answer satisfies its own criteria.
- Evidence text is byte-identical to `chunks.json`, whose SHA-256 is pinned in the manifest.
- No distractor passage of an answerable item satisfies the answer criteria. Overlapping neighbour chunks
  of a gold passage are excluded.
- Context-swap evidence contains no alternative of any fact of the original answer and does not come
  from the original paper.
- `absent_terms` do not occur in the evidence. For out-of-corpus items they do not occur anywhere in the
  corpus.
- Every item's evidence is a valid request under integration contract v1.1, and its label hash matches
  its content.

These checks show that labels are **traceable to the source text**. They do not show that a label is
**correct**: a wrong reference answer can still quote a real sentence. Deciding correctness is the
reviewer's job.

## Human review workflow

```bash
cd llm
../.venv/bin/python -m ps3_llm.real_corpus status
# interactive: shows the question, reference answer, criteria, support quotes and all 4 passages in full
../.venv/bin/python -m ps3_llm.real_corpus review --split test --reviewer "Full Name"
# or offline: a Markdown sheet plus a CSV to fill in (decision = accept | reject | needs_revision)
../.venv/bin/python -m ps3_llm.real_corpus review-sheet --split test --out /tmp/review_test
../.venv/bin/python -m ps3_llm.real_corpus import-reviews /tmp/review_test.csv
# once every test item has a decision: freeze the verified items (refuses while any item is unreviewed)
../.venv/bin/python -m ps3_llm.real_corpus freeze --split test --name test-v1
```

Rules enforced in code:

- A review needs a human name; reviewer strings that look like a model or tool are refused.
- Reject and needs-revision decisions require notes.
- A review is bound to the label SHA-256 the reviewer saw. Editing a label afterwards returns it to
  `machine_draft`, and the old review is reported as stale.
- `freeze` writes `frozen/<name>.jsonl`, an adjudication file and a manifest with file hashes and
  reviewers. It never overwrites.

What the reviewer checks, per item:

- the reference answer is correct and complete against the paper;
- each required fact is necessary and its alternatives are fair;
- the GOLD passages support the answer, and no other passage answers it;
- for abstain items, no passage answers the question.

Two reviewers per item would be better. The tooling records one decision per review line, so a second
reviewer can append their own.

### Items that need a reviewer decision

- `rcd-resnet-02`: the criteria require "identity". Both models' dev answers omit it and are scored wrong.
- `rct-gpt3-02`: the criteria for this conceptual comparison are loose.
- `rct-gpt3-15`: the answer is read from a flattened table row.
- `rct-gpt3-25`: this item depends on the PDF-vs-chunk difference described above.

### Pre-screening and label revisions

- The drafter pre-screened all 28 abstain-type items and found no passage that answers its question.
  That check is not independent.
- One dev item (`rcd-rag-01`) was revised after a dev model run: its criteria rejected the paper's own
  wording.
- Two test items were tightened on 2026-10-25, after a criteria-sensitivity probe and a second read and
  **before any model produced output on the test split**:
  - `rct-gpt3-15` now requires the learning-rate exponent;
  - `rct-gpt3-10` now requires a beam-width phrase instead of a bare "4".
- The probe is now a test (`test_changing_a_required_number_makes_the_reference_answer_fail`).
- No test item has been exposed to any model output.
- Full review queue: [`REVIEW_QUEUE.md`](REVIEW_QUEUE.md). Pre-registered protocol:
  [`EVAL_PROTOCOL.md`](EVAL_PROTOCOL.md). Generated review sheets: `review/`.

## Leakage

`python -m ps3_llm.real_corpus leakage [--train file ...]` (also part of `validate` and the tests) fails if:

- a document is in more than one split, or the split does not cover the corpus;
- an item uses a passage from outside its split;
- any training artefact contains a dev/test document name, chunk id, evaluation question, or any 13-gram
  of dev/test document text. 13-grams shared with train-split documents (e.g. common reference-list
  entries) are ignored.

By default the scan covers `llm/data/curated/**` and `llm/data/mock/train_*`. Current result: **0
errors, 12 artefacts clean.** The current model was trained only on synthetic documents.

**Hazard for Stage 3:** in Member 1's `data/datasets/grounded_qa/examples.jsonl`, **42 of the 104
train-split rows contain passages from dev/test documents** (mostly context-swap contexts drawn from the
GPT-3 and RLHF papers). `leakage` reports this as a warning. Remove those passages or rows before using
that file for training, then run `leakage --train <new file>`.

## Evaluation

```bash
cd llm
# 1. convert (regenerates converted/; 'verified' writes nothing until labels are reviewed)
../.venv/bin/python -m ps3_llm.real_corpus convert --split dev --labels provisional
# 2. run a model (one SGLang server at a time); results go to a NEW directory, never over committed runs
scripts/serve_sglang.sh Qwen/Qwen2.5-1.5B-Instruct & scripts/wait_for_server.sh
../.venv/bin/python -m ps3_llm.evaluate --adapter sglang \
    --data data/real_corpus/converted/dev_provisional.jsonl \
    --adjudication data/real_corpus/converted/dev_provisional_adjudication.json \
    --tag real_corpus/<dataset-version>/dev-provisional/baseline-qwen2.5-1.5b
pkill -f sglang
scripts/serve_sglang.sh "$PWD/outputs/sft-qwen2.5-1.5b/merged" --served-model-name ps3-qwen2.5-1.5b-grounded-sft &
scripts/wait_for_server.sh
../.venv/bin/python -m ps3_llm.evaluate --adapter sglang --data ... --adjudication ... --tag real_corpus/<v>/dev-provisional/sft-qwen2.5-1.5b
# 3. real-corpus metrics (abstention precision/recall, per category, verified vs draft subsets) and comparison
../.venv/bin/python -m ps3_llm.real_corpus report results/real_corpus/<v>/dev-provisional/<model>
../.venv/bin/python -m ps3_llm.compare real_corpus/<v>/dev-provisional/baseline-qwen2.5-1.5b real_corpus/<v>/dev-provisional/sft-qwen2.5-1.5b
# final claims: the same commands with --data/--adjudication from frozen/test-v1*, for BOTH models
```

Metrics (v2 definitions in `ps3_llm/metrics.py`, under both the serving and the intent decision):

- answer correctness (all required facts present, no reject phrase);
- hallucinated-answer rate on abstain items;
- unsupported-claim rate;
- citation correctness: cites ≥1 gold passage, all gold passages, citation precision;
- added in `report`: abstention precision (abstentions on abstain items / all abstentions), recall
  (abstain items abstained on / all abstain items) and F1.

`compare` refuses to compare runs scored on different data files.

### Provisional dev runs (2026-10-24, machine-draft labels — not results)

Both models were served by SGLang with the same settings as the synthetic benchmark (temperature 0, max
384 tokens). Outputs are in `llm/results/real_corpus/v0.1.0-draft/`:

- `dev-provisional/` holds the generation runs, scored with the first draft labels.
- `dev-provisional-labels-r2/` re-scores the same saved outputs after the `rcd-rag-01` revision.

n = 34 (20 answer, 14 abstain); one item is about 3 points. Figures below are from r2.

| Metric | Baseline serving | Fine-tuned serving | Baseline intent | Fine-tuned intent |
|---|---|---|---|---|
| Answer correct (20) | 0.250 | 0.750 | 0.600 | 0.800 |
| Hallucinated answer (14) | 0.000 | 0.214 | 0.429 | 0.214 |
| Abstention precision / recall | 0.48 / 1.00 | 0.85 / 0.79 | 0.53 / 0.57 | 1.00 / 0.79 |
| Unsupported-claim rate | 0.500 (6 claims) | 0.455 | 0.794 | 0.500 |
| Schema-valid JSON (34) | 0.147 | 1.000 | | |

How to read this:

- The baseline's serving numbers mostly reflect its misuse of the abstain flag (29/34 abstentions).
- Its intent hallucination rate is inflated: 2 of its 6 intent "hallucinations" (`rcd-swap-01`,
  `rcd-ooc-04`) are refusals the parser does not recognise, "The passage does not provide specific
  information…" and a bare `abstain citations []` (see Limitations).
- The fine-tuned model fell for the planted traps on `rcd-resnet-10` ("two GPUs") and `rcd-resnet-11`
  ("7.93%"), and gave unsupported answers on `rcd-swap-02`.
- `real_corpus_metrics_v2.json` in each r2 run directory is the extended report. It covers Wilson intervals,
  the serving outcome per reason, and the number-guard decision. With the guard, the fine-tuned model has
  11/20 correct and 2/14 hallucinated, against 15/20 and 3/14 without it.
- AuditKIT, intent mode, deterministic metrics: F1 0.420 → 0.588, LexicalGroundedness 0.584 → 0.757
  (`../../results/auditkit_real_corpus_*_intent.json`).
- The independent-judge (Llama-3.1-8B) serving-mode run is incomplete because of a GPU fault: see
  `../../results/*serving.INCOMPLETE.md`.
- **No improvement is claimed.** The labels are unverified, n = 34, and this is dev.
- **The test split has deliberately not been run** for either model. It should be run only after
  `freeze`, for both models, on the same frozen file.

## Limitations

- **No human verification yet.** All 106 labels were drafted by a single model drafter, which is also
  the author of the checks.
- **Lexical scoring.** Answer criteria are word-boundary string matches. Paraphrases outside the listed
  alternatives score as wrong; criteria for conceptual questions are looser than for numerical ones.
- **Parser limitation (Stage 1 code, unchanged here).** `prompting.is_refusal_text` misses some refusal
  phrasings, so under the *intent* decision some refusals count as answers. Changing it would alter the
  committed synthetic v2 results, so it is left for a separate change; serving numbers are not affected.
- **Small and narrow.**
  - The test split has 72 items from 2 papers; a 95% Wilson interval around 0.75 is about ±0.10.
  - Context swaps use only the other document of the same split.
  - Distractors come from a lexical ranking, not from Papermind's retriever, which is not in this
    repository; real retrieval may produce easier or harder evidence.
- **Text extraction artefacts** (lost table columns, merged footnotes, lost hyphens) are part of the
  evidence on purpose, because the model sees the same chunks. Items affected by this are documented in
  their `rationale`.
- Third-party papers: the licence caveats in the root README apply to the quoted text too.
