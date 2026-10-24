# Evaluation protocol (pre-registered 2026-10-25, before the real-paper SFT model was trained)

This protocol was written before any dev or test output of the real-paper fine-tuned model existed. It fixes
how the final candidate is chosen and how it is compared with the baseline.

## Models

| Role | Model | Served as |
|---|---|---|
| baseline | `Qwen/Qwen2.5-1.5B-Instruct` (no fine-tuning) | SGLang, `--served-model-name ps3-baseline` |
| Stage 1 candidate | `outputs/sft-qwen2.5-1.5b/merged` (synthetic-policy SFT; SHA-256 in `verification/checkpoint_sha256.txt`) | SGLang |
| Stage 3 candidate | `outputs/sft-real-v1-qwen2.5-1.5b/merged` (real-paper SFT, `configs/sft_real_v1_qwen2.5-1.5b.yaml`) | SGLang |

## Locked evaluation configuration

These settings are fixed for every model (`scripts/run_real_eval.sh`):

- **Prompt:** `ps3_llm.prompting.SYSTEM_PROMPT` and `render_user_prompt`. The prompt's SHA-256 is recorded
  in each run's `run_info.json`.
- **Evidence:** exactly the 4 passages stored with each item in `data/real_corpus/converted/*` or `frozen/*`.
  No retrieval at evaluation time.
- **Decoding:**
  - temperature 0, `max_tokens` 384;
  - SGLang `sampling_defaults: model`, so each checkpoint's `generation_config` applies; it is identical
    for the base model and both fine-tunes;
  - concurrency 4, context length 4096;
  - unconstrained decoding: the serving parser enforces the schema.
- **Metrics:** `ps3_llm.metrics` v2 (serving and intent decisions) plus the `ps3_llm.real_corpus report`
  rates, with Wilson 95% intervals and the `serving_guarded` decision.
- **Comparison:** `ps3_llm.compare`. It refuses to compare runs scored on different data files.

## Model selection (dev split only)

- **Data:** the 34 dev items (RAG and ResNet papers), provisional labels.
- **Primary criterion:** serving grounded accuracy.
- **Tie-break (difference ≤ 1 item):** lower serving hallucinated-answer rate.
- **Guardrail:** the selected candidate must not have more serving hallucinated answers than the baseline.
  If neither fine-tune satisfies this, the report says so and the Stage 3 candidate is reported anyway,
  as the trained artefact, without an improvement claim.

## Final comparison (test split)

1. The test labels are human-reviewed (`REVIEW_QUEUE.md`) and frozen with
   `python -m ps3_llm.real_corpus freeze --split test --name test-v1`.
2. The baseline and the selected candidate are each run once with
   `scripts/run_real_eval.sh … frozen/test-v1.jsonl frozen/test-v1_adjudication.json`.
3. Report both models side by side: every rate with k/n and its Wilson interval, item flips, and the exact
   McNemar p from `compare`.
4. Claim a hallucination reduction only if the candidate's serving hallucinated-answer rate is lower
   **and** answer correctness has not regressed by more than one item. Even then, describe the result as
   small-sample evidence.

If anything is changed after test outputs are seen, the observed test results become development feedback.
"Anything" means the candidate, prompt, labels, thresholds or metric definitions. In that case a new
independent test set is needed before any confirmatory claim.

## Not part of the protocol

- The retrieval gate and the end-to-end pipeline (`ps3_llm.pipeline`) are evaluated qualitatively in the
  demo and by their integration tests. The gate's dev check is recorded in
  `data/real_train/v1/retrieval_gate.json`.
- AuditKIT runs are a second view, judged by `Meta-Llama-3.1-8B-Instruct-4bit`. That judge is a different
  model family from the Qwen teacher and from the student. Its runs never select a model.
