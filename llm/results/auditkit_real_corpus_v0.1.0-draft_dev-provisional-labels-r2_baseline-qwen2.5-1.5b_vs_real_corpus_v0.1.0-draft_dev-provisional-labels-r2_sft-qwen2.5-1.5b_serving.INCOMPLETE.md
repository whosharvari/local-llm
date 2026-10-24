# INCOMPLETE — do not cite as a result

`auditkit_real_corpus_v0.1.0-draft_dev-provisional-labels-r2_baseline-qwen2.5-1.5b_vs_real_corpus_v0.1.0-draft_dev-provisional-labels-r2_sft-qwen2.5-1.5b_serving.json`: AuditKIT serving-mode run with an independent judge
(`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`, a different model family from the Qwen generator and student),
2026-10-25 02:47–02:49.

The SGLang judge server crashed after 33 judge calls with a Metal GPU fault
(`kIOGPUCommandBufferCallbackErrorInnocentVictim`; log: `real_corpus/v0.1.0-draft/dev-provisional-labels-r2/serve_judge_llama.log`).
AuditKIT reports "Metric hallucination failed on 10 sample(s); they are left out of its headline", so the
candidate's `hallucination_answered` (0.379) covers 11 of 21 answered items. The baseline's 0.300 covers its
5 answered items. Coverage differs between the two models, so the judged hallucination numbers are not comparable.
F1 and LexicalGroundedness in this file are deterministic, use all items, and are valid.
