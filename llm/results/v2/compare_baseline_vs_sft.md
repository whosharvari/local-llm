# baseline-qwen2.5-1.5b vs sft-qwen2.5-1.5b (metrics v2)

n = 40 (23 answerable, 16 unanswerable, 1 ambiguous). One item = 2.5 points of grounded accuracy. Small-sample evidence only; no statistical significance is claimed.

**serving** = what the API returns (flag respected, contradictions withheld, citation required). **intent** = whether the answer text is substantive, flag ignored.

| Metric | baseline-qwen2.5-1.5b serving | sft-qwen2.5-1.5b serving | baseline-qwen2.5-1.5b intent | sft-qwen2.5-1.5b intent | Better (intent) |
|---|---|---|---|---|---|
| Grounded accuracy (all items) | 0.725 | 0.775 | 0.775 | 0.775 | = |
| Grounded accuracy excl. ambiguous h30 | 0.718 | 0.769 | 0.769 | 0.769 | = |
| Answer correct (answerable) | 0.652 | 0.783 | 0.826 | 0.783 | baseline-qwen2.5-1.5b |
| Wrong answer (answerable) | 0.043 | 0.043 | 0.130 | 0.043 | sft-qwen2.5-1.5b |
| Inappropriate abstention (answerable) | 0.304 | 0.174 | 0.043 | 0.174 | baseline-qwen2.5-1.5b |
| Correct abstention (unanswerable) | 0.812 | 0.812 | 0.688 | 0.812 | sft-qwen2.5-1.5b |
| Hallucinated answer (unanswerable) | 0.188 | 0.188 | 0.312 | 0.188 | sft-qwen2.5-1.5b |
| Cites >=1 gold passage (answered answerable) | 1.000 | 0.947 | 1.000 | 0.947 | baseline-qwen2.5-1.5b |
| Cites all gold passages (answered answerable) | 0.938 | 0.842 | 0.955 | 0.842 | baseline-qwen2.5-1.5b |
| Citation precision (answered answerable) | 1.000 | 0.947 | 1.000 | 0.947 | baseline-qwen2.5-1.5b |
| Unsupported claim rate (claims in answers) | 0.200 | 0.273 | 0.259 | 0.273 | baseline-qwen2.5-1.5b |
| Answers with no valid citation (count) | 0 | 0 | 1 | 0 | sft-qwen2.5-1.5b |
| Answer token F1 vs gold (answerable) | 0.499 | 0.456 | 0.628 | 0.456 | baseline-qwen2.5-1.5b |

| Output / runtime | baseline-qwen2.5-1.5b | sft-qwen2.5-1.5b | Better |
|---|---|---|---|
| Valid JSON | 0.950 | 1.000 | sft-qwen2.5-1.5b |
| Schema-valid JSON (3 keys, boolean flag) | 0.500 | 1.000 | sft-qwen2.5-1.5b |
| Invented citation ids (count) | 3 | 0 | sft-qwen2.5-1.5b |
| Inference errors (count) | 0 | 0 | = |
| Latency p50 (s), single run | 1.754 | 2.183 | baseline-qwen2.5-1.5b |
| Latency p95 (s), single run | 2.882 | 4.148 | baseline-qwen2.5-1.5b |
| SGLang scheduler memory (GB), single reading | 8.690 | 8.220 | sft-qwen2.5-1.5b |
| Contract violations | {'non_boolean_flag': 18, 'flag_true_but_substantive_answer': 7, 'invalid_citation_ids': 3, 'flag_false_but_refusal_or_empty': 1, 'not_json': 2, 'missing_keys': 1} | {} | |

## Item flips — intent

2 improved, 2 regressed, 7 wrong in both; exact McNemar p = 1.000

| id | category | question | baseline-qwen2.5-1.5b | sft-qwen2.5-1.5b |
|---|---|---|---|---|
| h21 (improved) | distractor | What is the approval limit for corporate credit card purchases? | ANSWER: {"abstain": "true", "answer": "The approval limit for corporate credit card purchases is EUR 2,500.", "citatio | ABSTAIN: {"abstain": true, "answer": "", "citations": []} |
| h31 (improved) | off_topic | How many days of annual leave do new employees receive? | ANSWER: {"abstain": false, "answer": "New employees receive 20 days of annual leave.", "citations": ["halden_leave_pol | ABSTAIN: {"abstain": true, "answer": "", "citations": []} |
| h01 (regressed) | answerable | Who can approve a purchase of EUR 1,800? | ANSWER: {"abstain": "abstain", "answer": "The purchase of EUR 1,800 can be approved by the requesting employee's line  | ABSTAIN: {"abstain": true, "answer": "", "citations": []} |
| h39 (regressed) | answerable_multi | For a P1 incident involving personal data, who must be notified and within what times? | ANSWER: {"abstain": "false", "answer": "For a P1 incident involving personal data, the Chief Information Security Offi | ANSWER: {"abstain": false, "answer": "For a P1 incident involving personal data, the Chief Information Security Office |
| h02 (wrong in both) | answerable | How many written quotes are needed for a EUR 10,000 purchase? | ABSTAIN: {"abstain": "true", "answer": "abstain", "citations": ["abstain"]} | ABSTAIN: {"abstain": true, "answer": "", "citations": []} |
| h14 (wrong in both) | answerable | What class can I book for a 4-hour flight? | ANSWER: {"abstain": "true", "answer": "Premium economy", "citations": ["travel#c1"]} | ABSTAIN: {"abstain": true, "answer": "", "citations": []} |
| h20 (wrong in both) | answerable | Is alcohol reimbursable on a business trip? | ANSWER: {"abstain": "true", "answer": "No", "citations": ["travel#c3"]} | ABSTAIN: {"abstain": true, "answer": "", "citations": []} |
| h23 (wrong in both) | distractor | Who chairs the tender evaluation panel? | ANSWER: {"abstain": false, "answer": "The evaluation panel must include at least one member from Finance.", "citations | ANSWER: {"abstain": false, "answer": "The evaluation panel must include at least one member from Finance.", "citations |
| h25 (wrong in both) | distractor | Within how many hours must the regulator be notified of a personal data breach? | ANSWER: {"abstain": false, "answer": "24 hours", "citations": ["[incident#c1]"]} | ANSWER: {"abstain": false, "answer": "The regulator must be notified within 24 hours of a personal data breach.", "cit |
| h29 (wrong in both) | distractor | Which cities are on the high-cost city list? | ANSWER: {"abstain": false, "answer": "The list of high-cost cities is maintained by the Finance team and reviewed annu | ANSWER: {"abstain": false, "answer": "The high-cost cities are maintained by the Finance team and reviewed annually.", |
| h40 (wrong in both) | answerable_multi | On an international trip staying in a standard city, what are the nightly hotel cap and the daily meal allowance? | ANSWER: {"abstain": false, "answer": "In a standard city, the nightly hotel cap is EUR 260 per night, and the daily me | ANSWER: {"abstain": false, "answer": "On an international trip staying in a standard city, the nightly hotel cap is EU |

## Item flips — serving

3 improved, 1 regressed, 8 wrong in both; exact McNemar p = 0.625


## Grounded accuracy by category

| Category | baseline-qwen2.5-1.5b serving | sft-qwen2.5-1.5b serving | baseline-qwen2.5-1.5b intent | sft-qwen2.5-1.5b intent |
|---|---|---|---|---|
| ambiguous | 1.000 | 1.000 | 1.000 | 1.000 |
| answerable | 0.650 | 0.800 | 0.850 | 0.800 |
| answerable_multi | 0.667 | 0.333 | 0.667 | 0.333 |
| distractor | 0.667 | 0.667 | 0.556 | 0.667 |
| no_evidence | 1.000 | 1.000 | 1.000 | 1.000 |
| off_topic | 1.000 | 1.000 | 0.750 | 1.000 |

## Error buckets (intent)

| Bucket | baseline-qwen2.5-1.5b | sft-qwen2.5-1.5b |
|---|---|---|
| answered_unanswerable:distractor | h21, h23, h25, h29 | h23, h25, h29 |
| answered_unanswerable:off_topic | h31 | – |
| cited_nonexistent_passage | h02, h27, h31 | – |
| flag_misuse | h01, h02, h10, h12, h14, h18, h20, h21, h22, h24, h26, h27, h28, h30, h32, h35, h36, h37, h39 | – |
| has_unsupported_claims | h01, h13, h21, h25, h31, h38, h40 | h10, h13, h25, h38, h39, h40 |
| inappropriate_abstention | h02 | h01, h02, h14, h20 |
| not_json | h33, h34 | – |
| right_answer_wrong_citation | – | h39 |
| wrong_answer | h14, h20, h40 | h40 |
