# baseline-qwen2.5-1.5b vs sft-qwen2.5-1.5b

Same held-out set (n=40), same metrics, temperature 0.

| Metric | baseline-qwen2.5-1.5b | sft-qwen2.5-1.5b | Better |
|---|---|---|---|
| Grounded accuracy (all 40) | 0.575 | 0.775 | sft-qwen2.5-1.5b |
| Answer accuracy (answerable) | 0.652 | 0.783 | sft-qwen2.5-1.5b |
| Answer token F1 (answerable) | 0.530 | 0.456 | baseline-qwen2.5-1.5b |
| Appropriate abstention (unanswerable) | 0.471 | 0.824 | sft-qwen2.5-1.5b |
| Hallucinated answer rate (unanswerable) | 0.529 | 0.176 | sft-qwen2.5-1.5b |
| Over-abstention rate (answerable) | 0.261 | 0.174 | sft-qwen2.5-1.5b |
| Unsupported claim rate (answered) | 0.407 | 0.273 | sft-qwen2.5-1.5b |
| Answers with invented numbers | 1 | 0 | sft-qwen2.5-1.5b |
| Answers without citations | 5 | 0 | sft-qwen2.5-1.5b |
| Cited non-existent passage ids | 3 | 0 | sft-qwen2.5-1.5b |
| Citation precision | 1.000 | 0.947 | baseline-qwen2.5-1.5b |
| Citation recall | 0.971 | 0.895 | baseline-qwen2.5-1.5b |
| Valid JSON output rate | 0.950 | 1.000 | sft-qwen2.5-1.5b |
| Latency p50 (s) | 1.754 | 2.183 | baseline-qwen2.5-1.5b |
| Latency p95 (s) | 2.882 | 4.148 | baseline-qwen2.5-1.5b |
| Mean completion tokens | 33.600 | 32.200 | sft-qwen2.5-1.5b |
| Throughput (queries/s, concurrency 4) | 2.140 | 1.535 | baseline-qwen2.5-1.5b |
| SGLang scheduler memory (GB) | 8.690 | 8.220 | sft-qwen2.5-1.5b |

## Grounded accuracy by category

| Category | baseline-qwen2.5-1.5b | sft-qwen2.5-1.5b |
|---|---|---|
| answerable | 0.650 | 0.800 |
| answerable_multi | 0.667 | 0.333 |
| distractor | 0.500 | 0.700 |
| no_evidence | 0.667 | 1.000 |
| off_topic | 0.250 | 1.000 |

## Per-example changes (10 fixed, 2 regressed, 7 wrong in both)

### Fixed

| id | category | question | baseline-qwen2.5-1.5b | sft-qwen2.5-1.5b |
|---|---|---|---|---|
| h10 | answerable | What hotline extension should be used to report an incident? | ABSTAIN | The hotline extension to use for reporting an incident is 4400. ['incident#c2'] |
| h12 | answerable | When is a post-incident review due for a P2 incident? | ABSTAIN | A post-incident review for a P2 incident must be completed within 10 business days. ['incident#c4'] |
| h17 | answerable | How long do I have to submit a travel expense claim? | 30 days ['travel#c5'] | You have to submit a travel expense claim within 30 days of the trip's end date. ['travel#c5'] |
| h18 | answerable | Who must approve business class travel? | ABSTAIN | Business class travel must be approved by a member of the Executive Committee. ['travel#c1'] |
| h26 | distractor | Who is the current Chief Information Security Officer? | Current Chief Information Security Officer not specified in the given evidence. ['incident#c3'] | ABSTAIN |
| h27 | distractor | What is the response time target for a P3 incident? | abstain [] | ABSTAIN |
| h31 | off_topic | How many days of annual leave do new employees receive? | New employees receive 20 days of annual leave. [] | ABSTAIN |
| h33 | off_topic | What is the minimum password length for company laptops? | abstain citations: [] [] | ABSTAIN |
| h34 | off_topic | Who founded Halden Group? | abstain citations: [] [] | ABSTAIN |
| h37 | no_evidence | How do I request a second monitor? | abstain [] | ABSTAIN |

### Regressed

| id | category | question | baseline-qwen2.5-1.5b | sft-qwen2.5-1.5b |
|---|---|---|---|---|
| h01 | answerable | Who can approve a purchase of EUR 1,800? | The purchase of EUR 1,800 can be approved by the requesting employee's line manager. ['procurement#c1'] | ABSTAIN |
| h39 | answerable_multi | For a P1 incident involving personal data, who must be notified and within what times? | For a P1 incident involving personal data, the Chief Information Security Officer must be notified within 30 minutes of classification, and the Data Protection Officer must assess whether the regulator must be notified within 24 hours. ['incident#c3'] | For a P1 incident involving personal data, the Chief Information Security Officer must be notified within 30 minutes of classification, and the Data Protection Officer must assess whether the regulator must be notified within 24 hours. ['incident#c1'] |

### Wrong in both

| id | category | question | baseline-qwen2.5-1.5b | sft-qwen2.5-1.5b |
|---|---|---|---|---|
| h02 | answerable | How many written quotes are needed for a EUR 10,000 purchase? | ABSTAIN | ABSTAIN |
| h14 | answerable | What class can I book for a 4-hour flight? | ABSTAIN | ABSTAIN |
| h20 | answerable | Is alcohol reimbursable on a business trip? | ABSTAIN | ABSTAIN |
| h23 | distractor | Who chairs the tender evaluation panel? | The evaluation panel must include at least one member from Finance. ['procurement#c2'] | The evaluation panel must include at least one member from Finance. ['procurement#c2'] |
| h25 | distractor | Within how many hours must the regulator be notified of a personal data breach? | 24 hours ['incident#c1'] | The regulator must be notified within 24 hours of a personal data breach. ['incident#c1'] |
| h29 | distractor | Which cities are on the high-cost city list? | The list of high-cost cities is maintained by the Finance team and reviewed annually. ['travel#c2'] | The high-cost cities are maintained by the Finance team and reviewed annually. ['travel#c2'] |
| h40 | answerable_multi | On an international trip staying in a standard city, what are the nightly hotel cap and the daily meal allowance? | In a standard city, the nightly hotel cap is EUR 260 per night, and the daily meal allowance is EUR 75. ['travel#c2'] | On an international trip staying in a standard city, the nightly hotel cap is EUR 260 per night, and the daily meal allowance is EUR 75. ['travel#c2'] |

## Error buckets

| Bucket | baseline-qwen2.5-1.5b | sft-qwen2.5-1.5b |
|---|---|---|
| answered_unanswerable:distractor | h23, h25, h26, h27, h29 | h23, h25, h29 |
| answered_unanswerable:no_evidence | h37 | – |
| answered_unanswerable:off_topic | h31, h33, h34 | – |
| cited_nonexistent_passage | h02, h27, h31 | – |
| has_unsupported_claims | h01, h13, h25, h26, h27, h31, h33, h34, h37, h38, h40 | h10, h13, h25, h38, h39, h40 |
| invented_numbers | h31 | – |
| over_abstained | h02, h10, h12, h14, h18, h20 | h01, h02, h14, h20 |
| right_answer_wrong_citation | – | h39 |
| unparseable_output | h33, h34 | – |
| wrong_answer | h17, h40 | h40 |
