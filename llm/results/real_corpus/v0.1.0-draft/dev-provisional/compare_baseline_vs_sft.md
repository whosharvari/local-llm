# baseline-qwen2.5-1.5b vs sft-qwen2.5-1.5b (metrics v2)

n = 34 (20 answerable, 14 unanswerable, 0 ambiguous). One item = 2.9 points of grounded accuracy. Small-sample evidence only; no statistical significance is claimed.

**serving** = what the API returns (flag respected, contradictions withheld, citation required). **intent** = whether the answer text is substantive, flag ignored.

| Metric | baseline-qwen2.5-1.5b serving | sft-qwen2.5-1.5b serving | baseline-qwen2.5-1.5b intent | sft-qwen2.5-1.5b intent | Better (intent) |
|---|---|---|---|---|---|
| Grounded accuracy (all items) | 0.500 | 0.529 | 0.382 | 0.529 | sft-qwen2.5-1.5b |
| Grounded accuracy excl. ambiguous h30 | 0.500 | 0.529 | 0.382 | 0.529 | sft-qwen2.5-1.5b |
| Answer correct (answerable) | 0.250 | 0.700 | 0.600 | 0.750 | sft-qwen2.5-1.5b |
| Wrong answer (answerable) | 0.000 | 0.200 | 0.050 | 0.250 | baseline-qwen2.5-1.5b |
| Inappropriate abstention (answerable) | 0.750 | 0.100 | 0.350 | 0.000 | sft-qwen2.5-1.5b |
| Correct abstention (unanswerable) | 1.000 | 0.786 | 0.571 | 0.786 | sft-qwen2.5-1.5b |
| Hallucinated answer (unanswerable) | 0.000 | 0.214 | 0.429 | 0.214 | sft-qwen2.5-1.5b |
| Cites >=1 gold passage (answered answerable) | 0.600 | 0.556 | 0.385 | 0.500 | sft-qwen2.5-1.5b |
| Cites all gold passages (answered answerable) | 0.600 | 0.500 | 0.308 | 0.450 | sft-qwen2.5-1.5b |
| Citation precision (answered answerable) | 0.467 | 0.556 | 0.333 | 0.500 | sft-qwen2.5-1.5b |
| Unsupported claim rate (claims in answers) | 0.500 | 0.455 | 0.794 | 0.500 | sft-qwen2.5-1.5b |
| Answers with no valid citation (count) | 0 | 0 | 4 | 2 | sft-qwen2.5-1.5b |
| Answer token F1 vs gold (answerable) | 0.158 | 0.509 | 0.342 | 0.551 | sft-qwen2.5-1.5b |

| Output / runtime | baseline-qwen2.5-1.5b | sft-qwen2.5-1.5b | Better |
|---|---|---|---|
| Valid JSON | 0.853 | 1.000 | sft-qwen2.5-1.5b |
| Schema-valid JSON (3 keys, boolean flag) | 0.147 | 1.000 | sft-qwen2.5-1.5b |
| Invented citation ids (count) | 0 | 2 | baseline-qwen2.5-1.5b |
| Inference errors (count) | 0 | 0 | = |
| Latency p50 (s), single run | 5.831 | 6.424 | baseline-qwen2.5-1.5b |
| Latency p95 (s), single run | 9.879 | 10.288 | baseline-qwen2.5-1.5b |
| SGLang scheduler memory (GB), single reading | 6.750 | 12.000 | baseline-qwen2.5-1.5b |
| Contract violations | {'non_boolean_flag': 24, 'flag_true_but_substantive_answer': 12, 'not_json': 5, 'text_outside_json': 3, 'missing_keys': 2} | {'invalid_citation_ids': 2} | |

## Item flips — intent

9 improved, 4 regressed, 12 wrong in both; exact McNemar p = 0.267

| id | category | question | baseline-qwen2.5-1.5b | sft-qwen2.5-1.5b |
|---|---|---|---|---|
| rcd-ooc-04 (improved) | out_of_corpus | How does PagedAttention in vLLM manage the KV cache? | ANSWER: abstain citations [] | ABSTAIN: {"abstain": true, "answer": "", "citations": []} |
| rcd-ooc-05 (improved) | out_of_corpus | What is the recommended adult dosage of ibuprofen? | ANSWER: abstain citations [] {"abstain": "true", "answer": "The recommended adult dosage of ibuprofen can vary dependi | ABSTAIN: {"abstain": true, "answer": "", "citations": []} |
| rcd-ooc-06 (improved) | out_of_corpus | How many paid vacation days do employees get per year? | ANSWER: abstain citations [] {"abstain": "true", "answer": "The number of paid vacation days per year varies by countr | ABSTAIN: {"abstain": true, "answer": "", "citations": []} |
| rcd-rag-07 (improved) | numerical | How accurately does RAG answer questions about world leaders when using the index that matches the year of the leaders? | ABSTAIN: {"abstain": "true", "answer": "abstain", "citations": ["rag_knowledge_intensive.pdf::0027"]} | ANSWER: {"abstain": false, "answer": "RAG answers 70% correctly using the 2016 index for 2016 world leaders and 68% us |
| rcd-rag-09 (improved) | numerical | What hardware were the RAG models trained on? | ABSTAIN: {"abstain": "true", "answer": "", "citations": ["rag_knowledge_intensive.pdf::0108"]} | ANSWER: {"abstain": false, "answer": "The RAG models were trained using mixed precision floating point arithmetic, dis |
| rcd-rag-10 (improved) | numerical | How much CPU memory is needed to store the RAG document index for all of Wikipedia? | ABSTAIN: {"abstain": "true", "answer": "", "citations": ["rag_knowledge_intensive.pdf::0109"]} | ANSWER: {"abstain": false, "answer": "The RAG document index for all of Wikipedia requires ∼100 GB of CPU memory.", "c |
| rcd-resnet-05 (improved) | numerical | What mini-batch size, initial learning rate, weight decay and momentum were used for ImageNet training? | ABSTAIN: {"abstain": "true", "answer": "", "citations": ["resnet_deep_residual_learning.pdf::0028", "resnet_deep_residu | ANSWER: {"abstain": false, "answer": "For ImageNet training, the mini-batch size was 256, the initial learning rate wa |
| rcd-resnet-07 (improved) | numerical | What top-1 error (10-crop testing) do the 34-layer plain network and the 34-layer ResNet have on ImageNet validation? | ABSTAIN: {"abstain": "true", "answer": "abstain", "citations": ["resnet_deep_residual_learning.pdf::0034"]} | ANSWER: {"abstain": false, "answer": "The 34-layer plain network has a top-1 error of 28.54% on ImageNet validation, w |
| rcd-swap-01 (improved) | context_swap | Which generator model does RAG use, and how many parameters does it have? | ANSWER: {"abstain": "true", "answer": "The passage does not provide specific information about the generator model use | ABSTAIN: {"abstain": true, "answer": "", "citations": []} |
| rcd-rag-01 (regressed) | answerable | What two components make up a Retrieval-Augmented Generation (RAG) model? | ANSWER: {"abstain": false, "answer": "A Retrieval-Augmented Generation (RAG) model consists of a pre-trained seq2seq t | ANSWER: {"abstain": false, "answer": "A Retrieval-Augmented Generation (RAG) model consists of two components: a pre-t |
| rcd-resnet-01 (regressed) | answerable | What problem do residual connections address in very deep neural networks? | ANSWER: {"abstain": "true", "answer": "The problem addressed by residual connections in very deep neural networks is t | ANSWER: {"abstain": false, "answer": "Residual connections address the problem of vanishing/exploding gradients in ver |
| rcd-resnet-09 (regressed) | answerable | What two shortcut options are considered when the dimensions increase? | ANSWER: {"abstain": "true", "answer": "The shortcut options considered when the dimensions increase are: (A) The short | ANSWER: {"abstain": false, "answer": "When the dimensions increase, two options are considered: (A) The shortcut still |
| rcd-swap-02 (regressed) | context_swap | What hardware were the RAG models trained on? | ABSTAIN: {"abstain": "true", "answer": "", "citations": ["resnet_deep_residual_learning.pdf::0030"]} | ANSWER: {"abstain": false, "answer": "The RAG models were trained on the 1.28 million training images.", "citations":  |
| rcd-rag-02 (wrong in both) | answerable | What retriever architecture does the RAG paper use to fetch supporting documents? | ANSWER: {"abstain": false, "answer": "The RAG paper uses a retriever architecture called Dense Passage Retriever (DPR) | ANSWER: {"abstain": false, "answer": "The RAG paper uses a retriever architecture called Dense Passage Retriever (DPR) |
| rcd-rag-03 (wrong in both) | answerable | What is the difference between the RAG-Sequence and RAG-Token model variants? | ANSWER: {"abstain": "true", "answer": "The RAG-Sequence model uses the same document to predict each target token, whi | ANSWER: {"abstain": false, "answer": "The RAG-Sequence model uses the same document to predict each target token, whil |
| rcd-rag-05 (wrong in both) | answerable | Which RAG components are fine-tuned during training, and which are kept fixed? | ANSWER: {"abstain": false, "answer": "The document encoder (and index) are kept fixed, only fine-tuning the query enco | ANSWER: {"abstain": false, "answer": "During training, the query encoder BERTq and the BART generator are fine-tuned,  |
| rcd-rag-08 (wrong in both) | numerical | On FEVER, how often is the top document retrieved by RAG from a gold article? | ABSTAIN: {"abstain": "true", "answer": "abstain", "citations": ["rag_knowledge_intensive.pdf::0044"]} | ANSWER: {"abstain": false, "answer": "The top document retrieved by RAG from a gold article is present in the top 10 r |
| rcd-rag-11 (wrong in both) | numerical | In the Jeopardy question generation human evaluation, how often was RAG judged more factual than BART? | ANSWER: {"abstain": "true", "answer": "42.7%", "citations": ["rag_knowledge_intensive.pdf::0107"]} | ANSWER: {"abstain": false, "answer": "RAG was more factual in 42.7% of cases, and both RAG and BART were factual in a  |
| rcd-resnet-02 (wrong in both) | answerable | What is a shortcut (skip) connection in a residual block? | ANSWER: {"abstain": false, "answer": "A shortcut connection, also known as a skip connection or skip link, is a type o | ANSWER: {"abstain": false, "answer": "A shortcut connection, also known as a skip connection or skip link, is a type o |
| rcd-resnet-03 (wrong in both) | answerable | What layers make up the bottleneck building block used in ResNet-50/101/152? | ANSWER: {"abstain": "true", "answer": "The three layers in the bottleneck building block used in ResNet-50/101/152 are | ANSWER: {"abstain": false, "answer": "The bottleneck building block used in ResNet-50/101/152 consists of three layers |
| rcd-resnet-04 (wrong in both) | numerical | What top-5 error did the ResNet ensemble achieve on the ImageNet test set? | ABSTAIN: {"abstain": "true", "answer": "abstain", "citations": ["resnet_deep_residual_learning.pdf::0093"]} | ANSWER: {"abstain": false, "answer": "The ResNet ensemble achieved a top-5 localization error of 9.0% on the test set. |
| rcd-resnet-06 (wrong in both) | numerical | How many FLOPs does the 152-layer ResNet have? | ANSWER: {"abstain": "true", "answer": "11.3 billion FLOPs", "citations": ["resnet_deep_residual_learning.pdf::0038"]} | ANSWER: {"abstain": false, "answer": "The 152-layer ResNet has 11.3 billion FLOPs.", "citations": ["resnet_deep_residu |
| rcd-resnet-08 (wrong in both) | numerical | How was training of the 110-layer ResNet on CIFAR-10 warmed up? | ANSWER: {"abstain": "true", "answer": "The training of the 110-layer ResNet on CIFAR-10 was warmed up with an initial  | ANSWER: {"abstain": false, "answer": "The training of the 110-layer ResNet on CIFAR-10 was warmed up with an initial l |
| rcd-resnet-10 (wrong in both) | unanswerable | How many GPUs were used to train the ImageNet classification models? | ANSWER: {"abstain": "true", "answer": "2", "citations": ["resnet_deep_residual_learning.pdf::0051"]} | ANSWER: {"abstain": false, "answer": "The ImageNet classification models were trained using two GPUs.", "citations": [ |
| rcd-resnet-11 (wrong in both) | unanswerable | What top-1 error does the 1202-layer ResNet achieve on ImageNet? | ANSWER: {"abstain": "true", "answer": "7.93%", "citations": ["resnet_deep_residual_learning.pdf::0058"]} | ANSWER: {"abstain": false, "answer": "The 1202-layer ResNet achieves a top-1 error of 7.93% on the ImageNet validation |

## Item flips — serving

5 improved, 4 regressed, 12 wrong in both; exact McNemar p = 1.000


## Grounded accuracy by category

| Category | baseline-qwen2.5-1.5b serving | sft-qwen2.5-1.5b serving | baseline-qwen2.5-1.5b intent | sft-qwen2.5-1.5b intent |
|---|---|---|---|---|
| answerable | 0.125 | 0.000 | 0.375 | 0.000 |
| context_swap | 1.000 | 0.750 | 0.750 | 0.750 |
| numerical | 0.167 | 0.583 | 0.167 | 0.583 |
| out_of_corpus | 1.000 | 1.000 | 0.500 | 1.000 |
| unanswerable | 1.000 | 0.500 | 0.500 | 0.500 |

## Error buckets (intent)

| Bucket | baseline-qwen2.5-1.5b | sft-qwen2.5-1.5b |
|---|---|---|
| answered_unanswerable:context_swap | rcd-swap-01 | rcd-swap-02 |
| answered_unanswerable:out_of_corpus | rcd-ooc-04, rcd-ooc-05, rcd-ooc-06 | – |
| answered_unanswerable:unanswerable | rcd-resnet-10, rcd-resnet-11 | rcd-resnet-10, rcd-resnet-11 |
| cited_nonexistent_passage | – | rcd-resnet-02, rcd-resnet-09 |
| flag_misuse | rcd-rag-03, rcd-rag-07, rcd-rag-08, rcd-rag-09, rcd-rag-10, rcd-rag-11, rcd-resnet-01, rcd-resnet-03, rcd-resnet-04, rcd-resnet-05, rcd-resnet-06, rcd-resnet-07, rcd-resnet-08, rcd-resnet-09, rcd-resnet-10, rcd-resnet-11, rcd-swap-01, rcd-swap-02, rcd-swap-04, rcd-ooc-01, rcd-ooc-02, rcd-ooc-03, rcd-ooc-05, rcd-ooc-06 | – |
| has_unsupported_claims | rcd-rag-02, rcd-rag-03, rcd-rag-04, rcd-rag-05, rcd-rag-11, rcd-resnet-01, rcd-resnet-02, rcd-resnet-03, rcd-resnet-06, rcd-resnet-08, rcd-resnet-10, rcd-swap-01, rcd-ooc-04, rcd-ooc-05, rcd-ooc-06 | rcd-rag-02, rcd-rag-03, rcd-rag-05, rcd-rag-11, rcd-resnet-01, rcd-resnet-02, rcd-resnet-03, rcd-resnet-06, rcd-resnet-08, rcd-resnet-09, rcd-resnet-11 |
| inappropriate_abstention | rcd-rag-07, rcd-rag-08, rcd-rag-09, rcd-rag-10, rcd-resnet-04, rcd-resnet-05, rcd-resnet-07 | – |
| not_json | rcd-rag-12, rcd-rag-13, rcd-resnet-02, rcd-swap-03, rcd-ooc-04 | – |
| right_answer_wrong_citation | rcd-rag-02, rcd-rag-03, rcd-rag-05, rcd-rag-11, rcd-resnet-03, rcd-resnet-06, rcd-resnet-08 | rcd-rag-02, rcd-rag-03, rcd-rag-05, rcd-rag-11, rcd-resnet-03, rcd-resnet-06, rcd-resnet-08, rcd-resnet-09 |
| wrong_answer | rcd-resnet-02 | rcd-rag-01, rcd-rag-08, rcd-resnet-01, rcd-resnet-02, rcd-resnet-04 |
