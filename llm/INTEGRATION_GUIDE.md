# Integration guide — Papermind retrieval → PS3 model module

**Status: PROPOSED contract v1.1 — not yet approved by Member 1, and not yet tested against the real
Papermind module.** Everything below was executed against this module only (mock adapter and the
SGLang-served fine-tuned model). Integration is complete only after Member 1 runs the checklist at the
end with real Papermind output.

Responsibilities: Papermind (Member 1) owns ingestion, chunking, embedding, the vector store and
retrieval. This module (Member 2) owns prompting, SGLang inference, output parsing, citation validation
and abstention. The only interface is JSON; Papermind never needs to import `ps3_llm`.

---

## 1. Start the model server (SGLang)

Prerequisites: the two environments from `llm/README.md` (`.venv` and `.venv-sglang`, Apple Silicon
lock files in `llm/env/`), and the model weights (§6).

```bash
cd llm
# fine-tuned model (recommended); stable id via --served-model-name
scripts/serve_sglang.sh "$PWD/outputs/sft-qwen2.5-1.5b/merged" --served-model-name ps3-qwen2.5-1.5b-grounded-sft &
scripts/wait_for_server.sh http://127.0.0.1:30000      # blocks until /v1/models answers; prints the model id
# baseline instead:  scripts/serve_sglang.sh Qwen/Qwen2.5-1.5B-Instruct &
```

| Variable / flag | Default | Meaning |
|---|---|---|
| `HOST`, `PORT` | `127.0.0.1`, `30000` | SGLang bind address (env vars read by `serve_sglang.sh`) |
| `CTX` | `4096` | context length (tokens) |
| `SGLANG_PYTHON` | `<repo>/.venv-sglang/bin/python` | interpreter with SGLang |
| `SGLANG_USE_MLX=1` | set automatically on Apple Silicon | SGLang MLX backend |
| `--base-url` (CLI) | `http://127.0.0.1:30000/v1` | where the model module sends requests |
| `--model` (CLI) | discovered from `/v1/models` | served model id; optional |
| `--timeout` (CLI) | `180` s | per-request HTTP timeout |

Only one model is served at a time on a 16 GB machine (~8.5 GB SGLang footprint).

## 2. Call it: JSON in, JSON out

```bash
cd llm
../.venv/bin/python -m ps3_llm.cli --adapter sglang --base-url http://127.0.0.1:30000/v1 < request.json
../.venv/bin/python -m ps3_llm.cli --adapter mock < request.json      # no server; deterministic lexical stand-in
```

- stdin: one request object, or a JSON array of requests (answered in the same order).
- stdout: one response object, or an array of the same length and order.
- Exit code `0` whenever input was JSON — including invalid items, abstentions and server failures, which
  are reported *inside* the response. Exit code `2` only when stdin is not JSON (stdout then holds
  `{"error": "..."}`).
- No API keys or secrets are needed (`api_key` sent to SGLang is the placeholder `EMPTY`).

Minimal Python client without internal imports: [`examples/integration/papermind_client.py`](examples/integration/papermind_client.py)
(`ask(request_or_list) -> response_or_list`, plus a `render()` showing how to display results).

## 3. Request schema

```json
{
  "query_id": "string (optional, echoed back; recommended)",
  "question": "string (required, non-empty)",
  "evidence": [
    {
      "chunk_id": "string (required)",
      "text":     "string (required)",
      "source":   "string (optional, shown to the model as the document name)",
      "page":     "integer (optional, shown as 'p.N')",
      "score":    "number (optional, informational only — ignored by the model)"
    }
  ]
}
```

`evidence` is required and may be `[]`. Unknown extra fields are ignored.

**Evidence-id rules** (validated; violations return `invalid_request`):
- non-empty string, unique within the request;
- no leading/trailing whitespace and not wrapped in `[...]` (the model sees ids as `[chunk_id]`);
- stable across re-indexing, so a citation can be resolved back to the same passage later; recommended
  form `<document>#p<page>-c<chunk>`, e.g. `procurement.pdf#p2-c2`;
- matched case-sensitively and exactly.

**Evidence budget** (not enforced, but outside it behaviour is untested): ≤ 5 chunks, ≤ ~200 words each.
The model was trained with 0–4 passages totalling ≤ 447 tokens, and the server context is 4096 tokens.
An over-long prompt fails at SGLang and comes back as `inference_error`, not as an abstention.

## 4. Response schema

```json
{
  "answer": "string; '' when abstained",
  "citations": ["chunk_id", "..."],
  "abstained": "boolean",
  "abstain_reason": "string; '' when answered (see table)",
  "query_id": "string",
  "model": "string, e.g. 'sglang:ps3-qwen2.5-1.5b-grounded-sft' ('sglang:unresolved' if no model was reached)",
  "latency_s": "number | null",
  "raw_output": "string; unparsed model text (for logging/debugging, do not display)",
  "parse_ok": "boolean; false if the model output was not JSON, inference failed, or the request was invalid",
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "invalid_citations": ["ids the model invented"]},
  "warnings": ["model-output contract violations; diagnostic only"],
  "contract_version": "1.1"
}
```

**Guarantees**
- `citations` ⊆ request `chunk_id`s. Invented ids are never returned; they are listed in `usage.invalid_citations`.
- `abstained == true` ⇔ `answer == ""` and `citations == []`.
- `abstained == false` ⇒ non-empty `answer` and **at least one valid citation**. An answer the model gave
  without any valid citation is converted to an abstention (`no_valid_citation`): for grounded answering,
  uncited text is not shown.

| `abstain_reason` | Who decided | What Papermind should do |
|---|---|---|
| `""` | — | show `answer` with its `citations` |
| `model_abstained` | model (declined, or wrote refusal text such as "abstain") | show "not found in the documents" |
| `contradictory_flag` | policy: model set `abstain=true` but wrote an answer | treat as not found; logged in `warnings` |
| `no_valid_citation` | policy: answer had no citation from the evidence | treat as not found |
| `unparseable_output` | policy: model output was not JSON | treat as not found; `parse_ok=false` |
| `inference_error: …` | system: server down, HTTP error, timeout | **system error** — retry/alert, do not show "not found" |
| `invalid_request: …` | system: request failed validation, model not called | **bug in the request** — fix the caller |

**Empty evidence** (`"evidence": []`) is valid and is sent to the model, which (in all tests so far)
abstains with `model_abstained`. Send it rather than skipping the call, so abstentions are logged consistently.

`warnings` values: `not_json`, `missing_keys:…`, `flag_missing`, `flag_unrecognised`, `non_boolean_flag:…`,
`flag_true_but_substantive_answer`, `flag_false_but_refusal_or_empty`, `invalid_citation_ids`, `inference_error`.

## 5. Examples (real outputs)

Captured on 2026-10-24 by running `examples/integration/papermind_client.py` against the fine-tuned model
served by SGLang (`--served-model-name ps3-qwen2.5-1.5b-grounded-sft`); full files are in
`examples/integration/` (`<name>.json` request, `response_<name>.json` response). Server log:
`verification/logs/serve_integration_sft.log` (3 completions; the malformed request never reached the model).
Latencies are single measurements on an M4.

### 5.1 Answerable — [`answerable.json`](examples/integration/answerable.json)

Request: "How long must a tender remain open?" with chunks `procurement.pdf#p2-c2` (contains the answer)
and `procurement.pdf#p1-c1`.

```json
{
  "answer": "A tender must remain open for a minimum of 15 business days.",
  "citations": ["procurement.pdf#p2-c2"],
  "abstained": false,
  "abstain_reason": "",
  "query_id": "demo-answerable",
  "model": "sglang:ps3-qwen2.5-1.5b-grounded-sft",
  "latency_s": 3.93,
  "raw_output": "{\"abstain\": false, \"answer\": \"A tender must remain open for a minimum of 15 business days.\", \"citations\": [\"procurement.pdf#p2-c2\"]}",
  "parse_ok": true,
  "usage": {"prompt_tokens": 309, "total_tokens": 350, "completion_tokens": 41, "prompt_tokens_details": null,
            "reasoning_tokens": 0, "invalid_citations": []},
  "warnings": [],
  "contract_version": "1.1"
}
```

### 5.2 Unanswerable (related but insufficient evidence) — [`unanswerable.json`](examples/integration/unanswerable.json)

Request: "What is the approval limit for corporate credit card purchases?" with two procurement chunks
that discuss approval thresholds but not credit cards.

```json
{
  "answer": "",
  "citations": [],
  "abstained": true,
  "abstain_reason": "model_abstained",
  "query_id": "demo-unanswerable",
  "model": "sglang:ps3-qwen2.5-1.5b-grounded-sft",
  "latency_s": 2.279,
  "raw_output": "{\"abstain\": true, \"answer\": \"\", \"citations\": []}",
  "parse_ok": true,
  "usage": {"prompt_tokens": 312, "total_tokens": 330, "completion_tokens": 18, "prompt_tokens_details": null,
            "reasoning_tokens": 0, "invalid_citations": []},
  "warnings": [],
  "contract_version": "1.1"
}
```

`empty_evidence.json` (`"evidence": []`) produced the same shape with `query_id` `demo-empty`.

**Caveat:** the model does not always abstain on near-miss evidence. On the held-out set it still
answered 3 of 16 unanswerable questions with a related but non-answering fact (§7).

### 5.3 Inference failure (server down)

The same answerable request, sent after stopping SGLang:

```json
{
  "answer": "",
  "citations": [],
  "abstained": true,
  "abstain_reason": "inference_error: URLError: <urlopen error [Errno 61] Connection refused>",
  "query_id": "demo-answerable",
  "model": "sglang:unresolved",
  "latency_s": 0.018,
  "raw_output": "",
  "parse_ok": false,
  "usage": {},
  "warnings": ["inference_error"],
  "contract_version": "1.1"
}
```

A hung server produces `inference_error: TimeoutError: …` after `--timeout` seconds (tested).

### 5.4 Malformed request — [`malformed.json`](examples/integration/malformed.json)

Bracketed `chunk_id` and non-string `text`:

```json
{
  "answer": "",
  "citations": [],
  "abstained": true,
  "abstain_reason": "invalid_request: evidence[0].chunk_id '[procurement#c2]' has surrounding whitespace or brackets; evidence[1].text must be a string",
  "query_id": "demo-malformed",
  "model": "sglang:unresolved",
  "latency_s": null,
  "raw_output": "",
  "parse_ok": false,
  "usage": {},
  "warnings": [],
  "contract_version": "1.1"
}
```

## 6. Model weights

| Model | Source | Size |
|---|---|---|
| Baseline `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face Hub (downloaded on first serve) | ~3 GB |
| Fine-tuned `outputs/sft-qwen2.5-1.5b/merged` | **not in Git** (gitignored, 2.9 GB); regenerate with `python -m ps3_llm.train --config configs/sft_qwen2.5-1.5b.yaml` (~10 min on M4), or copy the directory from Member 2 | 2.9 GB |

Expected SHA-256 of the merged weights produced by the recorded run: `verification/checkpoint_sha256.txt`.
A re-trained checkpoint is not guaranteed to be bit-identical (MPS non-determinism).

## 7. What the model can and cannot do (measured, 40 held-out items, v2 metrics)

Results for the fine-tuned model under the serving policy (`results/v2/compare_baseline_vs_sft.md`):
answer correct on 18/23 answerable items, abstains inappropriately on 4/23, answers 3/16 unanswerable items
(hallucinated answers), 100% valid JSON, 0 invented citation ids. Compared with the untuned baseline it
mainly follows the output contract reliably. It does **not** demonstrate fewer hallucinated facts once
the baseline's formatting errors are accounted for (see `VERIFICATION_REPORT.md`). Forty synthetic items
are limited evidence; expect different numbers on Papermind's real documents.

## 8. Tests

```bash
cd llm
../.venv/bin/python -m pytest -q tests                        # 65 tests; no model server needed
../.venv/bin/python -m pytest -q tests/test_adapter_contract.py  # contract only: CLI, batch order, errors, timeout
python3 examples/integration/papermind_client.py examples/integration/answerable.json --adapter mock   # no server
```

## 9. Integration checklist (for Member 1 + Member 2 together)

- [ ] Member 1 approves this contract (or proposes changes) — then bump `contract_version`.
- [ ] Papermind emits `chunk_id`s that satisfy §3 for real documents (run a few hundred requests through
      `--adapter mock`; expect zero `invalid_request`).
- [ ] Papermind respects the evidence budget (≤ 5 chunks, ≤ ~200 words each) or the budget is renegotiated.
- [ ] Papermind distinguishes `inference_error` / `invalid_request` from model abstentions in its UI and logs.
- [ ] Papermind resolves every returned citation id back to its passage for display.
- [ ] End-to-end run: real Papermind retrieval → `ps3_llm.cli --adapter sglang` on real questions, with
      results recorded. **Not yet done.**
- [ ] No changes to Papermind's branch from this module; merge coordinated by both members.
