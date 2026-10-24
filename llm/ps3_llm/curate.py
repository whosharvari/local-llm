"""Stage 1 — data curation with CuratorKIT, then grounded SFT example construction.

Step A (CuratorKIT): generate QA pairs from each training chunk with an LLM served by SGLang,
    and keep only pairs that pass CuratorKIT's HallucinationGate (answer verified against the
    exact source chunk). Output: <out>/curatorkit/ (manifest.json, rejected.jsonl, dataset card...).

Step B (this module): turn each accepted QA pair into grounded chat examples in the exact
    serving format (ps3_llm.prompting):
      - answerable:  gold chunk + distractor chunks, target cites the gold chunk id
      - abstain:     same question, gold chunk removed (same-document distractors only)
      - no_evidence: same question with no passages (a fraction)
    Split train/dev BY DOCUMENT so dev never shares a document with train. The held-out test set
    (data/mock/heldout_*) is a separate set of documents and is never read here.

    python -m ps3_llm.curate --chunks data/mock/train_chunks.jsonl \
        --llm-model openai/mlx-community/Qwen2.5-7B-Instruct-4bit --api-base http://127.0.0.1:30000/v1
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .metrics import claim_supported
from .prompting import SYSTEM_PROMPT, format_target, render_user_prompt
from .schemas import EvidenceChunk, GroundedQuery

ROOT = Path(__file__).resolve().parent.parent


def export_train_chunks(docs_path: Path, out_path: Path) -> list[dict]:
    """Flatten the mock train_docs.json into contract-format chunk rows (what Member 1 will supply)."""
    docs = json.loads(docs_path.read_text())
    rows = [
        {"chunk_id": cid, "text": text, "source": doc["source"], "doc_id": key}
        for key, doc in docs.items() if not key.startswith("_")
        for cid, text in doc["chunks"].items()
    ]
    out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows


def run_curatorkit(chunks_path: Path, out_dir: Path, llm_model: str, api_base: str,
                   num_questions: int, hallucination_threshold: float, concurrency: int,
                   judge_model: str | None = None, judge_base: str | None = None):
    from curatorkit import Curator, CuratorConfig

    # CuratorKIT reads raw text rows as language-modelling chunks ("pretrain" format), injects each
    # chunk into the QA prompt and stores it in sample.input so the HallucinationGate can check
    # the generated answer against exactly that chunk.
    text_only = out_dir / "chunks_text.jsonl"
    text_only.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in chunks_path.read_text().splitlines() if line.strip()]
    text_only.write_text("".join(json.dumps({"text": r["text"]}) + "\n" for r in rows))

    cfg = CuratorConfig(
        name="ps3_grounded_qa",
        dataset=str(text_only),
        format="pretrain",
        llm_model=llm_model,
        llm_api_base=api_base,
        llm_api_key="EMPTY",
        llm_temperature=0.3,
        llm_concurrency=concurrency,
        judge_llm_model=judge_model or llm_model,     # a different model makes the gate independent of the generator
        judge_llm_api_base=judge_base or api_base,
        generation_task="qa",
        num_questions=num_questions,
        difficulty="medium",
        hallucination_threshold=hallucination_threshold,
        min_tokens=3,
        dedup="exact",
        export_formats=["alpaca"],
        output_dir=str(out_dir / "curatorkit"),
    )
    return rows, Curator(cfg).run()


_BOILERPLATE = re.compile(
    r",?\s*(?:as per|according to|as (?:stated|mentioned|detailed|outlined|specified|described|indicated|noted|"
    r"highlighted)(?:\s+(?:in|by))?)\s+the\s+(?:company's\s+)?(?:provided\s+|given\s+|source\s+|above\s+)?"
    r"(?:text|passage|policy|source)(?:\s+text)?",
    re.IGNORECASE,
)


def strict_ground(answer: str, source: str) -> tuple[str | None, str]:
    """Second, deterministic grounding gate on top of CuratorKIT's LLM-judged HallucinationGate.

    Removes "as stated in the provided text" boilerplate, then keeps only sentences whose numbers
    and >=60% of content words are in the source chunk (same check as the eval metric). Returns
    (cleaned answer, "") or (None, reason) when the leading sentence itself is unsupported.
    """
    text = _BOILERPLATE.sub("", answer).strip()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sents:
        return None, "empty_after_cleaning"
    if not claim_supported(sents[0], source):
        return None, "leading_sentence_unsupported"
    kept = [s for s in sents if claim_supported(s, source)]
    out = " ".join(kept)
    if not out.endswith((".", "!", "?")):
        out += "."
    return out, ("dropped_unsupported_sentences" if len(kept) < len(sents) else "")


def build_examples(qa: list[dict], chunks: list[dict], seed: int, n_distractors: int = 2,
                   abstain_ratio: float = 0.6, no_evidence_ratio: float = 0.15) -> list[dict]:
    """qa rows: {question, answer, chunk_id}. Returns chat-format rows with metadata."""
    rng = random.Random(seed)
    by_id = {c["chunk_id"]: c for c in chunks}
    by_doc: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c["chunk_id"])

    def to_chunk(cid: str) -> EvidenceChunk:
        c = by_id[cid]
        return EvidenceChunk(chunk_id=cid, text=c["text"], source=c["source"])

    def row(q: str, ev_ids: list[str], target: str, kind: str, gold: str, doc: str) -> dict:
        query = GroundedQuery(question=q, evidence=[to_chunk(i) for i in ev_ids])
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_user_prompt(query)},
                {"role": "assistant", "content": target},
            ],
            "kind": kind, "gold_chunk": gold, "doc_id": doc,
        }

    out = []
    for item in qa:
        gold = item["chunk_id"]
        doc = by_id[gold]["doc_id"]
        same_doc = [c for c in by_doc[doc] if c != gold]
        other = [c for c in by_id if by_id[c]["doc_id"] != doc]

        k_same = min(len(same_doc), rng.randint(1, n_distractors))
        distractors = rng.sample(same_doc, k_same) + rng.sample(other, rng.randint(0, 1))
        ev = distractors + [gold]
        rng.shuffle(ev)
        out.append(row(item["question"], ev, format_target(item["answer"], [gold], False), "answerable", gold, doc))

        if rng.random() < abstain_ratio:
            ev_abs = rng.sample(same_doc, min(len(same_doc), n_distractors))
            out.append(row(item["question"], ev_abs, format_target("", [], True), "abstain_distractor", gold, doc))
        if rng.random() < no_evidence_ratio:
            out.append(row(item["question"], [], format_target("", [], True), "abstain_no_evidence", gold, doc))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, default=ROOT / "data/mock/train_docs.json")
    ap.add_argument("--chunks", type=Path, default=ROOT / "data/mock/train_chunks.jsonl",
                    help="contract-format chunk JSONL (chunk_id, text, source, doc_id)")
    ap.add_argument("--out", type=Path, default=ROOT / "data/curated")
    ap.add_argument("--llm-model", default="openai/mlx-community/Qwen2.5-7B-Instruct-4bit",
                    help="LiteLLM model string; 'openai/<served id>' targets the SGLang OpenAI endpoint")
    ap.add_argument("--api-base", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--num-questions", type=int, default=3)
    ap.add_argument("--hallucination-threshold", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dev-docs", default="safety,classification", help="documents reserved for the dev split")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--reuse-curatorkit", action="store_true",
                    help="skip generation; rebuild SFT data from an existing <out>/curatorkit/sft_alpaca.jsonl")
    args = ap.parse_args()

    if not args.chunks.exists():
        export_train_chunks(args.docs, args.chunks)
    if args.reuse_curatorkit:
        chunks = [json.loads(line) for line in args.chunks.read_text().splitlines() if line.strip()]
        alpaca = args.out / "curatorkit" / "sft_alpaca.jsonl"
        passed = [json.loads(line) for line in alpaca.read_text().splitlines() if line.strip()]
        n_ck_rejected = sum(1 for line in (args.out / "curatorkit/rejected.jsonl").read_text().splitlines() if line)
    else:
        chunks, result = run_curatorkit(args.chunks, args.out, args.llm_model, args.api_base,
                                        args.num_questions, args.hallucination_threshold, args.concurrency)
        result.print_summary()
        passed = [{"instruction": s.instruction, "input": s.input, "output": s.output} for s in result.passed]
        n_ck_rejected = len(result.rejected)

    text_to_id = {c["text"].strip(): c["chunk_id"] for c in chunks}
    qa, unmatched, strict_rejects, strict_log = [], 0, [], Counter()
    for s in passed:
        cid = text_to_id.get((s["input"] or "").strip())
        if cid is None:
            unmatched += 1
            continue
        answer, note = strict_ground(s["output"].strip(), s["input"])
        strict_log[note or "kept_unchanged"] += 1
        if answer is None:
            strict_rejects.append({**s, "chunk_id": cid, "reason": note})
            continue
        qa.append({"question": s["instruction"].strip(), "answer": answer, "chunk_id": cid,
                   "curatorkit_answer": s["output"].strip()})
    (args.out / "accepted_qa.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in qa))
    (args.out / "strict_gate_rejected.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in strict_rejects))

    examples = build_examples(qa, chunks, seed=args.seed)
    dev_docs = set(args.dev_docs.split(","))
    splits = {"train": [e for e in examples if e["doc_id"] not in dev_docs],
              "dev": [e for e in examples if e["doc_id"] in dev_docs]}
    for name, rows in splits.items():
        (args.out / f"sft_{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    stats = {
        "curatorkit_passed": len(passed), "curatorkit_rejected": n_ck_rejected,
        "unmatched_to_chunk": unmatched, "strict_gate": dict(strict_log), "qa_pairs": len(qa),
        **{f"{n}_examples": len(r) for n, r in splits.items()},
        **{f"{n}_by_kind": {k: sum(e["kind"] == k for e in r) for k in
                            ("answerable", "abstain_distractor", "abstain_no_evidence")} for n, r in splits.items()},
        "dev_docs": sorted(dev_docs),
    }
    (args.out / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
