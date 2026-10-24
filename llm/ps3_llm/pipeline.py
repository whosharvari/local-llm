"""End-to-end grounded QA: question -> retrieval -> (optional gate) -> SGLang-served model -> checked answer.

    python -m ps3_llm.pipeline "How many layers does the BERT-base encoder have?"            # SGLang on :30000
    python -m ps3_llm.pipeline "..." --adapter mock --log logs/ask.jsonl --json

Steps: retrieve the top-k chunks from the vector store (ps3_llm.retrieval) keeping chunk id, document and
page; optionally apply the retrieval gate; build the prompt from the retrieved passages only
(ps3_llm.prompting, the same format the model was trained and evaluated with); call the model; parse with
the serving policy (invalid citation ids dropped, malformed output -> abstention); then withhold answers
that state numbers absent from the passages they cite. No dataset labels are read on this path.

Abstain reasons added by the pipeline (on top of ps3_llm.schemas): "retrieval_error: ...",
"retrieval_gate" (gate predicts the corpus cannot answer), "unsupported_numbers".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .adapters import get_adapter
from .metrics import numbers
from .retrieval import Gate, Retriever, features
from .schemas import EvidenceChunk, GroundedAnswer, GroundedQuery

REASON_RETRIEVAL = "retrieval_error: "
REASON_GATE = "retrieval_gate"
REASON_NUMBERS = "unsupported_numbers"


def unsupported_numbers(answer: GroundedAnswer, query: GroundedQuery) -> list[str]:
    """Numbers in the answer that occur in none of the passages it cites."""
    cited = " ".join(c.text for c in query.evidence if c.chunk_id in answer.citations)
    return sorted(numbers(answer.answer) - numbers(cited))


def guard(answer: GroundedAnswer, query: GroundedQuery) -> GroundedAnswer:
    if answer.abstained:
        return answer
    missing = unsupported_numbers(answer, query)
    if not missing:
        return answer
    return GroundedAnswer(answer="", citations=[], abstained=True, abstain_reason=REASON_NUMBERS,
                          query_id=answer.query_id, model=answer.model, latency_s=answer.latency_s,
                          raw_output=answer.raw_output, parse_ok=answer.parse_ok, usage=answer.usage,
                          warnings=answer.warnings + [f"unsupported_numbers:{missing}"])


def ask(question: str, retriever, adapter, k: int = 4, gate: Gate | None = None, query_id: str = "") -> dict:
    t0 = time.perf_counter()
    rec: dict = {"question": question, "query_id": query_id, "k": k,
                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "retrieval": [], "gate": None}
    try:
        hits = retriever.search(question, k)
        if not hits:
            raise ValueError("no passages returned")
    except Exception as e:  # missing index, encoder failure, empty result
        rec["response"] = GroundedAnswer(answer="", citations=[], abstained=True, query_id=query_id,
                                         abstain_reason=f"{REASON_RETRIEVAL}{type(e).__name__}: {e}",
                                         parse_ok=False).to_dict()
        rec["latency_s"] = round(time.perf_counter() - t0, 3)
        return rec
    rec["retrieval"] = [{"chunk_id": h.chunk_id, "document_id": h.document_id, "page_number": h.page_number,
                         "score": round(h.score, 4)} for h in hits]
    if gate is not None:
        f = features(question, hits)
        rec["gate"] = {"features": f, "probability": round(gate.probability(f), 4), "answerable": gate.answerable(f)}
        if not rec["gate"]["answerable"]:
            rec["response"] = GroundedAnswer(answer="", citations=[], abstained=True, query_id=query_id,
                                             abstain_reason=REASON_GATE).to_dict()
            rec["latency_s"] = round(time.perf_counter() - t0, 3)
            return rec
    query = GroundedQuery(question=question, query_id=query_id,
                          evidence=[EvidenceChunk(h.chunk_id, h.text, source=h.document_id, page=h.page_number,
                                                  score=h.score) for h in hits])
    answer = guard(adapter.answer_batch([query])[0], query)
    rec["response"] = answer.to_dict()
    rec["latency_s"] = round(time.perf_counter() - t0, 3)
    return rec


def render(rec: dict) -> str:
    r = rec["response"]
    if r["abstained"]:
        return f"ABSTAINED ({r['abstain_reason']}): the retrieved passages do not support an answer."
    where = {h["chunk_id"]: f"{h['document_id']} p.{h['page_number']}" for h in rec["retrieval"]}
    return r["answer"] + "\n" + "\n".join(f"  [{c}] {where.get(c, '')}" for c in r["citations"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ps3_llm.pipeline")
    ap.add_argument("question")
    ap.add_argument("--adapter", choices=["sglang", "mock"], default="sglang")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default=None)
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--gate", type=Path, default=None, help="retrieval gate JSON (off by default)")
    ap.add_argument("--log", type=Path, default=None, help="append the full run record (JSONL)")
    ap.add_argument("--json", action="store_true", help="print the full record instead of the rendered answer")
    args = ap.parse_args(argv)
    kwargs = {} if args.adapter == "mock" else dict(base_url=args.base_url, model=args.model, temperature=0.0,
                                                     max_tokens=384, concurrency=1)
    rec = ask(args.question, Retriever(), get_adapter(args.adapter, **kwargs), args.k,
              Gate.load(args.gate) if args.gate else None)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, indent=2, ensure_ascii=False) if args.json else render(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
