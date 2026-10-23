"""Deterministic groundedness metrics for (EvalExample, GroundedAnswer) pairs.

All metrics are lexical and reproducible (no judge model), so baseline vs fine-tuned
comparisons are not confounded by judge variance. Definitions:

- correct_answer: answerable item, model answered, and the answer matches gold — every number in
  the gold answer appears in the prediction and content-token recall of gold >= 0.5.
- appropriate_abstention: unanswerable item and the model abstained.
- over_abstention: answerable item and the model abstained.
- hallucinated_answer: unanswerable item and the model answered anyway.
- citation precision/recall: against gold citation ids (answerable, answered items).
- invalid_citations: cited ids that were not in the provided evidence.
- claims: answer split into sentences/clauses. A claim is *supported* when all its numbers appear in
  the cited passages and >= 60% of its content tokens appear in the cited passages. Claims with no
  citations are unsupported by definition.
- unsupported_numbers: numbers in the answer that appear in no evidence passage at all.
- grounded_correct (headline): correct_answer with >=1 gold citation for answerable items,
  appropriate_abstention for unanswerable items.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from typing import Any

from .schemas import EvalExample, GroundedAnswer

_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_STOP = set(
    "a an the of to in on for and or is are was were be been by with what which who whom when where why how "
    "do does did can could should would will may might must this that these those it its as at from our your "
    "their there than then into about over under per any all each not no if be has have had also within least".split()
)


def _norm_num(n: str) -> str:
    return n.replace(",", "")


def numbers(text: str) -> set[str]:
    return {_norm_num(n) for n in _NUM.findall(text)}


def content_tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


def token_f1(pred: str, gold: str) -> float:
    p, g = content_tokens(pred), content_tokens(gold)
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def answer_matches(pred: str, gold: str) -> bool:
    if not pred:
        return False
    if not numbers(gold) <= numbers(pred):
        return False
    g = set(content_tokens(gold))
    if not g:
        return True
    return len(g & set(content_tokens(pred))) / len(g) >= 0.5


def split_claims(answer: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\s+(?:and|but)\s+(?=[A-Z])", answer.strip())
    return [p for p in (s.strip() for s in parts) if len(content_tokens(p)) >= 1]


def claim_supported(claim: str, cited_text: str) -> bool:
    if not cited_text:
        return False
    if not numbers(claim) <= numbers(cited_text):
        return False
    toks = set(content_tokens(claim))
    if not toks:
        return True
    return len(toks & set(content_tokens(cited_text))) / len(toks) >= 0.6


def score_example(ex: EvalExample, pred: GroundedAnswer) -> dict[str, Any]:
    ev_text = {c.chunk_id: c.text for c in ex.query.evidence}
    all_ev = " ".join(ev_text.values())
    answered = not pred.abstained and bool(pred.answer)
    r: dict[str, Any] = {
        "query_id": ex.query.query_id,
        "category": ex.category,
        "answerable": ex.answerable,
        "abstained": pred.abstained,
        "parse_ok": pred.parse_ok,
        "latency_s": pred.latency_s,
        "invalid_citations": len((pred.usage or {}).get("invalid_citations", [])),
        "completion_tokens": (pred.usage or {}).get("completion_tokens"),
    }
    if ex.answerable:
        correct = answered and answer_matches(pred.answer, ex.gold_answer)
        cited, gold = set(pred.citations), set(ex.gold_citations)
        r.update(
            correct_answer=correct,
            over_abstention=not answered,
            token_f1=token_f1(pred.answer, ex.gold_answer) if answered else 0.0,
            citation_precision=(len(cited & gold) / len(cited)) if (answered and cited) else (0.0 if answered else None),
            citation_recall=(len(cited & gold) / len(gold)) if answered else None,
            grounded_correct=correct and bool(cited & gold),
        )
    else:
        r.update(appropriate_abstention=not answered, hallucinated_answer=answered,
                 grounded_correct=not answered)

    if answered:
        cited_text = " ".join(ev_text.get(c, "") for c in pred.citations)
        claims = split_claims(pred.answer)
        sup = [claim_supported(c, cited_text) for c in claims]
        r.update(
            n_claims=len(claims),
            n_unsupported_claims=sum(not s for s in sup),
            unsupported_numbers=sorted(numbers(pred.answer) - numbers(all_ev) - numbers(ex.query.question)),
            uncited_answer=not pred.citations,
        )
    return r


def _mean(xs: list) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def _pct(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return round(xs[k], 3)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if "n_claims" in r]
    total_claims = sum(r["n_claims"] for r in answered)
    lat = [r["latency_s"] for r in rows]
    summary = {
        "n": len(rows),
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "grounded_accuracy": _mean([r["grounded_correct"] for r in rows]),
        "answer_accuracy": _mean([r["correct_answer"] for r in ans]),
        "answer_token_f1": _mean([r["token_f1"] for r in ans]),
        "over_abstention_rate": _mean([r["over_abstention"] for r in ans]),
        "appropriate_abstention_rate": _mean([r["appropriate_abstention"] for r in una]),
        "hallucinated_answer_rate": _mean([r["hallucinated_answer"] for r in una]),
        "citation_precision": _mean([r.get("citation_precision") for r in ans]),
        "citation_recall": _mean([r.get("citation_recall") for r in ans]),
        "unsupported_claim_rate": round(sum(r["n_unsupported_claims"] for r in answered) / total_claims, 4)
        if total_claims else None,
        "answers_with_unsupported_numbers": sum(bool(r["unsupported_numbers"]) for r in answered),
        "uncited_answers": sum(r["uncited_answer"] for r in answered),
        "invalid_citation_ids": sum(r["invalid_citations"] for r in rows),
        "json_parse_rate": _mean([r["parse_ok"] for r in rows]),
        "latency_p50_s": _pct(lat, 0.5),
        "latency_p95_s": _pct(lat, 0.95),
        "latency_mean_s": _mean(lat),
    }
    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["grounded_correct"])
    summary["grounded_accuracy_by_category"] = {k: _mean(v) for k, v in sorted(by_cat.items())}
    return summary


def error_buckets(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Group failing query ids by failure mode for error analysis."""
    b: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        qid = r["query_id"]
        if not r["parse_ok"]:
            b["unparseable_output"].append(qid)
        if r["answerable"]:
            if r["over_abstention"]:
                b["over_abstained"].append(qid)
            elif not r["correct_answer"]:
                b["wrong_answer"].append(qid)
            elif not r["grounded_correct"]:
                b["right_answer_wrong_citation"].append(qid)
        elif r["hallucinated_answer"]:
            b[f"answered_unanswerable:{r['category']}"].append(qid)
        if r.get("n_unsupported_claims"):
            b["has_unsupported_claims"].append(qid)
        if r.get("unsupported_numbers"):
            b["invented_numbers"].append(qid)
        if r["invalid_citations"]:
            b["cited_nonexistent_passage"].append(qid)
    return dict(b)
