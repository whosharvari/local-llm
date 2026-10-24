"""Groundedness metrics (v2), computed from raw model text + gold labels + adjudication.

v2 replaces v1 after the verification audit (VERIFICATION_REPORT.md). v1 results under results/<tag>/
were produced by the previous definitions and are kept unchanged as historical evidence.

Every item is scored under two decisions from ps3_llm.prompting.parse_model_output:
  - "serving": what the API returns (flag respected, contradictions withheld, citation required);
  - "intent":  whether the answer text is substantive, flag ignored (compares model behaviour fairly
               when models differ in how well they use the abstain flag).

Definitions (denominators in brackets; A = answerable items, U = unanswerable items, h30 = ambiguous):
  answer_correct_rate        [A]  answered and all adjudicated key facts present, no rejected phrase
  wrong_answer_rate          [A]  answered but not correct
  inappropriate_abstention   [A]  did not answer an answerable item
  correct_abstention_rate    [U]  did not answer
  hallucinated_answer_rate   [U]  answered an unanswerable item (= 1 - correct_abstention_rate)
  cites_gold_rate            [answered A]  at least one gold citation among returned citations
  all_gold_cited_rate        [answered A]  every gold citation returned
  citation_precision         [answered A]  mean share of returned citations that are gold
  grounded_accuracy          [all 40]  A: correct and cites gold; U: not answered;
                                        h30: not answered, or an accepted answer citing travel#c1
  grounded_accuracy_excl_ambiguous [39]
  unsupported_claim_rate     [claims in answered items]  claim (sentence/clause) unsupported unless its
                             numbers and >=60% of its content words occur in the passages it cites.
                             Refusal text is never an answer, so it is never counted as a claim.
  json_valid_rate / schema_valid_rate [all]  parseable JSON / JSON with the 3 keys and a boolean flag
  inference_errors           count of items with no model output (server failure)
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .prompting import parse_model_output
from .schemas import INFERENCE_ERROR_PREFIX, EvalExample, GroundedAnswer

_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_STOP = set(
    "a an the of to in on for and or is are was were be been by with what which who whom when where why how "
    "do does did can could should would will may might must this that these those it its as at from our your "
    "their there than then into about over under per any all each not no if be has have had also within least".split()
)
MODES = ("serving", "intent")


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM.findall(text)}


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


def _has(text: str, phrase: str) -> bool:
    t = text.lower().replace(",", "")
    p = phrase.lower().replace(",", "")
    return re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", t) is not None


def facts_match(answer: str, facts: list[list[str]], reject: list[str] | None = None) -> bool:
    if not answer:
        return False
    if any(_has(answer, r) for r in (reject or [])):
        return False
    return all(any(_has(answer, alt) for alt in fact) for fact in facts)


def load_adjudication(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text())["items"]


def score_example(ex: EvalExample, pred: GroundedAnswer, adj: dict[str, dict]) -> dict[str, Any]:
    qid = ex.query.query_id
    ev_text = {c.chunk_id: c.text for c in ex.query.evidence}
    a = adj.get(qid, {})
    category = a.get("category_override", ex.category)
    inference_error = pred.abstain_reason.startswith(INFERENCE_ERROR_PREFIX)
    parsed = parse_model_output(pred.raw_output, set(ev_text))
    row: dict[str, Any] = {
        "query_id": qid, "category": category, "answerable": ex.answerable and category != "ambiguous",
        "ambiguous": category == "ambiguous", "latency_s": pred.latency_s, "inference_error": inference_error,
        "json_valid": parsed["parse_ok"], "schema_valid": parsed["schema_ok"], "violations": parsed["violations"],
        "invalid_citations": len(parsed["invalid_citations"]),
        "completion_tokens": (pred.usage or {}).get("completion_tokens"),
    }
    decisions = {
        "serving": (not parsed["abstained"], parsed["answer"], parsed["citations"]),
        "intent": (parsed["intent_answered"], parsed["intent_answer"], parsed["intent_citations"]),
    }
    if inference_error:
        decisions = {m: (False, "", []) for m in MODES}
    for mode, (answered, answer, cits) in decisions.items():
        r: dict[str, Any] = {"answered": answered}
        gold = set(ex.gold_citations)
        if row["ambiguous"]:
            ok_answer = answered and facts_match(answer, a["accept_answer_facts"], a.get("reject")) \
                and bool(set(cits) & set(a.get("accept_citations", [])))
            r["grounded"] = (not answered) or ok_answer
            r["hallucinated"] = answered and not ok_answer
        elif row["answerable"]:
            correct = answered and facts_match(answer, a.get("facts", []), a.get("reject"))
            r.update(correct=correct, wrong=answered and not correct, inappropriate_abstention=not answered,
                     grounded=correct and bool(set(cits) & gold), token_f1=token_f1(answer, ex.gold_answer) if answered else 0.0)
            if answered:
                r.update(cites_gold=bool(set(cits) & gold), all_gold_cited=gold <= set(cits),
                         citation_precision=len(set(cits) & gold) / len(cits) if cits else 0.0)
        else:
            r.update(grounded=not answered, hallucinated=answered)
        if answered:
            cited = " ".join(ev_text.get(c, "") for c in cits)
            claims = split_claims(answer)
            r.update(n_claims=len(claims), n_unsupported=sum(not claim_supported(c, cited) for c in claims),
                     uncited=not cits)
        row[mode] = r
    return row


def _mean(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def _pct(xs: list[float], q: float) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return round(xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))], 3)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    A = [r for r in rows if r["answerable"]]
    U = [r for r in rows if not r["answerable"] and not r["ambiguous"]]
    out: dict[str, Any] = {
        "metrics_version": "v2",
        "n": len(rows), "n_answerable": len(A), "n_unanswerable": len(U),
        "n_ambiguous": sum(r["ambiguous"] for r in rows),
        "json_valid_rate": _mean([r["json_valid"] for r in rows]),
        "schema_valid_rate": _mean([r["schema_valid"] for r in rows]),
        "invalid_citation_ids": sum(r["invalid_citations"] for r in rows),
        "contract_violations": dict(Counter(v.split(":")[0] for r in rows for v in r["violations"])),
        "inference_errors": sum(r["inference_error"] for r in rows),
        "latency_p50_s": _pct([r["latency_s"] for r in rows], 0.5),
        "latency_p95_s": _pct([r["latency_s"] for r in rows], 0.95),
    }
    for m in MODES:
        ansA = [r[m] for r in A if r[m]["answered"]]
        answered = [r[m] for r in rows if r[m]["answered"]]
        n_claims = sum(x["n_claims"] for x in answered)
        out[m] = {
            "grounded_accuracy": _mean([r[m]["grounded"] for r in rows]),
            "grounded_accuracy_excl_ambiguous": _mean([r[m]["grounded"] for r in rows if not r["ambiguous"]]),
            "answer_correct_rate": _mean([r[m]["correct"] for r in A]),
            "wrong_answer_rate": _mean([r[m]["wrong"] for r in A]),
            "inappropriate_abstention_rate": _mean([r[m]["inappropriate_abstention"] for r in A]),
            "correct_abstention_rate": _mean([not r[m]["hallucinated"] for r in U]),
            "hallucinated_answer_rate": _mean([r[m]["hallucinated"] for r in U]),
            "ambiguous_items_grounded": [r["query_id"] for r in rows if r["ambiguous"] and r[m]["grounded"]],
            "n_answered": len(answered),
            "cites_gold_rate": _mean([x["cites_gold"] for x in ansA]),
            "all_gold_cited_rate": _mean([x["all_gold_cited"] for x in ansA]),
            "citation_precision": _mean([x["citation_precision"] for x in ansA]),
            "answer_token_f1": _mean([r[m]["token_f1"] for r in A]),
            "unsupported_claim_rate": round(sum(x["n_unsupported"] for x in answered) / n_claims, 4) if n_claims else None,
            "n_claims": n_claims,
            "uncited_answers": sum(x["uncited"] for x in answered),
        }
        by_cat: dict[str, list] = defaultdict(list)
        for r in rows:
            by_cat[r["category"]].append(r[m]["grounded"])
        out[m]["grounded_accuracy_by_category"] = {k: _mean(v) for k, v in sorted(by_cat.items())}
    return out


def error_buckets(rows: list[dict[str, Any]], mode: str = "intent") -> dict[str, list[str]]:
    """Group failing query ids by failure mode for error analysis."""
    b: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        qid, x = r["query_id"], r[mode]
        if r["inference_error"]:
            b["inference_error"].append(qid)
        if not r["json_valid"]:
            b["not_json"].append(qid)
        for v in r["violations"]:
            if v.startswith(("flag_true_but", "flag_false_but", "non_boolean_flag")):
                b["flag_misuse"].append(qid)
                break
        if r["answerable"]:
            if x["inappropriate_abstention"]:
                b["inappropriate_abstention"].append(qid)
            elif x["wrong"]:
                b["wrong_answer"].append(qid)
            elif not x["grounded"]:
                b["right_answer_wrong_citation"].append(qid)
        elif not x["grounded"]:
            b[f"answered_unanswerable:{r['category']}"].append(qid)
        if x.get("n_unsupported"):
            b["has_unsupported_claims"].append(qid)
        if r["invalid_citations"]:
            b["cited_nonexistent_passage"].append(qid)
    return dict(b)
