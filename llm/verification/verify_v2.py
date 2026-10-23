"""Independent re-implementation of metrics v2 from the written policy (does NOT import ps3_llm).

Reads raw model text from results/<src>/predictions.jsonl, gold labels from data/mock/heldout_eval.jsonl
and correctness criteria from data/mock/heldout_adjudication.json; recomputes the headline v2 metrics
under both decisions and compares them with results/v2/<src>/summary.json.

    python3 -I verification/verify_v2.py . baseline-qwen2.5-1.5b sft-qwen2.5-1.5b
"""
import json
import re
import sys
from pathlib import Path

REFUSAL_WORD = re.compile(r"^\W*(abstain(ed|ing)?|refuse[ds]?|no answer)\W*$", re.I)
REFUSAL_PHRASES = ["do not contain", "does not contain", "doesn't contain", "don't contain", "not enough information",
                   "insufficient information", "insufficient evidence", "cannot be answered", "can't be answered",
                   "cannot be determined", "can't be determined", "no information", "no relevant information",
                   "not mentioned", "not explicitly mentioned", "not provided in", "not specified", "not stated",
                   "unable to answer", "cannot answer", "can't answer", "not found in the"]


def refusal(t):
    t = t.strip().lower()
    return bool(REFUSAL_WORD.match(t)) or any(p in t for p in REFUSAL_PHRASES)


def flag_of(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        v = v.strip().lower()
        return True if v in ("true", "yes", "1", "abstain", "abstained") else False if v in ("false", "no", "0") else None
    return None


def decide(raw, ids):
    s = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    m = re.search(r"\{.*\}", s, re.S)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
            obj = obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            obj = None
    if obj is None:
        ans = re.sub(r"\n?\s*citations\s*:.*$", "", s, flags=re.I | re.S).strip()
        cits, flag = re.findall(r"\[([^\[\]]+)\]", s), None
    else:
        ans, flag = str(obj.get("answer") or "").strip(), flag_of(obj.get("abstain"))
        cits = obj.get("citations") or []
        cits = [cits] if isinstance(cits, str) else cits
    cits = list(dict.fromkeys(c for c in (re.sub(r"^\[(.*)\]$", r"\1", str(c).strip()).strip() for c in cits) if c in ids))
    subst = bool(ans) and not refusal(ans)
    serving_answer = obj is not None and subst and flag is not True and bool(cits)
    return {"serving": (serving_answer, ans if serving_answer else "", cits if serving_answer else []),
            "intent": (subst, ans if subst else "", cits if subst else []),
            "json": obj is not None}


def has(text, phrase):
    t, p = text.lower().replace(",", ""), phrase.lower().replace(",", "")
    return re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", t) is not None


def ok(ans, facts, reject=()):
    return bool(ans) and not any(has(ans, r) for r in reject) and all(any(has(ans, a) for a in f) for f in facts)


def main(llm, tags):
    gold = [json.loads(l) for l in (llm / "data/mock/heldout_eval.jsonl").read_text().splitlines()]
    adj = json.loads((llm / "data/mock/heldout_adjudication.json").read_text())["items"]
    bad = 0
    for tag in tags:
        P = {p["query_id"]: p for p in map(json.loads, (llm / "results" / tag / "predictions.jsonl").read_text().splitlines())}
        rep = json.loads((llm / "results/v2" / tag / "summary.json").read_text())
        print(f"== {tag}")
        for mode in ("serving", "intent"):
            g, corr, inap, hall, nA, nU = 0, 0, 0, 0, 0, 0
            for it in gold:
                q, a = it["query_id"], adj.get(it["query_id"], {})
                answered, ans, cits = decide(P[q]["raw_output"], {e["chunk_id"] for e in it["evidence"]})[mode]
                if a.get("category_override") == "ambiguous":
                    g += (not answered) or (ok(ans, a["accept_answer_facts"], a.get("reject", ())) and bool(set(cits) & set(a["accept_citations"])))
                elif it["answerable"]:
                    nA += 1
                    c = answered and ok(ans, a["facts"], a.get("reject", ()))
                    corr += c
                    inap += not answered
                    g += c and bool(set(cits) & set(it["gold_citations"]))
                else:
                    nU += 1
                    hall += answered
                    g += not answered
            mine = {"grounded_accuracy": round(g / len(gold), 4), "answer_correct_rate": round(corr / nA, 4),
                    "inappropriate_abstention_rate": round(inap / nA, 4), "hallucinated_answer_rate": round(hall / nU, 4)}
            for k, v in mine.items():
                r = rep[mode][k]
                flag = "" if abs(v - r) < 1e-4 else "   <-- MISMATCH"
                bad += bool(flag)
                print(f"  {mode:7s} {k:30s} independent={v:<7} reported={r:<7}{flag}")
        jv = sum(decide(P[q]["raw_output"], set())["json"] for q in P) / len(P)
        print(f"  json_valid_rate independent={jv} reported={rep['json_valid_rate']}")
        bad += abs(jv - rep["json_valid_rate"]) > 1e-4
    print("MISMATCHES:", bad)
    return bad


if __name__ == "__main__":
    sys.exit(1 if main(Path(sys.argv[1]), sys.argv[2:]) else 0)
