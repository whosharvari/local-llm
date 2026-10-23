"""Intent-based scoring: an item is 'answered' iff the answer TEXT is substantive (ignores the abstain flag).
Independent of ps3_llm; reuses only extract/correct/claims/supported from verify_metrics.py (also independent)."""
import json, re, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from verify_metrics import extract, correct, claims, supported, toks

REFUSAL = re.compile(r"^\W*abstain\W*$|not (?:explicitly )?(?:specified|mentioned|stated|provided)|do(?:es)? not contain|"
                     r"no information|not enough information|cannot answer|unable to answer|insufficient", re.I)

def intent(raw, ev_ids):
    o = extract(raw)
    if o is None:  # non-JSON: take whole text
        text = re.sub(r"citations:.*", "", raw, flags=re.S).strip()
        cits = re.findall(r"\[([^\[\]]+)\]", raw)
    else:
        text = str(o.get("answer") or "").strip()
        c = o.get("citations") or []
        cits = [c] if isinstance(c, str) else c
    cits = [str(x).strip().strip("[]") for x in cits]
    answered = bool(text) and not REFUSAL.search(text)
    return answered, text, [x for x in cits if x in ev_ids]

def run(llm, tag, manual_overrides=None):
    gold = [json.loads(l) for l in (llm / "data/mock/heldout_eval.jsonl").read_text().splitlines()]
    P = {p["query_id"]: p for p in map(json.loads, (llm / "results" / tag / "predictions.jsonl").read_text().splitlines())}
    out = {"answered_unanswerable": [], "abstained_answerable": [], "grounded": [], "correct": [], "claims": 0, "unsup": 0}
    for g in gold:
        ev = {e["chunk_id"]: e["text"] for e in g["evidence"]}
        a, text, cits = intent(P[g["query_id"]]["raw_output"], set(ev))
        if a:
            cs = claims(text); cited = " ".join(ev[c] for c in cits)
            out["claims"] += len(cs); out["unsup"] += sum(not supported(c, cited) for c in cs)
        if g["answerable"]:
            ok = a and correct(text, g["gold_answer"])
            if manual_overrides and g["query_id"] in manual_overrides: ok = a and manual_overrides[g["query_id"]]
            if ok: out["correct"].append(g["query_id"])
            if ok and set(cits) & set(g["gold_citations"]): out["grounded"].append(g["query_id"])
            if not a: out["abstained_answerable"].append(g["query_id"])
        else:
            if a: out["answered_unanswerable"].append(g["query_id"])
            else: out["grounded"].append(g["query_id"])
    return {"grounded_accuracy": len(out["grounded"]) / 40, "answer_accuracy": round(len(out["correct"]) / 23, 4),
            "hallucinated_answer_rate": round(len(out["answered_unanswerable"]) / 17, 4),
            "over_abstention_rate": round(len(out["abstained_answerable"]) / 23, 4),
            "unsupported_claim_rate": round(out["unsup"] / out["claims"], 4) if out["claims"] else None,
            "answered_unanswerable": out["answered_unanswerable"], "abstained_answerable": out["abstained_answerable"]}

if __name__ == "__main__":
    llm = Path(sys.argv[1])
    # manual correctness overrides from hand review (only where the lexical rule disagrees with a human reading)
    # h17: terse correct answer the lexical rule rejects; h14: "Premium economy" wrongly accepted; h20: bare "No" omits the exception
    manual = {"h17": True, "h14": False, "h20": False}
    for tag in sys.argv[2:]:
        print("==", tag, "| lexical rule only:", json.dumps(run(llm, tag)))
        print("==", tag, "| + manual overrides (report 4c):", json.dumps(run(llm, tag, manual)))
