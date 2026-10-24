"""Independent recomputation of PS3 eval metrics from raw model text + gold labels.

Deliberately does NOT import ps3_llm. Own JSON extraction, own tokenisation, own aggregation.
Usage: python verify_metrics.py <llm_dir> <tag> [<tag> ...]
"""
import json
import re
import statistics
import sys
from pathlib import Path

STOP = set("""a an the of to in on for and or is are was were be been by with what which who whom when where why
how do does did can could should would will may might must this that these those it its as at from our your their
there than then into about over under per any all each not no if has have had also within least""".split())


def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", s.lower()) if t not in STOP]


def nums(s):
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", s)}


def extract(raw):
    """Return (obj or None). Takes the substring from first '{' to last '}'."""
    s = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        o = json.loads(s[i:j + 1])
        return o if isinstance(o, dict) else None
    except json.JSONDecodeError:
        return None


def decide(raw, ev_ids):
    o = extract(raw)
    if o is None:
        return dict(valid_json=False, schema_ok=False, answered=None, answer="", cits=[], bad_cits=[])
    flag = o.get("abstain")
    flag_true = flag is True or (isinstance(flag, str) and flag.strip().lower() == "true")
    ans = str(o.get("answer") or "").strip()
    cits = o.get("citations") or []
    cits = [cits] if isinstance(cits, str) else cits
    cits = [str(c).strip().strip("[]") for c in cits]
    answered = (not flag_true) and bool(ans)
    return dict(valid_json=True, schema_ok={"abstain", "answer", "citations"} <= set(o), answered=answered,
                answer=ans if answered else "", cits=[c for c in cits if c in ev_ids] if answered else [],
                bad_cits=[c for c in cits if c not in ev_ids])


def correct(pred, gold):
    if not nums(gold) <= nums(pred):
        return False
    g = set(toks(gold))
    return not g or len(g & set(toks(pred))) / len(g) >= 0.5


def claims(ans):
    return [c for c in re.split(r"(?<=[.!?;])\s+|\s+(?:and|but)\s+(?=[A-Z])", ans) if toks(c)]


def supported(c, text):
    t = set(toks(c))
    return bool(text) and nums(c) <= nums(text) and (not t or len(t & set(toks(text))) / len(t) >= 0.6)


def pct(xs, q):  # nearest-rank on sorted list, also report statistics.quantiles for comparison
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def run(llm, tag):
    gold = [json.loads(l) for l in (llm / "data/mock/heldout_eval.jsonl").read_text().splitlines()]
    preds = {p["query_id"]: p for p in map(json.loads, (llm / "results" / tag / "predictions.jsonl").read_text().splitlines())}
    assert len(preds) == len(gold) == 40 and set(preds) == {g["query_id"] for g in gold}
    A, U = [g for g in gold if g["answerable"]], [g for g in gold if not g["answerable"]]
    rec, n_claims, n_unsup, inval_json, items = {}, 0, 0, 0, {}
    for g in gold:
        ev = {e["chunk_id"]: e["text"] for e in g["evidence"]}
        d = decide(preds[g["query_id"]]["raw_output"], set(ev))
        # an unparseable output: treat as answered iff it does not look like a refusal (same policy as audited code)
        if d["answered"] is None:
            inval_json += 1
            low = preds[g["query_id"]]["raw_output"].lower()
            d["answered"] = not any(p in low for p in ("do not contain", "does not contain", "not enough information",
                                                        "no information", "not mentioned", "not provided", "not specified",
                                                        "cannot answer", "unable to answer", "insufficient", "not stated"))
            d["answer"] = preds[g["query_id"]]["raw_output"] if d["answered"] else ""
            d["cits"] = [c for c in re.findall(r"\[([^\[\]]+)\]", d["answer"]) if c in ev]
        items[g["query_id"]] = d
        if d["answered"]:
            cited = " ".join(ev[c] for c in d["cits"])
            cs = claims(d["answer"])
            n_claims += len(cs)
            n_unsup += sum(not supported(c, cited) for c in cs)
        if g["answerable"]:
            ok = d["answered"] and correct(d["answer"], g["gold_answer"])
            rec[g["query_id"]] = dict(correct=ok, grounded=ok and bool(set(d["cits"]) & set(g["gold_citations"])),
                                      over=not d["answered"],
                                      cp=(len(set(d["cits"]) & set(g["gold_citations"])) / len(d["cits"]) if d["cits"] else 0.0) if d["answered"] else None,
                                      cr=(len(set(d["cits"]) & set(g["gold_citations"])) / len(g["gold_citations"])) if d["answered"] else None)
        else:
            rec[g["query_id"]] = dict(grounded=not d["answered"], halluc=d["answered"])
    lat = [preds[q]["latency_s"] for q in preds]
    mean = lambda xs: round(statistics.fmean(xs), 4) if xs else None
    return {
        "grounded_accuracy": mean([r["grounded"] for r in rec.values()]),
        "answer_accuracy": mean([rec[g["query_id"]]["correct"] for g in A]),
        "appropriate_abstention_rate": mean([not rec[g["query_id"]]["halluc"] for g in U]),
        "hallucinated_answer_rate": mean([rec[g["query_id"]]["halluc"] for g in U]),
        "over_abstention_rate": mean([rec[g["query_id"]]["over"] for g in A]),
        "citation_precision": mean([rec[g["query_id"]]["cp"] for g in A if rec[g["query_id"]]["cp"] is not None]),
        "citation_recall": mean([rec[g["query_id"]]["cr"] for g in A if rec[g["query_id"]]["cr"] is not None]),
        "unsupported_claim_rate": round(n_unsup / n_claims, 4) if n_claims else None,
        "n_answered": sum(d["answered"] for d in items.values()),
        "json_parse_rate": round(1 - inval_json / 40, 4),
        "schema_ok_rate": mean([d["schema_ok"] for d in items.values()]),
        "invalid_citation_ids": sum(len(d["bad_cits"]) for d in items.values()),
        "latency_p50_s": round(pct(lat, 0.5), 3), "latency_p95_s": round(pct(lat, 0.95), 3),
        "latency_median_exact": round(statistics.median(lat), 3),
        "_items": items, "_rec": rec,
    }


if __name__ == "__main__":
    llm = Path(sys.argv[1])
    out = {}
    for tag in sys.argv[2:]:
        r = run(llm, tag)
        rep = json.loads((llm / "results" / tag / "summary.json").read_text())
        print(f"\n== {tag}")
        for k, v in r.items():
            if k.startswith("_"):
                continue
            rv = rep.get(k)
            flag = "" if rv is None or v is None or abs(float(v) - float(rv)) < 1e-3 else "   <-- MISMATCH"
            print(f"  {k:30s} independent={v!s:10s} reported={rv!s:10s}{flag}")
        out[tag] = {k: v for k, v in r.items() if not k.startswith("_")}
        out[tag]["grounded_ids"] = sorted(q for q, x in r["_rec"].items() if x["grounded"])
    json.dump(out, open(Path(__file__).with_name("verify_out.json"), "w"), indent=1)
