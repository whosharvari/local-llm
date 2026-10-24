"""Side-by-side comparison of two v2-scored runs on the same held-out set.

    python -m ps3_llm.compare v2/baseline-qwen2.5-1.5b v2/sft-qwen2.5-1.5b [--out results/v2/compare.md]
    # runs written elsewhere, e.g. by `evaluate --results-dir /tmp/res`; the report defaults to that directory too
    python -m ps3_llm.compare my-baseline my-sft --results-dir /tmp/res

Reports serving and intent decisions separately (see ps3_llm.metrics), per-category grounded accuracy,
item-level flips with an exact two-sided McNemar (binomial) test, and error buckets. No significance is
claimed by this tool; it only reports the test so readers can see how little 40 items can show.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

ROWS = [
    ("grounded_accuracy", "Grounded accuracy (all items)", True),
    ("grounded_accuracy_excl_ambiguous", "Grounded accuracy excl. ambiguous h30", True),
    ("answer_correct_rate", "Answer correct (answerable)", True),
    ("wrong_answer_rate", "Wrong answer (answerable)", False),
    ("inappropriate_abstention_rate", "Inappropriate abstention (answerable)", False),
    ("correct_abstention_rate", "Correct abstention (unanswerable)", True),
    ("hallucinated_answer_rate", "Hallucinated answer (unanswerable)", False),
    ("cites_gold_rate", "Cites >=1 gold passage (answered answerable)", True),
    ("all_gold_cited_rate", "Cites all gold passages (answered answerable)", True),
    ("citation_precision", "Citation precision (answered answerable)", True),
    ("unsupported_claim_rate", "Unsupported claim rate (claims in answers)", False),
    ("uncited_answers", "Answers with no valid citation (count)", False),
    ("answer_token_f1", "Answer token F1 vs gold (answerable)", True),
]
TOP = [
    ("json_valid_rate", "Valid JSON", True),
    ("schema_valid_rate", "Schema-valid JSON (3 keys, boolean flag)", True),
    ("invalid_citation_ids", "Invented citation ids (count)", False),
    ("inference_errors", "Inference errors (count)", False),
    ("latency_p50_s", "Latency p50 (s), single run", False),
    ("latency_p95_s", "Latency p95 (s), single run", False),
    ("sglang_server_mem_gb", "SGLang scheduler memory (GB), single reading", False),
]


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


RUN_FILES = ("summary.json", "per_example.jsonl", "predictions.jsonl", "errors.json")


def _load(tag: str, results: Path = RESULTS):
    d = results / tag
    missing = [f for f in RUN_FILES if not (d / f).is_file()]
    if missing:
        raise SystemExit(f"run {tag!r} in {results}: missing {', '.join(missing)} (expected an evaluate output dir)")
    try:
        loaded = (json.loads((d / "summary.json").read_text()),
                  {r["query_id"]: r for r in map(json.loads, (d / "per_example.jsonl").read_text().splitlines())},
                  {p["query_id"]: p for p in map(json.loads, (d / "predictions.jsonl").read_text().splitlines())},
                  json.loads((d / "errors.json").read_text()))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise SystemExit(f"run {tag!r} in {results}: unreadable result files ({type(e).__name__}: {e})")
    if not all(m in loaded[0] for m in ("serving", "intent")):
        raise SystemExit(f"run {tag!r} in {results}: summary.json is not v2-scored; re-score it with "
                         "`ps3_llm.evaluate --rescore-from`")
    return loaded


def _fmt(v):
    return "–" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def _better(va, vb, higher, a, b):
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != vb:
        return b if (vb > va) == higher else a
    return "="


def _run_data(tag: str, results: Path) -> tuple[Path, str | None]:
    """Eval data file and its SHA-256 recorded by `evaluate` (older runs: the synthetic held-out set)."""
    info_path = results / tag / "run_info.json"
    info = json.loads(info_path.read_text()) if info_path.is_file() else {}
    data = Path(info.get("data", "data/mock/heldout_eval.jsonl"))
    return (data if data.is_absolute() else ROOT / data), info.get("data_sha256")


def main(a: str, b: str, out: Path | None = None, results_dir: Path = RESULTS) -> Path:
    sa, ra, pa, ea = _load(a, results_dir)
    sb, rb, pb, eb = _load(b, results_dir)
    if sa["n"] != sb["n"] or set(ra) != set(rb):
        raise SystemExit(f"runs {a!r} and {b!r} are not on the same items")
    (data_a, sha_a), (_, sha_b) = _run_data(a, results_dir), _run_data(b, results_dir)
    if sha_a and sha_b and sha_a != sha_b:
        raise SystemExit(f"runs {a!r} and {b!r} were scored on different data files (SHA-256 differs)")
    qs = {json.loads(line)["query_id"]: json.loads(line)["question"]
          for line in data_a.read_text().splitlines() if line.strip()}
    na, nb = a.split("/")[-1], b.split("/")[-1]
    L = [f"# {na} vs {nb} (metrics {sa.get('metrics_version')})", "",
         f"n = {sa['n']} ({sa['n_answerable']} answerable, {sa['n_unanswerable']} unanswerable, "
         f"{sa['n_ambiguous']} ambiguous). One item = {100 / sa['n']:.1f} points of grounded accuracy. "
         "Small-sample evidence only; no statistical significance is claimed.", "",
         "**serving** = what the API returns (flag respected, contradictions withheld, citation required). "
         "**intent** = whether the answer text is substantive, flag ignored.", "",
         f"| Metric | {na} serving | {nb} serving | {na} intent | {nb} intent | Better (intent) |", "|---|---|---|---|---|---|"]
    for k, label, hi in ROWS:
        L.append(f"| {label} | {_fmt(sa['serving'][k])} | {_fmt(sb['serving'][k])} | {_fmt(sa['intent'][k])} | "
                 f"{_fmt(sb['intent'][k])} | {_better(sa['intent'][k], sb['intent'][k], hi, na, nb)} |")
    L += ["", f"| Output / runtime | {na} | {nb} | Better |", "|---|---|---|---|"]
    for k, label, hi in TOP:
        L.append(f"| {label} | {_fmt(sa.get(k))} | {_fmt(sb.get(k))} | {_better(sa.get(k), sb.get(k), hi, na, nb)} |")
    L.append(f"| Contract violations | {sa['contract_violations']} | {sb['contract_violations']} | |")

    for mode in ("intent", "serving"):
        fixed = sorted(q for q in ra if not ra[q][mode]["grounded"] and rb[q][mode]["grounded"])
        broke = sorted(q for q in ra if ra[q][mode]["grounded"] and not rb[q][mode]["grounded"])
        both = sorted(q for q in ra if not ra[q][mode]["grounded"] and not rb[q][mode]["grounded"])
        L += ["", f"## Item flips — {mode}", "",
              f"{len(fixed)} improved, {len(broke)} regressed, {len(both)} wrong in both; "
              f"exact McNemar p = {mcnemar_p(len(fixed), len(broke)):.3f}", ""]
        if mode == "intent":
            L += [f"| id | category | question | {na} | {nb} |", "|---|---|---|---|---|"]
            for title, ids in (("improved", fixed), ("regressed", broke), ("wrong in both", both)):
                for q in ids:
                    def show(r, p):
                        cell = f"{'ANSWER' if r[mode]['answered'] else 'ABSTAIN'}: {p['raw_output'][:110]}"
                        return cell.replace("\n", " ").replace("|", "\\|")
                    L.append(f"| {q} ({title}) | {ra[q]['category']} | {qs[q]} | {show(ra[q], pa[q])} | {show(rb[q], pb[q])} |")
    L += ["", "## Grounded accuracy by category", "", f"| Category | {na} serving | {nb} serving | {na} intent | {nb} intent |",
          "|---|---|---|---|---|"]
    for cat in sa["intent"]["grounded_accuracy_by_category"]:
        L.append(f"| {cat} | " + " | ".join(_fmt(s[m]["grounded_accuracy_by_category"].get(cat))
                                            for m in ("serving", "intent") for s in (sa, sb)).replace(" | ", " | ") + " |")
    L += ["", "## Error buckets (intent)", "", f"| Bucket | {na} | {nb} |", "|---|---|---|"]
    for k in sorted(set(ea["intent"]) | set(eb["intent"])):
        L.append(f"| {k} | {', '.join(ea['intent'].get(k, [])) or '–'} | {', '.join(eb['intent'].get(k, [])) or '–'} |")
    out = out or results_dir / f"compare_{na}_vs_{nb}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--out", type=Path, default=None, help="report path (default: <results-dir>/compare_<a>_vs_<b>.md)")
    ap.add_argument("--results-dir", type=Path, default=RESULTS, help="directory holding the run tags")
    args = ap.parse_args()
    main(args.baseline, args.candidate, args.out, args.results_dir)
