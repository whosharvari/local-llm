"""Side-by-side comparison of two eval runs on the same held-out set.

    python -m ps3_llm.compare baseline-qwen2.5-1.5b sft-qwen2.5-1.5b
-> results/compare_<a>_vs_<b>.md (metric table, per-category table, per-example flips, error buckets)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

HEADLINE = [
    ("grounded_accuracy", "Grounded accuracy (all 40)", True),
    ("answer_accuracy", "Answer accuracy (answerable)", True),
    ("answer_token_f1", "Answer token F1 (answerable)", True),
    ("appropriate_abstention_rate", "Appropriate abstention (unanswerable)", True),
    ("hallucinated_answer_rate", "Hallucinated answer rate (unanswerable)", False),
    ("over_abstention_rate", "Over-abstention rate (answerable)", False),
    ("unsupported_claim_rate", "Unsupported claim rate (answered)", False),
    ("answers_with_unsupported_numbers", "Answers with invented numbers", False),
    ("uncited_answers", "Answers without citations", False),
    ("invalid_citation_ids", "Cited non-existent passage ids", False),
    ("citation_precision", "Citation precision", True),
    ("citation_recall", "Citation recall", True),
    ("json_parse_rate", "Valid JSON output rate", True),
    ("latency_p50_s", "Latency p50 (s)", False),
    ("latency_p95_s", "Latency p95 (s)", False),
    ("mean_completion_tokens", "Mean completion tokens", False),
    ("throughput_qps", "Throughput (queries/s, concurrency 4)", True),
    ("sglang_server_mem_gb", "SGLang scheduler memory (GB)", False),
]


def _load(tag: str):
    d = RESULTS / tag
    return (json.loads((d / "summary.json").read_text()),
            {r["query_id"]: r for r in map(json.loads, (d / "per_example.jsonl").read_text().splitlines())},
            {p["query_id"]: p for p in map(json.loads, (d / "predictions.jsonl").read_text().splitlines())},
            json.loads((d / "errors.json").read_text()))


def _fmt(v):
    return "–" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main(a: str, b: str) -> Path:
    sa, ra, pa, ea = _load(a)
    sb, rb, pb, eb = _load(b)
    questions = {json.loads(line)["query_id"]: json.loads(line)["question"]
                 for line in (ROOT / "data/mock/heldout_eval.jsonl").read_text().splitlines()}
    lines = [f"# {a} vs {b}", "", f"Same held-out set (n={sa['n']}), same metrics, temperature 0.", "",
             f"| Metric | {a} | {b} | Better |", "|---|---|---|---|"]
    for key, label, higher in HEADLINE:
        va, vb = sa.get(key), sb.get(key)
        better = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != vb:
            better = b if (vb > va) == higher else a
        lines.append(f"| {label} | {_fmt(va)} | {_fmt(vb)} | {better} |")

    lines += ["", "## Grounded accuracy by category", "", f"| Category | {a} | {b} |", "|---|---|---|"]
    for cat in sa["grounded_accuracy_by_category"]:
        lines.append(f"| {cat} | {_fmt(sa['grounded_accuracy_by_category'][cat])} | "
                     f"{_fmt(sb['grounded_accuracy_by_category'].get(cat))} |")

    fixed = [q for q in ra if not ra[q]["grounded_correct"] and rb[q]["grounded_correct"]]
    broke = [q for q in ra if ra[q]["grounded_correct"] and not rb[q]["grounded_correct"]]
    still = [q for q in ra if not ra[q]["grounded_correct"] and not rb[q]["grounded_correct"]]
    lines += ["", f"## Per-example changes ({len(fixed)} fixed, {len(broke)} regressed, {len(still)} wrong in both)", ""]

    def show(p):
        return "ABSTAIN" if p["abstained"] else f"{p['answer']} {p['citations']}"

    for title, ids in (("Fixed", fixed), ("Regressed", broke), ("Wrong in both", still)):
        lines += [f"### {title}", "", f"| id | category | question | {a} | {b} |", "|---|---|---|---|---|"]
        for q in ids:
            lines.append(f"| {q} | {ra[q]['category']} | {questions[q]} | {show(pa[q])} | {show(pb[q])} |".replace("\n", " "))
        lines.append("")

    lines += ["## Error buckets", "", f"| Bucket | {a} | {b} |", "|---|---|---|"]
    for k in sorted(set(ea) | set(eb)):
        lines.append(f"| {k} | {', '.join(ea.get(k, [])) or '–'} | {', '.join(eb.get(k, [])) or '–'} |")

    out = RESULTS / f"compare_{a}_vs_{b}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
