"""Stage 3 (AuditKIT) — independent re-scoring of saved eval runs with Lexsi Labs' AuditKIT.

Uses AuditKIT's `precomputed` model route so the exact outputs produced by the SGLang-served
models (results/<tag>/predictions.jsonl) are scored — no re-generation, fully reproducible.

  1. F1Score over all held-out items. An abstention is rendered as ABSTAIN_ANSWER and unanswerable
     items use ABSTAIN_ANSWER as target, so correct refusals score and hallucinated answers don't.
     Baseline vs candidate are compared with ak.compare (per-sample deltas + significance).
  2. LexicalGroundedness on answered items only, against the passages the answer CITES
     (falls back to all passages when nothing is cited).
  3. Optional (--judge api:<model>): AuditKIT's LLM-judged Hallucination metric on answered items,
     with the judge served by SGLang's OpenAI-compatible endpoint (--judge-base).

    python -m ps3_llm.audit --baseline baseline-qwen2.5-1.5b --candidate sft-qwen2.5-1.5b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import auditkit as ak

from .prompting import ABSTAIN_ANSWER

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_samples(data: Path, preds_path: Path) -> tuple[list, list]:
    gold = {r["query_id"]: r for r in _load(data)}
    preds = {p["query_id"]: p for p in _load(preds_path)}
    all_s, answered = [], []
    for qid, r in gold.items():
        p = preds[qid]
        ev = {e["chunk_id"]: e["text"] for e in r["evidence"]}
        out = ABSTAIN_ANSWER if p["abstained"] or not p["answer"] else p["answer"]
        all_s.append(ak.Sample(input=r["question"], id=qid, target=r["gold_answer"] or ABSTAIN_ANSWER,
                               actual_output=out, tags=[r["category"]]))
        if out != ABSTAIN_ANSWER and ev:
            ctx = [ev[c] for c in p["citations"] if c in ev] or list(ev.values())
            answered.append(ak.Sample(input=r["question"], id=qid, retrieval_context=ctx,
                                      actual_output=out, tags=[r["category"]]))
    return all_s, answered


def audit_run(tag: str, data: Path, results: Path, judge: str | None, judge_base: str | None) -> dict:
    all_s, answered = build_samples(data, results / tag / "predictions.jsonl")
    out_dir = results / tag / "auditkit"
    out_dir.mkdir(parents=True, exist_ok=True)
    f1_run = ak.evaluate(all_s, model="precomputed", scorers=[ak.F1Score()], experiment_name=f"{tag}-f1")
    f1_run.save(str(out_dir / "f1_run.json"))
    summary = {"tag": tag, "n": len(all_s), "n_answered": len(answered),
               "f1_score_all": f1_run.headline.get("f1_score"), "f1_fingerprint": f1_run.fingerprint}
    if answered:
        scorers = [ak.LexicalGroundedness()]
        if judge:
            scorers.append(ak.Hallucination(judge_model=judge,
                                            judge_model_args={"api_base": judge_base, "api_key": "EMPTY"}))
        g_run = ak.evaluate(answered, model="precomputed", scorers=scorers, experiment_name=f"{tag}-grounding")
        g_run.save(str(out_dir / "grounding_run.json"))
        summary.update({f"{k}_answered": v for k, v in g_run.headline.items()})
        summary["grounding_fingerprint"] = g_run.fingerprint
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "f1_run": f1_run}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="results tag of the baseline run")
    ap.add_argument("--candidate", required=True, help="results tag of the fine-tuned run")
    ap.add_argument("--data", type=Path, default=ROOT / "data/mock/heldout_eval.jsonl")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--judge", default=None, help="AuditKIT model spec for the judge, e.g. api:<served id>")
    ap.add_argument("--judge-base", default="http://127.0.0.1:30000/v1")
    args = ap.parse_args()

    base = audit_run(args.baseline, args.data, args.results_dir, args.judge, args.judge_base)
    cand = audit_run(args.candidate, args.data, args.results_dir, args.judge, args.judge_base)
    cmp = ak.compare(base["f1_run"], cand["f1_run"])
    report = {"baseline": base["summary"], "candidate": cand["summary"], "comparison": cmp.summary()}
    out = args.results_dir / f"auditkit_{args.baseline}_vs_{args.candidate}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
