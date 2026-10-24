"""Stage 3 (AuditKIT) — independent re-scoring of saved eval runs with Lexsi Labs' AuditKIT.

Uses AuditKIT's `precomputed` model route: the exact outputs produced by the SGLang-served models
(results/<tag>/predictions.jsonl) are scored; AuditKIT does not call the models itself.

Decision modes (--mode): which answer counts as "given" for each item
  intent  (default) answer text if substantive, flag ignored (same rule as metrics v2 "intent")
  serving the API's serving-policy decision (re-derived from raw text by the current parser)
  legacy  the decision saved in predictions.jsonl (v1 behaviour; how results/*/auditkit were made)

What each AuditKIT metric measures and excludes:
  1. F1Score over ALL items. A non-answer is rendered as ABSTAIN_ANSWER; unanswerable items use
     ABSTAIN_ANSWER as target, so correct refusals score ~1 and answers to unanswerable items ~0.
     Token-overlap based: rewards answers worded like the (short) gold answers.
  2. LexicalGroundedness over ANSWERED items that have evidence: share of answer tokens found in the
     passages the answer cites (all passages if none cited). Answered items with no evidence are
     excluded and counted in n_answered_without_evidence_excluded.
  3. Optional (--judge api:<model>): LLM-judged Hallucination (fraction of extracted claims not supported
     by the context) over the same answered items; AuditKIT additionally skips outputs from which the
     judge extracts no claim. Judge served by SGLang (--judge-base).

    python -m ps3_llm.audit --baseline v2/baseline-qwen2.5-1.5b --candidate v2/sft-qwen2.5-1.5b --mode intent
Writes results/<tag>/auditkit-<mode>/ (refuses to overwrite) and results/auditkit_<a>_vs_<b>_<mode>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import auditkit as ak

from .prompting import ABSTAIN_ANSWER, parse_model_output
from .schemas import INFERENCE_ERROR_PREFIX

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _decision(p: dict, ev_ids: set[str], mode: str) -> tuple[str, list[str]]:
    """(answer or "", citations) for a saved prediction under the chosen decision mode."""
    if mode == "legacy":                      # v1 behaviour: trust the saved serving decision as-is
        return ("" if p["abstained"] else p["answer"]), p["citations"]
    if p.get("abstain_reason", "").startswith(INFERENCE_ERROR_PREFIX):
        return "", []
    parsed = parse_model_output(p["raw_output"], ev_ids)
    if mode == "intent":
        return parsed["intent_answer"], parsed["intent_citations"]
    return parsed["answer"], parsed["citations"]


def build_samples(data: Path, preds_path: Path, mode: str) -> tuple[list, list, int]:
    gold = {r["query_id"]: r for r in _load(data)}
    preds = {p["query_id"]: p for p in _load(preds_path)}
    all_s, answered, no_ev = [], [], 0
    for qid, r in gold.items():
        ev = {e["chunk_id"]: e["text"] for e in r["evidence"]}
        answer, cits = _decision(preds[qid], set(ev), mode)
        out = answer or ABSTAIN_ANSWER
        all_s.append(ak.Sample(input=r["question"], id=qid, target=r["gold_answer"] or ABSTAIN_ANSWER,
                               actual_output=out, tags=[r["category"]]))
        if answer and not ev:
            no_ev += 1            # answered with no evidence at all: AuditKIT context metrics cannot score it
        elif answer:
            ctx = [ev[c] for c in cits if c in ev] or list(ev.values())
            answered.append(ak.Sample(input=r["question"], id=qid, retrieval_context=ctx,
                                      actual_output=out, tags=[r["category"]]))
    return all_s, answered, no_ev


def audit_run(tag: str, data: Path, results: Path, judge: str | None, judge_base: str | None,
              mode: str = "intent") -> dict:
    all_s, answered, no_ev = build_samples(data, results / tag / "predictions.jsonl", mode)
    out_dir = results / tag / ("auditkit" if mode == "legacy" else f"auditkit-{mode}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite existing AuditKIT results in {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    f1_run = ak.evaluate(all_s, model="precomputed", scorers=[ak.F1Score()], experiment_name=f"{tag}-f1")
    f1_run.save(str(out_dir / "f1_run.json"))
    summary = {"tag": tag, "mode": mode, "n": len(all_s), "n_answered_scored": len(answered),
               "n_answered_without_evidence_excluded": no_ev,
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
    ap.add_argument("--mode", choices=["intent", "serving", "legacy"], default="intent",
                    help="which decision to score: intent (flag-agnostic), serving (API policy), legacy (v1 saved decision)")
    args = ap.parse_args()

    base = audit_run(args.baseline, args.data, args.results_dir, args.judge, args.judge_base, args.mode)
    cand = audit_run(args.candidate, args.data, args.results_dir, args.judge, args.judge_base, args.mode)
    cmp = ak.compare(base["f1_run"], cand["f1_run"])
    report = {"mode": args.mode, "judge": args.judge, "baseline": base["summary"], "candidate": cand["summary"],
              "comparison": cmp.summary()}
    name = lambda t: t.replace("/", "_")
    suffix = "" if args.mode == "legacy" else f"_{args.mode}"
    out = args.results_dir / f"auditkit_{name(args.baseline)}_vs_{name(args.candidate)}{suffix}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
