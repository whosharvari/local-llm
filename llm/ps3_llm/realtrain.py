"""Clean real-paper training data: leakage-filtered seeds, CuratorKIT curation and grounded SFT examples.

Only documents in the `train` split of data/real_corpus/split_manifest.json may contribute (the dev/test
papers are evaluation-only). Everything is written to a versioned directory, data/real_train/<version>/,
with a manifest of source hashes, counts and exclusion reasons. The original grounded_qa files (Member 1)
are read, never modified.

    python -m ps3_llm.realtrain clean                       # grounded_qa -> leakage-free subset + removed.jsonl
    python -m ps3_llm.realtrain chunks                      # select training chunks (deterministic)
    python -m ps3_llm.realtrain curate --llm-model openai/<teacher> --api-base URL \
        --judge-model openai/<judge> --judge-base URL       # CuratorKIT qa + HallucinationGate (+ strict gate)
    python -m ps3_llm.realtrain judge --judge-model openai/<independent judge> --judge-base URL
                                                            # second CuratorKIT HallucinationGate pass, other model family
    python -m ps3_llm.realtrain build                       # SFT train/validation files + manifest + leakage check

On a 16 GB Apple M4 the teacher and an independent judge cannot be resident in Metal memory at the same
time, so the HallucinationGate runs twice: once inside the generation run (judge = teacher, as in Stage 1)
and once as a gate-only CuratorKIT run over the accepted pairs with a judge from a different model family.
A pair is used only if it passes both gates and the deterministic strict-grounding gate.

Training example kinds (all in the exact serving prompt format, ps3_llm.prompting):
  answerable                 gold chunk + 3 same-paper distractors ranked by similarity to the question
  answerable_counterfactual  as answerable, but one number in the gold chunk and answer is changed
                             consistently, so the answer can only come from the evidence (not memory)
  abstain_hard_negative      same question, gold chunk and its overlapping neighbours removed; the 4 most
                             similar same-paper chunks that do not support the answer
  abstain_context_swap       same question, 4 passages from a different training paper
  abstain_no_evidence        same question, no passages
  abstain_out_of_corpus      Member 1's verified out-of-corpus train-split questions with their retrieved
                             contexts (only rows whose passages are all from training papers)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from . import real_corpus as rc
from .curate import run_curatorkit, strict_ground
from .metrics import claim_supported, numbers
from .prompting import SYSTEM_PROMPT, format_target, render_user_prompt
from .schemas import EvidenceChunk, GroundedQuery

ROOT = Path(__file__).resolve().parent.parent
VERSION = "v1"
OUT = ROOT / "data/real_train" / VERSION
SEED = 20261025
VALIDATION_DOC = "attention_is_all_you_need.pdf"   # train-split paper held out for validation loss only
CHUNKS_PER_DOC = 14
VALIDATION_CHUNKS = 10
K = 4
_REF = re.compile(r"arXiv|Proceedings|In Advances|Conference on|Journal of|\bpp\.|\bet al\.,? \d{4}|"
                  r"\[\d+(?:, ?\d+)*\]|\(\d{4}\)")


def chunk_quality(c: dict) -> str:
    """'' if the chunk is usable prose, else the exclusion reason."""
    t = c["text"]
    words = t.split()
    if len(t) < 600 or c["chunk_index"] == 0:
        return "short_or_title"
    if t.count("@") > 1:
        return "emails"
    if len(_REF.findall(t)) > 4:
        return "references"
    if sum(ch.isdigit() for ch in t) / len(t) > 0.08:
        return "table_or_numbers"
    if sum(w.isalpha() for w in words) / len(words) < 0.7:
        return "non_prose"
    return ""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passages(row: dict) -> list[dict]:
    ctx = row.get("context") or []
    sup = row.get("supporting_evidence") or []
    return (ctx if isinstance(ctx, list) else [ctx]) + list(sup)


def clean_grounded_qa(manifest: dict, chunks: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Keep grounded_qa train-split rows whose every passage is a verified train-document chunk."""
    split_of = {d: s for s, docs in manifest["splits"].items() for d in docs}
    kept, removed = [], []
    for r in rc.read_jsonl(rc.GROUNDED_QA / "examples.jsonl"):
        if r["split"] != "train":
            removed.append({"id": r["id"], "reason": f"grounded_qa split={r['split']} (evaluation question)"})
            continue
        bad = []
        for p in _passages(r):
            c = chunks.get(p["passage_id"])
            if c is None or c["text"] != p["text"] or c["source_file"] != p["document_id"]:
                bad.append(f"passage {p['passage_id']} does not match chunks.json")
            elif split_of[c["source_file"]] != "train":
                bad.append(f"passage {p['passage_id']} is from {split_of[c['source_file']]} document "
                           f"{c['source_file']}")
        if bad:
            removed.append({"id": r["id"], "example_type": r["example_type"], "reason": "; ".join(sorted(set(bad)))})
        else:
            kept.append(r)
    return kept, removed


def select_chunks(manifest: dict, chunks: dict[str, dict]) -> tuple[list[dict], Counter]:
    """Deterministic sample of non-adjacent, good-quality chunks from each training paper."""
    rng = random.Random(SEED)
    docs = rc.chunks_by_doc(chunks)
    reasons: Counter = Counter()
    rows = []
    for doc in sorted(manifest["splits"]["train"]):
        good = []
        for c in docs[doc]:
            why = chunk_quality(c)
            reasons[why or "usable"] += 1
            if not why:
                good.append(c)
        rng.shuffle(good)
        n = VALIDATION_CHUNKS if doc == VALIDATION_DOC else CHUNKS_PER_DOC
        picked: list[dict] = []
        for c in good:
            if all(abs(c["chunk_index"] - p["chunk_index"]) > 1 for p in picked):
                picked.append(c)
            if len(picked) == n:
                break
        for c in sorted(picked, key=lambda c: c["chunk_index"]):
            rows.append({"chunk_id": c["id"], "text": c["text"], "source": doc, "doc_id": doc,
                         "page": c["page_number"], "role": "validation" if doc == VALIDATION_DOC else "train"})
    return rows, reasons


# ---------------------------------------------------------------- SFT examples


def _evidence(ids: list[str], chunks: dict[str, dict], override: dict[str, str] | None = None) -> list[EvidenceChunk]:
    return [EvidenceChunk(chunk_id=i, text=(override or {}).get(i, chunks[i]["text"]), source=chunks[i]["source_file"],
                          page=chunks[i]["page_number"]) for i in ids]


def _row(question: str, ev: list[EvidenceChunk], target: str, kind: str, doc: str, gold: str, origin: str) -> dict:
    q = GroundedQuery(question=question, evidence=ev)
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": render_user_prompt(q)},
                         {"role": "assistant", "content": target}],
            "kind": kind, "doc_id": doc, "gold_chunk": gold, "evidence_ids": [e.chunk_id for e in ev],
            "origin": origin}


def _counterfactual(answer: str, text: str, rng: random.Random) -> tuple[str, str] | None:
    """Change one integer that occurs exactly once in the answer and once in the chunk."""
    cands = [n for n in sorted(numbers(answer)) if n.isdigit() and 1 < int(n) < 100000
             and len(re.findall(rf"(?<![\d.,]){n}(?![\d.,])", answer)) == 1
             and len(re.findall(rf"(?<![\d.,]){n}(?![\d.,])", text)) == 1]
    if not cands:
        return None
    n = rng.choice(cands)
    new = str(int(n) + rng.choice([-1, 1]) * max(1, round(int(n) * rng.uniform(0.2, 0.6))))
    if new == n or int(new) <= 0 or new in numbers(text):
        return None
    sub = lambda s: re.sub(rf"(?<![\d.,]){n}(?![\d.,])", new, s)  # noqa: E731
    return sub(answer), sub(text)


def build_examples(qa: list[dict], ooc_rows: list[dict], chunks: dict[str, dict], manifest: dict,
                   seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    docs = rc.chunks_by_doc(chunks)
    train_docs = sorted(manifest["splits"]["train"])
    out = []
    for item in qa:
        gold = chunks[item["chunk_id"]]
        doc, q, a = gold["source_file"], item["question"], item["answer"]
        near = {f"{doc}::{gold['chunk_index'] + o:04d}" for o in (-1, 0, 1)}
        pool = [c for c in docs[doc] if c["id"] not in near and len(c["text"]) >= 100
                and not claim_supported(a, c["text"])]
        ranked = rc.rank_chunks(q, pool)
        ev = [gold["id"]] + ranked[:K - 1]
        rng.shuffle(ev)
        out.append(_row(q, _evidence(ev, chunks), format_target(a, [gold["id"]], False), "answerable", doc,
                        gold["id"], "curatorkit"))
        cf = _counterfactual(a, gold["text"], rng) if rng.random() < 0.3 else None
        if cf:
            ev_cf = [gold["id"]] + ranked[:K - 1]
            rng.shuffle(ev_cf)
            out.append(_row(q, _evidence(ev_cf, chunks, {gold["id"]: cf[1]}), format_target(cf[0], [gold["id"]], False),
                            "answerable_counterfactual", doc, gold["id"], "curatorkit+number_perturbation"))
        if rng.random() < 0.5:
            hard = ranked[:K]
            rng.shuffle(hard)
            out.append(_row(q, _evidence(hard, chunks), format_target("", [], True), "abstain_hard_negative", doc,
                            gold["id"], "curatorkit"))
        if rng.random() < 0.25:
            other = rng.choice([d for d in train_docs if d != doc])
            swap = rc.rank_chunks(q, [c for c in docs[other] if len(c["text"]) >= 100
                                      and not claim_supported(a, c["text"])])[:K]
            out.append(_row(q, _evidence(swap, chunks), format_target("", [], True), "abstain_context_swap", doc,
                            gold["id"], "curatorkit"))
        if rng.random() < 0.08:
            out.append(_row(q, [], format_target("", [], True), "abstain_no_evidence", doc, gold["id"], "curatorkit"))
    for r in ooc_rows:
        ids = [p["passage_id"] for p in _passages(r)][:K]
        doc = chunks[ids[0]]["source_file"] if ids else ""
        out.append(_row(r["question"], _evidence(ids, chunks), format_target("", [], True), "abstain_out_of_corpus",
                        doc, "", f"grounded_qa:{r['id']}"))
    return out


# ---------------------------------------------------------------- pipeline


def run_judge_gate(out: Path, judge_model: str, judge_base: str, threshold: float, concurrency: int):
    """CuratorKIT gate-only run (no generation) over the pairs accepted by the generation run."""
    from curatorkit import Curator, CuratorConfig

    cfg = CuratorConfig(name="ps3_real_train_independent_gate", dataset=str(out / "curatorkit" / "sft_alpaca.jsonl"),
                        format="alpaca", judge_llm_model=judge_model, judge_llm_api_base=judge_base,
                        llm_api_key="EMPTY", llm_concurrency=concurrency, hallucination_threshold=threshold,
                        min_tokens=3, dedup="exact", export_formats=["alpaca"],
                        output_dir=str(out / "curatorkit_judge"))
    return Curator(cfg).run()


def load_curated(out: Path, chunks_rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Pairs accepted by both CuratorKIT gates -> strict gate. Returns (accepted, rejected, stats)."""
    text_to_id = {r["text"].strip(): r["chunk_id"] for r in chunks_rows}
    passed = rc.read_jsonl(out / "curatorkit_judge" / "sft_alpaca.jsonl")
    accepted, rejected, log = [], [], Counter()
    seen_q: set[str] = set()
    for s in passed:
        cid = text_to_id.get((s.get("input") or "").strip())
        if cid is None:
            log["unmatched_to_chunk"] += 1
            continue
        qn = " ".join(s["instruction"].lower().split())
        if qn in seen_q:
            log["duplicate_question"] += 1
            rejected.append({**s, "chunk_id": cid, "reason": "duplicate_question"})
            continue
        answer, note = strict_ground(s["output"].strip(), s["input"])
        if answer is None:
            log[note] += 1
            rejected.append({**s, "chunk_id": cid, "reason": note})
            continue
        if len(answer.split()) > 60:
            log["answer_too_long"] += 1
            rejected.append({**s, "chunk_id": cid, "reason": "answer_too_long"})
            continue
        seen_q.add(qn)
        log[note or "kept_unchanged"] += 1
        accepted.append({"question": s["instruction"].strip(), "answer": answer, "chunk_id": cid,
                         "curatorkit_answer": s["output"].strip()})
    return accepted, rejected, dict(log)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rc.write_jsonl(path, rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ps3_llm.realtrain")
    ap.add_argument("--out", type=Path, default=OUT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("clean")
    sub.add_parser("chunks")
    c = sub.add_parser("curate")
    c.add_argument("--llm-model", default="openai/mlx-community/Qwen2.5-7B-Instruct-4bit")
    c.add_argument("--api-base", default="http://127.0.0.1:30000/v1")
    c.add_argument("--judge-model", default=None)
    c.add_argument("--judge-base", default=None)
    c.add_argument("--num-questions", type=int, default=2)
    c.add_argument("--hallucination-threshold", type=float, default=0.7)
    c.add_argument("--concurrency", type=int, default=4)
    j = sub.add_parser("judge")
    j.add_argument("--judge-model", default="openai/mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    j.add_argument("--judge-base", default="http://127.0.0.1:30000/v1")
    j.add_argument("--hallucination-threshold", type=float, default=0.7)
    j.add_argument("--concurrency", type=int, default=4)
    sub.add_parser("build")
    args = ap.parse_args(argv)
    manifest, chunks = rc.load_manifest(), rc.load_chunks()
    out: Path = args.out

    if args.cmd == "clean":
        kept, removed = clean_grounded_qa(manifest, chunks)
        _write(out / "grounded_qa_train_clean.jsonl", kept)
        _write(out / "grounded_qa_removed.jsonl", removed)
        print(json.dumps({"kept": len(kept), "removed": len(removed),
                          "removed_train_rows": sum(not r["reason"].startswith("grounded_qa split") for r in removed),
                          "kept_by_type": dict(Counter(r["example_type"] for r in kept))}, indent=2))
        return 0
    if args.cmd == "chunks":
        rows, reasons = select_chunks(manifest, chunks)
        _write(out / "train_chunks.jsonl", rows)
        print(json.dumps({"selected": len(rows), "by_role": dict(Counter(r["role"] for r in rows)),
                          "chunk_quality": dict(reasons)}, indent=2))
        return 0
    rows = rc.read_jsonl(out / "train_chunks.jsonl")
    if args.cmd == "curate":
        if (out / "curatorkit").exists() and any((out / "curatorkit").iterdir()):
            raise SystemExit(f"refusing to overwrite {out / 'curatorkit'}")
        res_rows, result = run_curatorkit(out / "train_chunks.jsonl", out, args.llm_model, args.api_base,
                                          args.num_questions, args.hallucination_threshold, args.concurrency,
                                          judge_model=args.judge_model, judge_base=args.judge_base)
        result.print_summary()
        (out / "curatorkit_run.json").write_text(json.dumps({
            "llm_model": args.llm_model, "judge_model": args.judge_model or args.llm_model,
            "num_questions": args.num_questions, "hallucination_threshold": args.hallucination_threshold,
            "n_chunks": len(res_rows), "passed": len(result.passed), "rejected": len(result.rejected)}, indent=2))
        return 0
    if args.cmd == "judge":
        if (out / "curatorkit_judge").exists() and any((out / "curatorkit_judge").iterdir()):
            raise SystemExit(f"refusing to overwrite {out / 'curatorkit_judge'}")
        result = run_judge_gate(out, args.judge_model, args.judge_base, args.hallucination_threshold, args.concurrency)
        result.print_summary()
        (out / "curatorkit_judge_run.json").write_text(json.dumps({
            "judge_model": args.judge_model, "hallucination_threshold": args.hallucination_threshold,
            "passed": len(result.passed), "rejected": len(result.rejected)}, indent=2))
        return 0
    if args.cmd == "build":
        accepted, strict_rej, strict_log = load_curated(out, rows)
        _write(out / "accepted_qa.jsonl", accepted)
        _write(out / "strict_gate_rejected.jsonl", strict_rej)
        role = {r["chunk_id"]: r["role"] for r in rows}
        ooc = [r for r in rc.read_jsonl(out / "grounded_qa_train_clean.jsonl")
               if r["example_type"] == "unanswerable_out_of_corpus" and r.get("label_confidence") == "verified"]
        examples = build_examples(accepted, ooc, chunks, manifest)
        split = {"train": [e for e in examples if e["doc_id"] != VALIDATION_DOC],
                 "validation": [e for e in examples if e["doc_id"] == VALIDATION_DOC]}
        for name, ex in split.items():
            _write(out / f"sft_{name}.jsonl", ex)
        lk = rc.leakage_report(manifest, rc.load_items(), chunks,
                               [out / f"sft_{n}.jsonl" for n in split] + [out / "accepted_qa.jsonl"])
        ck = json.loads((out / "curatorkit" / "manifest.json").read_text())
        ck2 = json.loads((out / "curatorkit_judge" / "manifest.json").read_text())
        files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
        manifest_out = {
            "version": VERSION, "seed": SEED, "validation_doc": VALIDATION_DOC,
            "split_manifest_sha256": _sha(rc.RC / "split_manifest.json"),
            "chunks_sha256": manifest["source"]["chunks_sha256"],
            "grounded_qa_examples_sha256": _sha(rc.GROUNDED_QA / "examples.jsonl"),
            "train_documents": sorted(manifest["splits"]["train"]),
            "curatorkit": {"run": json.loads((out / "curatorkit_run.json").read_text()),
                           "stage_counts": ck.get("stage_counts"), "rejected_breakdown": ck.get("rejected_breakdown")},
            "curatorkit_independent_gate": {"run": json.loads((out / "curatorkit_judge_run.json").read_text()),
                                            "stage_counts": ck2.get("stage_counts"),
                                            "rejected_breakdown": ck2.get("rejected_breakdown")},
            "strict_gate": strict_log, "qa_pairs": len(accepted),
            "chunks_by_role": dict(Counter(role.values())),
            "examples": {n: dict(Counter(e["kind"] for e in ex)) | {"total": len(ex)} for n, ex in split.items()},
            "leakage": {"errors": lk["errors"], "files": lk["checked_files"]},
            "files": {str(p.relative_to(out)): _sha(p) for p in files},
        }
        (out / "manifest.json").write_text(json.dumps(manifest_out, indent=2) + "\n")
        print(json.dumps({k: manifest_out[k] for k in ("strict_gate", "qa_pairs", "examples")}, indent=2))
        if lk["errors"]:
            print("LEAKAGE:", lk["errors"], file=sys.stderr)
            return 1
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
