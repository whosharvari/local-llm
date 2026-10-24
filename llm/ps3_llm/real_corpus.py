"""Real-corpus evaluation set: build, validate, human review, conversion, leakage checks and reporting.

The 13 papers in ../data (Member 1) are split BY DOCUMENT (data/real_corpus/split_manifest.json).
Labels are drafted in data/real_corpus/drafts.jsonl; `build` expands them into data/real_corpus/items.jsonl
(evidence text, page numbers, gold citations, label SHA-256) and refuses to write if any provenance check
fails. Every label is a machine draft until a human reviewer accepts it in data/real_corpus/reviews.jsonl.
A review is bound to the label's SHA-256, so editing an accepted label returns it to machine_draft.

    python -m ps3_llm.real_corpus build                    # drafts.jsonl -> items.jsonl
    python -m ps3_llm.real_corpus validate                 # schema + provenance + reviews + leakage; exit 1 on error
    python -m ps3_llm.real_corpus status
    python -m ps3_llm.real_corpus review --split test --reviewer "Full Name"          # interactive
    python -m ps3_llm.real_corpus review-sheet --split test --out /tmp/review_test     # .md + .csv to fill in
    python -m ps3_llm.real_corpus import-reviews /tmp/review_test.csv
    python -m ps3_llm.real_corpus convert --split dev --labels provisional              # -> ps3_llm.evaluate format
    python -m ps3_llm.real_corpus freeze --split test --name test-v1                    # human-verified labels only
    python -m ps3_llm.real_corpus leakage [--train extra_training_file.jsonl ...]
    python -m ps3_llm.real_corpus report results/real_corpus/<tag>                      # abstention P/R etc.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .metrics import MODES, _has, aggregate, content_tokens, facts_match, numbers
from .schemas import validate_request

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
RC = ROOT / "data/real_corpus"
CHUNKS = REPO / "data/processed/chunks.json"
GROUNDED_QA = REPO / "data/datasets/grounded_qa"

# category -> answerable
CATEGORIES = {"answerable": True, "numerical": True, "unanswerable": False, "context_swap": False, "out_of_corpus": False}
STATUSES = ("machine_draft", "needs_revision", "human_rejected", "human_verified")
DECISIONS = {"accept": "human_verified", "reject": "human_rejected", "needs_revision": "needs_revision"}
DRAFTED_BY = "claude-opus-5-5 via Claude Code (machine draft)"
DRAFTED_AT = "2026-10-24"
MACHINE_REVIEWER = re.compile(r"claude|gpt|\bllm\b|\bmodel\b|\bauto|\bbot\b|machine|\bai\b|^\W*$", re.I)
ID_PATTERN = re.compile(r"^rc([td])-[a-z0-9]+-\d{2}$")
ITEM_KEYS = ("id", "split", "category", "tags", "answerable", "expected_behavior", "question", "reference_answer",
             "required_facts", "reject", "acceptable_variations", "support", "gold_citations", "evidence",
             "source_document", "swap_of", "absent_terms", "distractor_answers", "rationale", "provenance",
             "label_sha256")
LABEL_KEYS = ("id", "split", "category", "answerable", "question", "reference_answer", "required_facts", "reject",
              "support", "gold_citations", "evidence", "absent_terms")
MIN_DISTRACTOR_CHARS = 100
MIN_QUOTE_CHARS = 15
NGRAM = 13


# ---------------------------------------------------------------- loading


def norm(text: str) -> str:
    return " ".join(text.split())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def load_manifest(path: Path = RC / "split_manifest.json") -> dict:
    return json.loads(path.read_text())


def load_chunks(path: Path = CHUNKS) -> dict[str, dict]:
    return {c["id"]: c for c in json.loads(path.read_text())}


def chunks_by_doc(chunks: dict[str, dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for c in chunks.values():
        out[c["source_file"]].append(c)
    for cs in out.values():
        cs.sort(key=lambda c: c["chunk_index"])
    return dict(out)


def load_items(path: Path = RC / "items.jsonl") -> list[dict]:
    return read_jsonl(path)


def label_hash(item: dict) -> str:
    blob = json.dumps({k: item[k] for k in LABEL_KEYS}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------- build


def rank_chunks(question: str, candidates: list[dict]) -> list[str]:
    """Deterministic lexical ranking (idf-weighted question-term overlap), ties broken by chunk id.
    Stands in for retrieval to pick hard distractors from the allowed documents."""
    q = set(content_tokens(question))
    toks = {c["id"]: set(content_tokens(c["text"])) for c in candidates}
    df = Counter(t for ts in toks.values() for t in ts)
    n = len(candidates)
    score = {cid: sum(math.log(1 + n / df[t]) for t in q & ts) for cid, ts in toks.items()}
    return sorted(score, key=lambda cid: (-score[cid], cid))


def _any_alt(text: str, facts: list[list[str]]) -> bool:
    return any(_has(text, alt) for fact in facts for alt in fact)


def build_item(d: dict, drafts: dict[str, dict], chunks: dict[str, dict], docs: dict[str, list[dict]],
               manifest: dict) -> dict:
    cat = d["category"]
    if cat not in CATEGORIES:
        raise ValueError(f"{d['id']}: unknown category {cat!r}")
    answerable = CATEGORIES[cat]
    facts = d.get("required_facts", []) if answerable else []
    for s in d.get("support", []):
        if s["chunk_id"] not in chunks:
            raise ValueError(f"{d['id']}: support chunk {s['chunk_id']} not in chunks.json")
    support = [{"chunk_id": s["chunk_id"], "document_id": chunks[s["chunk_id"]]["source_file"],
                "page_number": chunks[s["chunk_id"]]["page_number"], "quote": s["quote"]} for s in d.get("support", [])]
    support_ids = list(dict.fromkeys(s["chunk_id"] for s in support))
    fixed = list(dict.fromkeys(support_ids + d.get("pinned_evidence", [])))
    excluded: set[str] = set()

    if answerable:
        source = support[0]["document_id"] if support else None
        pool = [source] if source else []
        for cid in support_ids:            # overlapping neighbours would duplicate (part of) the gold passage
            c = chunks[cid]
            excluded |= {f"{c['source_file']}::{c['chunk_index'] + o:04d}" for o in (-1, 1)}
        excluded |= {c["id"] for doc in pool for c in docs[doc] if facts_match(c["text"], facts)}
    elif cat == "unanswerable":
        source = d["source_document"]
        pool = [source]
    elif cat == "context_swap":
        orig = drafts[d["swap_of"]]
        source = chunks[orig["support"][0]["chunk_id"]]["source_file"]
        pool = list(d["evidence_from"])
        excluded |= {c["id"] for doc in pool for c in docs[doc] if _any_alt(c["text"], orig["required_facts"])}
    else:  # out_of_corpus
        source = None
        pool = list(manifest["splits"][d["split"]])

    candidates = [c for doc in pool for c in docs[doc]
                  if c["id"] not in fixed and c["id"] not in excluded and len(c["text"]) >= MIN_DISTRACTOR_CHARS]
    k = manifest["evidence_k"]
    evidence_ids = fixed + rank_chunks(d["question"], candidates)[: max(0, k - len(fixed))]
    random.Random(f"{manifest['seed']}:{d['id']}").shuffle(evidence_ids)
    quotes = [norm(s["quote"]) for s in support]
    gold = [cid for cid in evidence_ids
            if cid in support_ids or any(q in norm(chunks[cid]["text"]) for q in quotes)] if answerable else []

    item = {
        "id": d["id"], "split": d["split"], "category": cat, "tags": d.get("tags", []), "answerable": answerable,
        "expected_behavior": "answer" if answerable else "abstain",
        "question": d["question"],
        "reference_answer": d.get("reference_answer") if answerable else None,
        "required_facts": facts, "reject": d.get("reject", []),
        "acceptable_variations": d.get("acceptable_variations", []),
        "support": support, "gold_citations": gold,
        "evidence": [{"chunk_id": cid, "document_id": chunks[cid]["source_file"],
                      "page_number": chunks[cid]["page_number"], "text": chunks[cid]["text"]} for cid in evidence_ids],
        "source_document": source, "swap_of": d.get("swap_of"),
        "absent_terms": d.get("absent_terms", []), "distractor_answers": d.get("distractor_answers", []),
        "rationale": d.get("rationale", ""),
        "provenance": {"question_origin": d.get("question_origin", ""), "label_origin": "machine_draft",
                       "drafted_by": DRAFTED_BY, "drafted_at": DRAFTED_AT,
                       "chunks_sha256": manifest["source"]["chunks_sha256"]},
    }
    item["label_sha256"] = label_hash(item)
    return item


def build_items(drafts: list[dict], chunks: dict[str, dict], manifest: dict) -> list[dict]:
    by_id = {d["id"]: d for d in drafts}
    if len(by_id) != len(drafts):
        raise ValueError("duplicate draft ids")
    docs = chunks_by_doc(chunks)
    return [build_item(d, by_id, chunks, docs, manifest) for d in drafts]


# ---------------------------------------------------------------- validation (schema + provenance)


def _is_facts(x: Any) -> bool:
    return isinstance(x, list) and all(isinstance(f, list) and f and all(isinstance(a, str) and a.strip() for a in f)
                                       for f in x)


def validate_item(it: dict, chunks: dict[str, dict], manifest: dict, items: dict[str, dict],
                  corpus_text: list[str] | None = None) -> list[str]:
    """Schema, provenance and per-item leakage checks. Returns a list of error strings."""
    iid = it.get("id", "?")
    errs = [f"{iid}: missing key {k}" for k in ITEM_KEYS if k not in it]
    if errs:
        return errs
    e = lambda msg: errs.append(f"{iid}: {msg}")  # noqa: E731
    m = ID_PATTERN.match(iid)
    if not m or {"t": "test", "d": "dev"}[m.group(1)] != it["split"]:
        e("id must match rc{t|d}-<name>-NN with t=test, d=dev")
    if it["split"] not in ("test", "dev"):
        e(f"split must be test or dev, got {it['split']!r}")
    if it["category"] not in CATEGORIES:
        return errs + [f"{iid}: unknown category {it['category']!r}"]
    ans = CATEGORIES[it["category"]]
    if it["answerable"] is not ans or it["expected_behavior"] != ("answer" if ans else "abstain"):
        e("answerable / expected_behavior inconsistent with category")
    if not isinstance(it["question"], str) or not it["question"].strip().endswith("?"):
        e("question must be a non-empty string ending with '?'")
    if not isinstance(it["rationale"], str) or not it["rationale"].strip():
        e("rationale is required")
    if it["label_sha256"] != label_hash(it):
        e("label_sha256 does not match the label content (rebuild items.jsonl)")
    prov = it["provenance"]
    if prov.get("label_origin") != "machine_draft" or not prov.get("drafted_by") or not prov.get("question_origin"):
        e("provenance must record label_origin=machine_draft, drafted_by and question_origin")
    if prov.get("chunks_sha256") != manifest["source"]["chunks_sha256"]:
        e("provenance.chunks_sha256 differs from the manifest")

    split_docs = set(manifest["splits"].get(it["split"], []))
    ev_ids = [x["chunk_id"] for x in it["evidence"]]
    if len(set(ev_ids)) != len(ev_ids) or not ev_ids:
        e("evidence chunk ids must be non-empty and unique")
    req_errs = validate_request({"query_id": iid, "question": it["question"],
                                 "evidence": [{"chunk_id": x["chunk_id"], "text": x["text"]} for x in it["evidence"]]})
    errs.extend(f"{iid}: contract: {x}" for x in req_errs)
    for x in it["evidence"]:
        c = chunks.get(x["chunk_id"])
        if c is None:
            e(f"evidence {x['chunk_id']} not in chunks.json")
            continue
        if x["text"] != c["text"] or x["document_id"] != c["source_file"] or x["page_number"] != c["page_number"]:
            e(f"evidence {x['chunk_id']} text/document/page differs from chunks.json")
        if c["source_file"] not in split_docs:
            e(f"LEAKAGE: evidence {x['chunk_id']} is from {c['source_file']}, not a {it['split']} document")
    if it["source_document"] is not None and it["source_document"] not in split_docs:
        e(f"LEAKAGE: source_document {it['source_document']} is not a {it['split']} document")

    quotes = []
    for s in it["support"]:
        c = chunks.get(s["chunk_id"])
        if c is None:
            e(f"support {s['chunk_id']} not in chunks.json")
            continue
        if s["document_id"] != c["source_file"] or s["page_number"] != c["page_number"]:
            e(f"support {s['chunk_id']} document/page differs from chunks.json")
        if c["source_file"] not in split_docs:
            e(f"LEAKAGE: support {s['chunk_id']} is not from a {it['split']} document")
        if len(s["quote"]) < MIN_QUOTE_CHARS or norm(s["quote"]) not in norm(c["text"]):
            e(f"support quote is not a verbatim passage of {s['chunk_id']}: {s['quote'][:60]!r}")
        if s["chunk_id"] not in ev_ids:
            e(f"support chunk {s['chunk_id']} is not in the evidence given to the model")
        quotes.append(s["quote"])

    if ans:
        if not isinstance(it["reference_answer"], str) or not it["reference_answer"].strip():
            e("answerable item needs a reference_answer")
        if not _is_facts(it["required_facts"]) or not it["required_facts"]:
            e("answerable item needs required_facts: a non-empty list of non-empty alternative lists")
        elif not facts_match(it["reference_answer"] or "", it["required_facts"], it["reject"]):
            e("reference_answer does not satisfy its own required_facts")
        if not it["support"] or not it["gold_citations"] or not set(it["gold_citations"]) <= set(ev_ids):
            e("answerable item needs support passages and gold_citations within the evidence")
        joined = " ".join(quotes)
        for fact in it["required_facts"] if _is_facts(it["required_facts"]) else []:
            if not any(_has(joined, alt) for alt in fact):
                e(f"required fact {fact} is not stated in the support quotes")
        for x in it["evidence"]:
            if x["chunk_id"] not in it["gold_citations"] and facts_match(x["text"], it["required_facts"]):
                e(f"distractor {x['chunk_id']} also satisfies the required facts (would need to be gold)")
    else:
        if it["reference_answer"] is not None or it["required_facts"] or it["support"] or it["gold_citations"]:
            e("unanswerable item must have reference_answer=null and no facts, support or gold citations")
    ev_text = " ".join(x["text"] for x in it["evidence"])
    for term in it["absent_terms"]:
        if _has(ev_text, term):
            e(f"absent term {term!r} occurs in the evidence")
    if it["category"] == "out_of_corpus":
        if not it["absent_terms"]:
            e("out_of_corpus item needs absent_terms")
        for term in it["absent_terms"]:
            if corpus_text is not None and any(_has(t, term) for t in corpus_text):
                e(f"out_of_corpus term {term!r} occurs in the corpus")
    if it["category"] == "context_swap":
        orig = items.get(it["swap_of"] or "")
        if not orig or not orig["answerable"] or orig["split"] != it["split"]:
            e("context_swap must reference an answerable item of the same split")
        else:
            if orig["question"] != it["question"]:
                e("context_swap question must equal the original question")
            if {x["document_id"] for x in it["evidence"]} & {orig["source_document"]}:
                e("context_swap evidence must not come from the original document")
            if _any_alt(ev_text, orig["required_facts"]):
                e("context_swap evidence contains a fact of the original answer")
    return errs


def validate_all(items: list[dict], chunks: dict[str, dict], manifest: dict, reviews: list[dict] | None = None,
                 chunks_path: Path = CHUNKS) -> list[str]:
    errs = check_partition(manifest, chunks)
    if sha256_file(chunks_path) != manifest["source"]["chunks_sha256"]:
        errs.append("chunks.json SHA-256 differs from split_manifest.json (source corpus changed)")
    by_id: dict[str, dict] = {}
    for it in items:
        if it.get("id") in by_id:
            errs.append(f"duplicate item id {it['id']}")
        by_id[it.get("id", "?")] = it
    corpus_text = [c["text"] for c in chunks.values()]
    for it in items:
        errs.extend(validate_item(it, chunks, manifest, by_id, corpus_text))
    q_split: dict[str, set] = defaultdict(set)
    for it in items:
        q_split[norm(it["question"]).lower()].add(it["split"])
    errs.extend(f"LEAKAGE: question appears in several splits: {q!r}" for q, s in q_split.items() if len(s) > 1)
    for i, r in enumerate(reviews or []):
        errs.extend(f"reviews.jsonl line {i + 1}: {x}" for x in validate_review(r, by_id))
    return errs


def check_partition(manifest: dict, chunks: dict[str, dict]) -> list[str]:
    splits = manifest["splits"]
    errs = []
    seen: dict[str, str] = {}
    for s, docs in splits.items():
        for d in docs:
            if d in seen:
                errs.append(f"LEAKAGE: document {d} is in both {seen[d]} and {s}")
            seen[d] = s
    corpus_docs = {c["source_file"] for c in chunks.values()}
    if set(seen) != corpus_docs:
        errs.append(f"split manifest does not cover the corpus exactly: missing {sorted(corpus_docs - set(seen))}, "
                    f"unknown {sorted(set(seen) - corpus_docs)}")
    return errs


# ---------------------------------------------------------------- human review


def validate_review(r: dict, items: dict[str, dict]) -> list[str]:
    errs = []
    if r.get("item_id") not in items:
        errs.append(f"unknown item_id {r.get('item_id')!r}")
    if r.get("decision") not in DECISIONS:
        errs.append(f"decision must be one of {sorted(DECISIONS)}")
    reviewer = r.get("reviewer") or ""
    if MACHINE_REVIEWER.search(reviewer) or reviewer == DRAFTED_BY:
        errs.append(f"reviewer {reviewer!r} is empty or looks like a model; reviews must be by a named human")
    if not re.fullmatch(r"[0-9a-f]{64}", r.get("label_sha256") or ""):
        errs.append("label_sha256 must be the 64-hex label hash of the reviewed item")
    if r.get("decision") in ("reject", "needs_revision") and not (r.get("notes") or "").strip():
        errs.append("reject / needs_revision needs notes")
    return errs


def effective_status(items: list[dict], reviews: list[dict]) -> dict[str, dict]:
    """Latest valid review whose label hash matches the current label wins; otherwise machine_draft."""
    by_id = {it["id"]: it for it in items}
    out = {it["id"]: {"status": "machine_draft", "reviewer": None, "reviewed_at": None, "stale_reviews": 0}
           for it in items}
    for r in reviews:
        if validate_review(r, by_id):
            continue
        st = out[r["item_id"]]
        if r["label_sha256"] != by_id[r["item_id"]]["label_sha256"]:
            st["stale_reviews"] += 1
            continue
        st.update(status=DECISIONS[r["decision"]], reviewer=r["reviewer"], reviewed_at=r.get("reviewed_at"))
    return out


def render_item(it: dict, status: str, markdown: bool = False) -> str:
    h = "### " if markdown else "=== "
    lines = [f"{h}{it['id']}  [{it['split']} / {it['category']}]  status: {status}",
             f"Question: {it['question']}",
             f"Expected: {it['expected_behavior'].upper()}"]
    if it["answerable"]:
        lines += [f"Reference answer: {it['reference_answer']}",
                  "Required facts (each needs one alternative): " + " AND ".join("(" + " | ".join(f) + ")"
                                                                                for f in it["required_facts"])]
        if it["reject"]:
            lines.append(f"Reject phrases: {it['reject']}")
        lines += [f"Support: [{s['chunk_id']}] p.{s['page_number']}: \"{s['quote']}\"" for s in it["support"]]
    else:
        if it["swap_of"]:
            lines.append(f"Context swap of: {it['swap_of']}")
        if it["absent_terms"]:
            lines.append(f"Verified-absent terms: {it['absent_terms']}")
        if it["distractor_answers"]:
            lines.append(f"Likely hallucinations: {it['distractor_answers']}")
    lines.append(f"Rationale: {it['rationale']}")
    lines.append(f"Question origin: {it['provenance']['question_origin']}   label hash: {it['label_sha256']}")
    lines.append("Evidence given to the model:")
    for x in it["evidence"]:
        mark = " [GOLD]" if x["chunk_id"] in it["gold_citations"] else ""
        lines.append(f"{'- ' if markdown else '  '}[{x['chunk_id']}] ({x['document_id']}, p.{x['page_number']}){mark}")
        lines.append(f"{'  > ' if markdown else '      '}{norm(x['text'])}")
    lines.append("Check: the reference answer is correct and complete; every required fact is necessary; the GOLD "
                 "passages support it; no other passage answers the question; for abstain items, NO passage answers it.")
    return "\n".join(lines)


def append_review(path: Path, rec: dict, items: dict[str, dict]) -> None:
    errs = validate_review(rec, items)
    if errs:
        raise ValueError(f"invalid review for {rec.get('item_id')}: {errs}")
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def interactive_review(items: list[dict], reviews_path: Path, split: str, reviewer: str, redo: bool = False) -> int:
    by_id = {it["id"]: it for it in items}
    status = effective_status(items, read_jsonl(reviews_path))
    todo = [it for it in items if it["split"] == split
            and (redo or status[it["id"]]["status"] in ("machine_draft", "needs_revision"))]
    print(f"{len(todo)} items to review in {split}. Decisions are appended to {reviews_path}.")
    done = 0
    for it in todo:
        print("\n" + render_item(it, status[it["id"]]["status"]))
        choice = input("\n[a]ccept  [r]eject  [n]eeds revision  [s]kip  [q]uit > ").strip().lower()[:1]
        if choice == "q":
            break
        if choice not in ("a", "r", "n"):
            continue
        decision = {"a": "accept", "r": "reject", "n": "needs_revision"}[choice]
        notes = input("notes (required unless accepting) > ").strip()
        append_review(reviews_path, {"item_id": it["id"], "label_sha256": it["label_sha256"], "decision": decision,
                                     "reviewer": reviewer, "reviewed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                     "notes": notes}, by_id)
        done += 1
    return done


def write_review_sheet(items: list[dict], status: dict[str, dict], split: str, out_stem: Path) -> tuple[Path, Path]:
    sel = [it for it in items if it["split"] == split]
    md = out_stem.with_suffix(".md")
    md.write_text(f"# Review sheet — {split} ({len(sel)} items)\n\nFill in `{out_stem.with_suffix('.csv').name}` "
                  "(decision = accept | reject | needs_revision; reviewer = your full name) and import it with "
                  "`python -m ps3_llm.real_corpus import-reviews <csv>`.\n\n"
                  + "\n\n".join(render_item(it, status[it["id"]]["status"], markdown=True) for it in sel) + "\n")
    out_csv = out_stem.with_suffix(".csv")
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "label_sha256", "decision", "reviewer", "notes"])
        for it in sel:
            w.writerow([it["id"], it["label_sha256"], "", "", ""])
    return md, out_csv


def import_reviews(csv_path: Path, items: list[dict], reviews_path: Path) -> int:
    by_id = {it["id"]: it for it in items}
    with csv_path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("decision") or "").strip()]
    recs = [{"item_id": r["item_id"].strip(), "label_sha256": r["label_sha256"].strip(),
             "decision": r["decision"].strip(), "reviewer": (r.get("reviewer") or "").strip(),
             "reviewed_at": (r.get("reviewed_at") or "").strip() or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "notes": (r.get("notes") or "").strip()} for r in rows]
    errs = [f"{r['item_id']}: {x}" for r in recs for x in validate_review(r, by_id)]
    errs += [f"{r['item_id']}: label changed since the sheet was exported" for r in recs
             if r["item_id"] in by_id and r["label_sha256"] != by_id[r["item_id"]]["label_sha256"]]
    if errs:
        raise SystemExit("not imported:\n" + "\n".join(errs))
    for r in recs:
        append_review(reviews_path, r, by_id)
    return len(recs)


# ---------------------------------------------------------------- conversion to the ps3_llm.evaluate format


def convert(items: list[dict], status: dict[str, dict], split: str, labels: str, version: str) -> tuple[list, dict]:
    """labels='verified': human-verified items only. labels='provisional': every item not rejected by a human,
    including machine drafts (each example records its annotation_status)."""
    if labels not in ("verified", "provisional"):
        raise ValueError("labels must be 'verified' or 'provisional'")
    keep = (lambda s: s == "human_verified") if labels == "verified" else (lambda s: s != "human_rejected")
    sel = [it for it in items if it["split"] == split and keep(status[it["id"]]["status"])]
    examples = [{
        "query_id": it["id"], "question": it["question"],
        "evidence": [{"chunk_id": x["chunk_id"], "text": x["text"], "source": x["document_id"],
                      "page": x["page_number"]} for x in it["evidence"]],
        "answerable": it["answerable"], "gold_answer": it["reference_answer"] or "",
        "gold_citations": it["gold_citations"], "category": it["category"],
        "annotation_status": status[it["id"]]["status"], "label_sha256": it["label_sha256"],
    } for it in sel]
    adjudication = {
        "_version": f"real-corpus {version}, split={split}, labels={labels}",
        "_purpose": "Correctness criteria generated from data/real_corpus/items.jsonl by ps3_llm.real_corpus convert. "
                    "Same semantics as data/mock/heldout_adjudication.json: every inner list is one required fact, "
                    "satisfied if ANY alternative occurs in the answer; 'reject' phrases make the answer wrong.",
        "_annotation_status": dict(Counter(e["annotation_status"] for e in examples)),
        "items": {it["id"]: {"facts": it["required_facts"], **({"reject": it["reject"]} if it["reject"] else {})}
                  for it in sel if it["answerable"]},
    }
    return examples, adjudication


def write_converted(examples: list, adjudication: dict, out_dir: Path, stem: str,
                    overwrite: bool = True) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data, adj = out_dir / f"{stem}.jsonl", out_dir / f"{stem}_adjudication.json"
    if not overwrite and (data.exists() or adj.exists()):
        raise SystemExit(f"refusing to overwrite {data} / {adj}")
    write_jsonl(data, examples)
    adj.write_text(json.dumps(adjudication, indent=2, ensure_ascii=False) + "\n")
    return data, adj


def freeze(items: list[dict], status: dict[str, dict], split: str, name: str, version: str,
           out_dir: Path = RC / "frozen") -> Path:
    sel = [it for it in items if it["split"] == split]
    pending = [it["id"] for it in sel if status[it["id"]]["status"] in ("machine_draft", "needs_revision")]
    if pending:
        raise SystemExit(f"cannot freeze {split}: {len(pending)} items are not human-reviewed: {pending[:10]}...")
    examples, adjudication = convert(items, status, split, "verified", version)
    if not examples:
        raise SystemExit(f"cannot freeze {split}: no human-verified items")
    data, adj = write_converted(examples, adjudication, out_dir, name, overwrite=False)
    man = out_dir / f"{name}.manifest.json"
    man.write_text(json.dumps({
        "name": name, "split": split, "dataset_version": version, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_items": len(examples), "n_rejected": sum(status[it["id"]]["status"] == "human_rejected" for it in sel),
        "data": data.name, "data_sha256": sha256_file(data), "adjudication": adj.name,
        "adjudication_sha256": sha256_file(adj),
        "items": {e["query_id"]: {"label_sha256": e["label_sha256"], "reviewer": status[e["query_id"]]["reviewer"]}
                  for e in examples},
    }, indent=2) + "\n")
    return man


# ---------------------------------------------------------------- leakage


def default_training_artefacts() -> list[Path]:
    """Every training artefact of the model module (curated SFT data, CuratorKIT export, synthetic train docs)."""
    paths = sorted(p for p in (ROOT / "data/curated").rglob("*") if p.is_file() and p.suffix in (".jsonl", ".json"))
    # real-paper training data; the removal log lists the excluded dev/test passages by design and is not training data
    paths += sorted(p for p in (ROOT / "data/real_train").rglob("*.jsonl") if not p.name.endswith("_removed.jsonl"))
    return paths + [ROOT / "data/mock/train_chunks.jsonl", ROOT / "data/mock/train_docs.json"]


def _ngrams(text: str) -> set[int]:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {hash(tuple(toks[i:i + NGRAM])) for i in range(len(toks) - NGRAM + 1)}


def leakage_report(manifest: dict, items: list[dict], chunks: dict[str, dict],
                   train_paths: list[Path] | None = None) -> dict[str, Any]:
    """Errors: any evaluation document, chunk id, question or 13-gram of evaluation-document text found in a
    training artefact; split/partition violations. Warnings: hazards in other teams' data for Stage 3."""
    errors = check_partition(manifest, chunks)
    eval_docs = set(manifest["splits"]["test"]) | set(manifest["splits"]["dev"])
    train_docs = set(manifest["splits"]["train"])
    for it in items:
        allowed = set(manifest["splits"][it["split"]])
        used = {x["document_id"] for x in it["evidence"]} | {s["document_id"] for s in it["support"]}
        if used - allowed:
            errors.append(f"{it['id']} ({it['split']}) uses documents outside its split: {sorted(used - allowed)}")
    eval_chunk_ids = {cid for cid, c in chunks.items() if c["source_file"] in eval_docs}
    eval_grams = set().union(*(_ngrams(c["text"]) for c in chunks.values() if c["source_file"] in eval_docs))
    eval_grams -= set().union(*(_ngrams(c["text"]) for c in chunks.values() if c["source_file"] in train_docs))
    questions = {norm(it["question"]).lower() for it in items}

    checked = []
    for path in train_paths if train_paths is not None else default_training_artefacts():
        text = path.read_text(errors="replace")
        low = norm(text).lower()
        hits = {
            "documents": sorted(d for d in eval_docs if d in text),
            "chunk_ids": sorted(c for c in eval_chunk_ids if c in text)[:5],
            "questions": sorted(q for q in questions if q in low)[:5],
            "ngram_overlaps": len(_ngrams(text) & eval_grams),
        }
        checked.append({"path": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path), **hits})
        if any(hits.values()):
            errors.append(f"LEAKAGE in training artefact {path}: {hits}")

    warnings = []
    for name, key in (("examples.jsonl", "context"), ("claim_probes.jsonl", "evidence")):
        rows = read_jsonl(GROUNDED_QA / name)
        train = [r for r in rows if r.get("split") == "train"]
        def docs_of(r):  # noqa: E306
            ev = r.get(key) or []
            ev = ev if isinstance(ev, list) else [ev]
            return {p.get("document_id") for p in ev + list(r.get("supporting_evidence") or [])}
        bad = [r["id"] for r in train if docs_of(r) & eval_docs]
        if bad:
            warnings.append(f"grounded_qa/{name}: {len(bad)} of {len(train)} train-split rows contain passages from "
                            f"dev/test documents (e.g. {bad[:3]}); drop those passages or rows before using it for "
                            "training in Stage 3")
    return {"errors": errors, "warnings": warnings, "checked_files": checked, "n_eval_ngrams": len(eval_grams)}


# ---------------------------------------------------------------- reporting


def abstention_metrics(rows: list[dict], mode: str) -> dict[str, Any]:
    """Abstention as a detector of unanswerable items: precision = abstentions that were on unanswerable
    items / all abstentions; recall = unanswerable items abstained on / all unanswerable items."""
    unans = [r for r in rows if not r["answerable"] and not r["ambiguous"]]
    abst = [r for r in rows if not r[mode]["answered"]]
    tp = sum(not r["answerable"] and not r["ambiguous"] for r in abst)
    p = tp / len(abst) if abst else None
    rec = tp / len(unans) if unans else None
    f1 = (2 * p * rec / (p + rec) if p + rec else 0.0) if p is not None and rec is not None else None
    rnd = lambda x: None if x is None else round(x, 4)  # noqa: E731
    return {"abstention_precision": rnd(p), "abstention_recall": rnd(rec), "abstention_f1": rnd(f1),
            "n_abstained": len(abst)}


REPORT_KEYS = ("grounded_accuracy", "answer_correct_rate", "wrong_answer_rate", "inappropriate_abstention_rate",
               "correct_abstention_rate", "hallucinated_answer_rate", "unsupported_claim_rate", "n_claims",
               "cites_gold_rate", "all_gold_cited_rate", "citation_precision", "uncited_answers", "n_answered")


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for k successes out of n."""
    if not n:
        return None
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def core_rates(rows: list[dict], answered: dict[str, bool], correct: dict[str, bool]) -> dict[str, Any]:
    """Headline rates with counts, denominators and Wilson intervals for one answer/abstain decision.
    answered[q]: the system gave an answer; correct[q]: that answer meets the item's criteria."""
    A = [r["query_id"] for r in rows if r["answerable"]]
    U = [r["query_id"] for r in rows if not r["answerable"] and not r["ambiguous"]]
    abst = [q for q in answered if not answered[q]]
    tp = sum(q in set(U) for q in abst)
    out = {
        "answer_correct": _rate(sum(answered[q] and correct[q] for q in A), len(A)),
        "wrong_answer": _rate(sum(answered[q] and not correct[q] for q in A), len(A)),
        "false_abstention": _rate(sum(not answered[q] for q in A), len(A)),
        "hallucinated_answer": _rate(sum(answered[q] for q in U), len(U)),
        "abstention_precision": _rate(tp, len(abst)),
        "abstention_recall": _rate(tp, len(U)),
    }
    p, r = out["abstention_precision"]["rate"], out["abstention_recall"]["rate"]
    out["abstention_f1"] = round(2 * p * r / (p + r), 4) if p and r else (0.0 if p is not None and r is not None else None)
    return out


def report(results_dir: Path, items_path: Path = RC / "items.jsonl") -> dict[str, Any]:
    """Real-corpus metrics for a run written by ps3_llm.evaluate, regenerated from its saved predictions.

    Decisions: serving and intent as in ps3_llm.metrics, plus serving_guarded = serving after the
    end-to-end pipeline's number guard (ps3_llm.pipeline.guard: answers stating a number absent from
    the cited passages are withheld)."""
    from .prompting import parse_model_output

    rows = read_jsonl(results_dir / "per_example.jsonl")
    preds = {p["query_id"]: p for p in read_jsonl(results_dir / "predictions.jsonl")}
    info = json.loads((results_dir / "run_info.json").read_text())
    data_path = ROOT / info["data"] if not Path(info["data"]).is_absolute() else Path(info["data"])
    data = {e["query_id"]: e for e in read_jsonl(data_path)}
    missing = [r["query_id"] for r in rows if r["query_id"] not in data or r["query_id"] not in preds]
    if missing or len(data) != len(rows):
        raise SystemExit(f"{results_dir} does not match {data_path}: missing {missing[:5]}")
    items = {it["id"]: it for it in read_jsonl(items_path)}
    st = {q: e.get("annotation_status", "unknown") for q, e in data.items()}
    counts = dict(Counter(st.values()))

    guarded: dict[str, bool] = {}
    for r in rows:
        q, ev = r["query_id"], {c["chunk_id"]: c["text"] for c in data[r["query_id"]]["evidence"]}
        parsed = parse_model_output(preds[q]["raw_output"], set(ev))
        cited = " ".join(ev[c] for c in parsed["citations"])
        guarded[q] = r["serving"]["answered"] and numbers(parsed["answer"]) <= numbers(cited)

    def reasons(rs: list[dict]) -> dict[str, int]:
        c: Counter = Counter()
        for r in rs:
            reason = preds[r["query_id"]].get("abstain_reason") or "answered"
            c[reason.split(":")[0]] += 1
        return dict(c)

    def block(rs: list[dict]) -> dict[str, Any]:
        agg = aggregate(rs)
        sub: dict[str, Any] = {"n": len(rs), "n_answerable": agg["n_answerable"], "n_unanswerable": agg["n_unanswerable"],
                               "serving_outcome": reasons(rs)}
        for m in MODES:
            ans = {r["query_id"]: r[m]["answered"] for r in rs}
            cor = {r["query_id"]: bool(r[m].get("correct")) for r in rs}
            sub[m] = {**{k: agg[m][k] for k in REPORT_KEYS}, **abstention_metrics(rs, m), "rates": core_rates(rs, ans, cor)}
        ans = {r["query_id"]: guarded[r["query_id"]] for r in rs}
        cor = {r["query_id"]: bool(r["serving"].get("correct")) for r in rs}
        sub["serving_guarded"] = {"withheld_by_guard": sum(r["serving"]["answered"] and not guarded[r["query_id"]] for r in rs),
                                  "rates": core_rates(rs, ans, cor)}
        return sub

    def groups(key) -> dict[str, Any]:
        g: dict[str, list] = defaultdict(list)
        for r in rows:
            g[key(r)].append(r)
        return {k: {"n": len(v), **{m: {"answered": sum(r[m]["answered"] for r in v),
                                        "grounded": sum(r[m]["grounded"] for r in v)} for m in MODES}}
                for k, v in sorted(g.items())}

    toks = [p.get("usage", {}).get("completion_tokens") for p in preds.values()]
    ptoks = [p.get("usage", {}).get("prompt_tokens") for p in preds.values()]
    summary = json.loads((results_dir / "summary.json").read_text())
    out: dict[str, Any] = {
        "results": results_dir.name, "data": info["data"], "data_sha256": info.get("data_sha256"),
        "adjudication_sha256": info.get("adjudication_sha256"), "adapter": info.get("adapter"),
        "adapter_args": info.get("adapter_args"), "models": sorted({p.get("model", "") for p in preds.values()}),
        "annotation_status_counts": counts,
        "labels_provisional": counts.get("human_verified", 0) != len(rows),
        "note": ("PROVISIONAL: metrics use machine-drafted labels that no human has verified. Do not report them as "
                 "ground-truth results." if counts.get("human_verified", 0) != len(rows)
                 else "All items are human-verified."),
        "inference_errors": sum(r["inference_error"] for r in rows),
        "json_valid_rate": summary.get("json_valid_rate"), "schema_valid_rate": summary.get("schema_valid_rate"),
        "invalid_citation_ids": summary.get("invalid_citation_ids"),
        "latency_p50_s": summary.get("latency_p50_s"), "latency_p95_s": summary.get("latency_p95_s"),
        "mean_completion_tokens": round(sum(t for t in toks if t) / max(1, sum(1 for t in toks if t)), 1),
        "mean_prompt_tokens": round(sum(t for t in ptoks if t) / max(1, sum(1 for t in ptoks if t)), 1),
        "subsets": {},
        "by_category": groups(lambda r: r["category"]),
        "by_document": groups(lambda r: items[r["query_id"]]["source_document"] or "out_of_corpus"),
    }
    for name, keep in (("all", lambda q: True), ("human_verified", lambda q: st[q] == "human_verified"),
                       ("machine_draft", lambda q: st[q] == "machine_draft")):
        rs = [r for r in rows if keep(r["query_id"])]
        out["subsets"][name] = block(rs) if rs else {"n": 0}
    return out


# ---------------------------------------------------------------- CLI


def _load_all(args) -> tuple[list[dict], dict, dict, list[dict]]:
    items = load_items(args.items)
    return items, load_manifest(args.manifest), load_chunks(), read_jsonl(args.reviews)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ps3_llm.real_corpus")
    ap.add_argument("--items", type=Path, default=RC / "items.jsonl")
    ap.add_argument("--manifest", type=Path, default=RC / "split_manifest.json")
    ap.add_argument("--reviews", type=Path, default=RC / "reviews.jsonl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="drafts.jsonl -> items.jsonl")
    b.add_argument("--drafts", type=Path, default=RC / "drafts.jsonl")
    sub.add_parser("validate", help="schema, provenance, review and leakage checks")
    sub.add_parser("status", help="annotation status counts")
    r = sub.add_parser("review", help="interactive human review")
    r.add_argument("--split", choices=["test", "dev"], required=True)
    r.add_argument("--reviewer", required=True)
    r.add_argument("--redo", action="store_true", help="also show items that already have a decision")
    rs = sub.add_parser("review-sheet", help="write a Markdown review sheet and a CSV to fill in")
    rs.add_argument("--split", choices=["test", "dev"], required=True)
    rs.add_argument("--out", type=Path, required=True, help="output path stem (.md and .csv are added)")
    ir = sub.add_parser("import-reviews", help="append decisions from a filled-in review CSV")
    ir.add_argument("csv", type=Path)
    c = sub.add_parser("convert", help="write the ps3_llm.evaluate data + adjudication files")
    c.add_argument("--split", choices=["test", "dev"], required=True)
    c.add_argument("--labels", choices=["verified", "provisional"], required=True)
    c.add_argument("--out-dir", type=Path, default=RC / "converted")
    f = sub.add_parser("freeze", help="freeze the human-verified labels of a split (refuses while any are unreviewed)")
    f.add_argument("--split", choices=["test", "dev"], required=True)
    f.add_argument("--name", required=True)
    lk = sub.add_parser("leakage", help="check that no training artefact contains evaluation documents")
    lk.add_argument("--train", type=Path, nargs="*", default=None, help="extra training files to check")
    rp = sub.add_parser("report", help="real-corpus metrics for a results directory written by ps3_llm.evaluate")
    rp.add_argument("results_dir", type=Path)
    rp.add_argument("--out", default="real_corpus_metrics.json", help="file name inside results_dir (never overwritten)")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        manifest, chunks = load_manifest(args.manifest), load_chunks()
        items = build_items(read_jsonl(args.drafts), chunks, manifest)
        errs = validate_all(items, chunks, manifest)
        if errs:
            print("\n".join(errs), file=sys.stderr)
            print(f"build FAILED: {len(errs)} errors; {args.items} not written", file=sys.stderr)
            return 1
        write_jsonl(args.items, items)
        print(f"wrote {len(items)} items -> {args.items}")
        return 0

    items, manifest, chunks, reviews = _load_all(args)
    status = effective_status(items, reviews)
    version = manifest["version"]
    if args.cmd == "validate":
        errs = validate_all(items, chunks, manifest, reviews)
        lk = leakage_report(manifest, items, chunks)
        errs += lk["errors"]
        for w in lk["warnings"]:
            print("WARNING:", w)
        print("\n".join(errs) if errs else f"OK: {len(items)} items, {len(reviews)} review records, "
                                           f"{len(lk['checked_files'])} training artefacts clean")
        return 1 if errs else 0
    if args.cmd == "status":
        table = Counter((it["split"], it["category"], status[it["id"]]["status"]) for it in items)
        for (s, cat, st), n in sorted(table.items()):
            print(f"{s:5} {cat:14} {st:15} {n}")
        print(f"total {len(items)}; by status: {dict(Counter(v['status'] for v in status.values()))}; "
              f"stale reviews: {sum(v['stale_reviews'] for v in status.values())}")
        return 0
    if args.cmd == "review":
        print(f"recorded {interactive_review(items, args.reviews, args.split, args.reviewer, args.redo)} decisions")
        return 0
    if args.cmd == "review-sheet":
        md, out_csv = write_review_sheet(items, status, args.split, args.out)
        print(f"wrote {md} and {out_csv}")
        return 0
    if args.cmd == "import-reviews":
        print(f"imported {import_reviews(args.csv, items, args.reviews)} decisions into {args.reviews}")
        return 0
    if args.cmd == "convert":
        examples, adj = convert(items, status, args.split, args.labels, version)
        if not examples:
            print(f"no items with labels={args.labels} in {args.split}; nothing written", file=sys.stderr)
            return 1
        data, adj_path = write_converted(examples, adj, args.out_dir, f"{args.split}_{args.labels}")
        print(json.dumps({"data": str(data), "data_sha256": sha256_file(data), "adjudication": str(adj_path),
                          "n": len(examples), "annotation_status": adj["_annotation_status"]}, indent=2))
        return 0
    if args.cmd == "freeze":
        print(f"froze {args.split} -> {freeze(items, status, args.split, args.name, version)}")
        return 0
    if args.cmd == "leakage":
        extra = args.train or []
        lk = leakage_report(manifest, items, chunks, default_training_artefacts() + extra)
        print(json.dumps(lk, indent=2))
        return 1 if lk["errors"] else 0
    if args.cmd == "report":
        out_path = args.results_dir / args.out
        if out_path.exists():
            raise SystemExit(f"refusing to overwrite {out_path}")
        rep = report(args.results_dir)
        out_path.write_text(json.dumps(rep, indent=2) + "\n")
        print(json.dumps(rep, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
