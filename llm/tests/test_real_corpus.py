"""Real-corpus evaluation set: schema, provenance, annotation status, document-level leakage and conversion
into the ps3_llm.evaluate format."""

import copy
import csv
import json
from pathlib import Path

import pytest

from ps3_llm import compare
from ps3_llm import real_corpus as rc
from ps3_llm.adapters import answer_from_text
from ps3_llm.evaluate import load_examples, run
from ps3_llm.prompting import format_target

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not rc.CHUNKS.exists(), reason="corpus chunks (../data/processed) not present")
REVIEWER = "Test Reviewer"


@pytest.fixture(scope="module")
def corpus():
    return rc.load_items(), rc.load_chunks(), rc.load_manifest()


def _mutated(item, chunks, manifest, items, change):
    it = copy.deepcopy(item)
    change(it)
    it["label_sha256"] = rc.label_hash(it)
    return rc.validate_item(it, chunks, manifest, {i["id"]: i for i in items}, [c["text"] for c in chunks.values()])


# ---------------------------------------------------------------- schema


def test_items_are_reproducible_from_drafts(corpus):
    items, chunks, manifest = corpus
    assert rc.build_items(rc.read_jsonl(rc.RC / "drafts.jsonl"), chunks, manifest) == items


def test_all_items_pass_schema_provenance_and_review_validation(corpus):
    items, chunks, manifest = corpus
    assert rc.validate_all(items, chunks, manifest, rc.read_jsonl(rc.RC / "reviews.jsonl")) == []


def test_schema_file_matches_the_validator():
    schema = json.loads((rc.RC / "schema.json").read_text())
    assert schema["required"] == list(rc.ITEM_KEYS)
    assert set(schema["properties"]["category"]["enum"]) == set(rc.CATEGORIES)


def test_composition_covers_every_category_in_both_splits(corpus):
    items, _, _ = corpus
    for split in ("test", "dev"):
        cats = {it["category"] for it in items if it["split"] == split}
        assert cats == set(rc.CATEGORIES), split
    assert len({it["id"] for it in items}) == len(items) == 106
    assert sum(it["split"] == "test" for it in items) == 72


# ---------------------------------------------------------------- provenance


def test_every_label_is_traceable_to_verbatim_source_text(corpus):
    items, chunks, _ = corpus
    for it in items:
        for s in it["support"]:
            c = chunks[s["chunk_id"]]
            assert rc.norm(s["quote"]) in rc.norm(c["text"]), (it["id"], s["chunk_id"])
            assert (s["document_id"], s["page_number"]) == (c["source_file"], c["page_number"])
        for x in it["evidence"]:
            assert x["text"] == chunks[x["chunk_id"]]["text"]
        if it["answerable"]:
            assert set(it["gold_citations"]) >= {s["chunk_id"] for s in it["support"]}


def test_validator_rejects_fabricated_or_inconsistent_labels(corpus):
    items, chunks, manifest = corpus
    by = {i["id"]: i for i in items}
    cases = {
        "fabricated quote": ("rct-rlhf-11", lambda it: it["support"][0].update(quote="trained using 9000 queries")),
        "answer contradicts facts": ("rct-rlhf-11", lambda it: it.update(reference_answer="About 1000 queries.")),
        "wrong page": ("rct-rlhf-11", lambda it: it["support"][0].update(page_number=1)),
        "edited evidence text": ("rct-gpt3-12", lambda it: it["evidence"][0].update(text="nctx = 4096 tokens.")),
        "answer in unanswerable evidence": ("rct-gpt3-25", lambda it: it.update(absent_terms=["table"])),
        "unanswerable with reference": ("rct-rlhf-22", lambda it: it.update(reference_answer="0.0007")),
        "absent term in corpus": ("rct-ooc-06", lambda it: it.update(absent_terms=["transformer"])),
        "swap evidence from source doc": ("rct-swap-01", lambda it: it["evidence"].__setitem__(
            0, dict(by["rct-gpt3-11"]["evidence"][0]))),
    }
    for name, (iid, change) in cases.items():
        assert _mutated(by[iid], chunks, manifest, items, change), name
    stale = copy.deepcopy(by["rct-rlhf-11"])
    stale["reference_answer"] = "900 queries."
    assert any("label_sha256" in e for e in rc.validate_item(stale, chunks, manifest, by))


# ---------------------------------------------------------------- annotation status / human review


def test_labels_are_machine_drafts_until_a_named_human_accepts_them(corpus):
    items, _, _ = corpus
    assert {v["status"] for v in rc.effective_status(items, rc.read_jsonl(rc.RC / "reviews.jsonl")).values()} \
        <= set(rc.STATUSES)
    assert {v["status"] for v in rc.effective_status(items, []).values()} == {"machine_draft"}
    it = items[0]
    rec = {"item_id": it["id"], "label_sha256": it["label_sha256"], "decision": "accept", "reviewer": REVIEWER}
    assert rc.effective_status(items, [rec])[it["id"]]["status"] == "human_verified"
    for bad in ("Claude", "gpt-4o judge", "auto", ""):
        assert rc.effective_status(items, [{**rec, "reviewer": bad}])[it["id"]]["status"] == "machine_draft"
    assert rc.validate_review({**rec, "decision": "reject"}, {it["id"]: it})  # rejection needs notes
    edited = copy.deepcopy(items)
    edited[0]["reference_answer"] += " (edited)"
    edited[0]["label_sha256"] = rc.label_hash(edited[0])
    st = rc.effective_status(edited, [rec])[it["id"]]
    assert st["status"] == "machine_draft" and st["stale_reviews"] == 1


def test_review_sheet_import_and_freeze(corpus, tmp_path):
    items, _, _ = corpus
    md, sheet = rc.write_review_sheet(items, rc.effective_status(items, []), "dev", tmp_path / "dev_review")
    assert "### rcd-rag-01" in md.read_text() and "[GOLD]" in md.read_text()
    rows = list(csv.DictReader(sheet.open()))
    reviews = tmp_path / "reviews.jsonl"
    for r in rows:
        r.update(decision="accept", reviewer=REVIEWER)
    rows[0].update(decision="reject", notes="test rejection")
    half = tmp_path / "half.csv"
    with half.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows[:10])
    assert rc.import_reviews(half, items, reviews) == 10
    status = rc.effective_status(items, rc.read_jsonl(reviews))
    with pytest.raises(SystemExit, match="not human-reviewed"):
        rc.freeze(items, status, "dev", "dev-test", "x", tmp_path / "frozen")
    rest = tmp_path / "rest.csv"
    with rest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows[10:])
    rc.import_reviews(rest, items, reviews)
    status = rc.effective_status(items, rc.read_jsonl(reviews))
    man = json.loads(rc.freeze(items, status, "dev", "dev-test", "x", tmp_path / "frozen").read_text())
    assert man["n_items"] == 33 and man["n_rejected"] == 1 and rows[0]["item_id"] not in man["items"]
    frozen = load_examples(tmp_path / "frozen" / man["data"])
    assert {e.query.query_id for e in frozen} == set(man["items"])
    with pytest.raises(SystemExit):
        rc.freeze(items, status, "dev", "dev-test", "x", tmp_path / "frozen")
    bad = tmp_path / "bad.csv"
    bad.write_text(f"item_id,label_sha256,decision,reviewer,notes\n{rows[1]['item_id']},{'0' * 64},accept,{REVIEWER},\n")
    with pytest.raises(SystemExit, match="label changed"):
        rc.import_reviews(bad, items, reviews)


# ---------------------------------------------------------------- leakage


def test_split_is_by_document_and_matches_member1_split(corpus):
    items, chunks, manifest = corpus
    assert rc.check_partition(manifest, chunks) == []
    for it in items:
        docs = {x["document_id"] for x in it["evidence"]} | {s["document_id"] for s in it["support"]}
        assert docs <= set(manifest["splits"][it["split"]]), it["id"]
    seeds = rc.read_jsonl(rc.GROUNDED_QA / "examples.jsonl")
    for split in ("test", "dev"):
        member1 = {r["source_document"] for r in seeds if r["split"] == split and r["answerable"]}
        assert member1 == set(manifest["splits"][split])
    assert rc.sha256_file(rc.CHUNKS) == manifest["source"]["chunks_sha256"]


def test_no_training_artefact_contains_evaluation_documents(corpus):
    items, chunks, manifest = corpus
    rep = rc.leakage_report(manifest, items, chunks)
    assert rep["errors"] == [] and len(rep["checked_files"]) >= 10
    assert all(not any(v for k, v in f.items() if k != "path") for f in rep["checked_files"])


def test_leakage_detector_catches_planted_leaks(corpus, tmp_path):
    items, chunks, manifest = corpus
    test_chunk = chunks["gpt3_few_shot_learners.pdf::0047"]["text"]
    train_chunk = chunks["bert.pdf::0010"]["text"]
    planted = {
        "ngram.jsonl": json.dumps({"text": " ".join(test_chunk.split()[5:40])}),
        "question.jsonl": json.dumps({"q": items[0]["question"]}),
        "docname.jsonl": json.dumps({"source": "rlhf_human_preferences.pdf"}),
    }
    for name, content in planted.items():
        (tmp_path / name).write_text(content + "\n")
        rep = rc.leakage_report(manifest, items, chunks, [tmp_path / name])
        assert rep["errors"], name
    (tmp_path / "clean.jsonl").write_text(json.dumps({"text": train_chunk}) + "\n")
    assert rc.leakage_report(manifest, items, chunks, [tmp_path / "clean.jsonl"])["errors"] == []


# ---------------------------------------------------------------- conversion / evaluation


class _Oracle:
    """Answers with the reference answer and its gold citations; abstains on unanswerable items."""
    name = "oracle"

    def __init__(self, examples):
        self.ex = {e.query.query_id: e for e in examples}

    def answer_batch(self, queries):
        out = []
        for q in queries:
            e = self.ex[q.query_id]
            raw = format_target(e.gold_answer, e.gold_citations, abstain=not e.answerable)
            out.append(answer_from_text(raw, q, self.name, 0.0, {}))
        return out


class _AlwaysAnswer:
    name = "always-answer"

    def answer_batch(self, queries):
        return [answer_from_text(format_target("It is 42.", [q.evidence[0].chunk_id], abstain=False), q,
                                 self.name, 0.0, {}) for q in queries]


def test_converted_files_are_current_and_verified_split_is_empty(corpus):
    items, _, manifest = corpus
    status = rc.effective_status(items, rc.read_jsonl(rc.RC / "reviews.jsonl"))
    for split in ("test", "dev"):
        ex, adj = rc.convert(items, status, split, "provisional", manifest["version"])
        assert rc.read_jsonl(rc.RC / f"converted/{split}_provisional.jsonl") == ex
        assert json.loads((rc.RC / f"converted/{split}_provisional_adjudication.json").read_text()) == adj
        assert set(adj["items"]) == {e["query_id"] for e in ex if e["answerable"]}
    if all(v["status"] == "machine_draft" for v in status.values()):
        assert rc.convert(items, status, "test", "verified", manifest["version"])[0] == []


@pytest.mark.parametrize("split", ["test", "dev"])
def test_evaluation_adapter_scores_oracle_and_always_answer(split, tmp_path):
    data = rc.RC / f"converted/{split}_provisional.jsonl"
    adj = rc.RC / f"converted/{split}_provisional_adjudication.json"
    examples = load_examples(data)
    s = run(_Oracle(examples), examples, tmp_path / "oracle", data, {}, adj, measure_server=False)
    for m in ("serving", "intent"):
        assert s[m]["answer_correct_rate"] == 1.0 and s[m]["hallucinated_answer_rate"] == 0.0
        assert s[m]["cites_gold_rate"] == 1.0 and s[m]["grounded_accuracy"] == 1.0
    rep = rc.report(tmp_path / "oracle")
    assert rep["labels_provisional"] and "PROVISIONAL" in rep["note"]
    allm = rep["subsets"]["all"]["serving"]
    assert allm["abstention_precision"] == allm["abstention_recall"] == 1.0
    assert rep["subsets"]["human_verified"] == {"n": 0}

    s = run(_AlwaysAnswer(), examples, tmp_path / "always", data, {}, adj, measure_server=False)
    assert s["serving"]["hallucinated_answer_rate"] == 1.0 and s["serving"]["answer_correct_rate"] == 0.0
    rep = rc.report(tmp_path / "always")["subsets"]["all"]["serving"]
    assert rep["abstention_recall"] == 0.0 and rep["abstention_precision"] is None
    assert examples[0].query.question in compare.main("oracle", "always", tmp_path / "cmp.md", tmp_path).read_text()


def test_compare_refuses_runs_scored_on_different_data(tmp_path):
    data = rc.RC / "converted/dev_provisional.jsonl"
    adj = rc.RC / "converted/dev_provisional_adjudication.json"
    relabelled = tmp_path / "relabelled.jsonl"
    rows = rc.read_jsonl(data)
    rows[0]["gold_answer"] += " (changed)"
    rc.write_jsonl(relabelled, rows)
    run(_AlwaysAnswer(), load_examples(data), tmp_path / "a", data, {}, adj, measure_server=False)
    run(_AlwaysAnswer(), load_examples(relabelled), tmp_path / "b", relabelled, {}, adj, measure_server=False)
    with pytest.raises(SystemExit, match="different data files"):
        compare.main("a", "b", tmp_path / "cmp.md", tmp_path)


def test_changing_a_required_number_makes_the_reference_answer_fail(corpus):
    """Acceptable variations must not reward a wrong number: perturb the first number of each numeric
    required fact where it occurs in the reference answer; the perturbed answer must fail the criteria."""
    import re

    from ps3_llm.metrics import facts_match

    items, _, _ = corpus
    numeric = re.compile(r"^[\d.,%×x^\-−e ]+(?:\s*(?:billion|million|gb|m|b|k|seconds|hours|tokens|layers?))?$")
    probed = 0
    for it in items:
        if not it["answerable"]:
            continue
        ref = it["reference_answer"].replace(",", "")
        for fact in it["required_facts"]:
            hit = next((a for a in fact if numeric.match(a.lower()) and a.lower() in ref.lower()), None)
            if hit is None:
                continue
            num = re.search(r"\d+(?:\.\d+)?", hit).group(0)
            wrong = str(int(float(num) * 3 + 7)) if "." not in num else f"{float(num) * 3 + 0.7:.2f}"
            pert = re.sub(re.escape(hit), hit.replace(num, wrong, 1), ref, flags=re.IGNORECASE)  # every occurrence
            probed += 1
            assert not facts_match(pert, it["required_facts"], it["reject"]), (it["id"], pert)
    assert probed >= 40
