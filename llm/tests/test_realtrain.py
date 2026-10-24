"""Leakage-free real-paper training data: document-level boundary, deterministic selection, example builders."""

import json

import pytest

from ps3_llm import real_corpus as rc
from ps3_llm import realtrain as rt

pytestmark = pytest.mark.skipif(not rc.CHUNKS.exists(), reason="corpus chunks not present")


@pytest.fixture(scope="module")
def env():
    return rc.load_manifest(), rc.load_chunks()


def _docs_of(row, chunks):
    return {chunks[p["passage_id"]]["source_file"] for p in rt._passages(row)}


def test_clean_grounded_qa_keeps_only_train_document_passages(env):
    manifest, chunks = env
    kept, removed = rt.clean_grounded_qa(manifest, chunks)
    train = set(manifest["splits"]["train"])
    assert kept and all(r["split"] == "train" and _docs_of(r, chunks) <= train for r in kept)
    leaky = [r for r in removed if not r["reason"].startswith("grounded_qa split")]
    assert len(leaky) == 42 and len(kept) + len(leaky) == 104
    assert all("test document" in r["reason"] or "dev document" in r["reason"] for r in leaky)


def test_planted_cross_boundary_row_is_removed(env, tmp_path, monkeypatch):
    manifest, chunks = env
    c = chunks["gpt3_few_shot_learners.pdf::0047"]
    row = {"id": "planted", "split": "train", "example_type": "answerable", "question": "q?", "supporting_evidence": [],
           "context": [{"passage_id": c["id"], "document_id": c["source_file"], "page_number": c["page_number"],
                        "text": c["text"]}]}
    forged = dict(row, id="forged", context=[dict(row["context"][0], document_id="bert.pdf")])
    (tmp_path / "examples.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(forged) + "\n")
    monkeypatch.setattr(rc, "GROUNDED_QA", tmp_path)
    kept, removed = rt.clean_grounded_qa(manifest, chunks)
    assert kept == [] and {r["id"] for r in removed} == {"planted", "forged"}   # forged document_id is caught too


def test_chunk_selection_is_deterministic_train_only_and_non_adjacent(env):
    manifest, chunks = env
    a, _ = rt.select_chunks(manifest, chunks)
    b, _ = rt.select_chunks(manifest, chunks)
    assert a == b
    assert {r["doc_id"] for r in a} <= set(manifest["splits"]["train"])
    assert {r["role"] for r in a if r["doc_id"] == rt.VALIDATION_DOC} == {"validation"}
    by_doc = {}
    for r in a:
        by_doc.setdefault(r["doc_id"], []).append(chunks[r["chunk_id"]]["chunk_index"])
    assert all(all(abs(x - y) > 1 for x in v for y in v if x != y) for v in by_doc.values())
    assert all(rt.chunk_quality(chunks[r["chunk_id"]]) == "" for r in a)


def test_example_builder_respects_boundaries_and_targets(env):
    manifest, chunks = env
    gold = chunks["bert.pdf::0047"]
    qa = [{"question": "How many sentence-pair completion examples does SWAG contain?",
           "answer": "SWAG contains 113k sentence-pair completion examples.", "chunk_id": gold["id"]}] * 20
    ex = rt.build_examples(qa, [], chunks, manifest, seed=1)
    train = set(manifest["splits"]["train"])
    near = {f"bert.pdf::{gold['chunk_index'] + o:04d}" for o in (-1, 0, 1)}
    kinds = {e["kind"] for e in ex}
    assert {"answerable", "abstain_hard_negative"} <= kinds
    for e in ex:
        assert {chunks[i]["source_file"] for i in e["evidence_ids"]} <= train
        target = json.loads(e["messages"][2]["content"])
        if e["kind"].startswith("abstain"):
            assert target == {"abstain": True, "answer": "", "citations": []} and not near & set(e["evidence_ids"])
        else:
            assert target["citations"] == [gold["id"]] and gold["id"] in e["evidence_ids"]
            assert len(e["evidence_ids"]) == rt.K


def test_counterfactual_changes_the_number_consistently():
    import random
    out = rt._counterfactual("It has 12 layers.", "The encoder has 12 layers and 768 units.", random.Random(0))
    assert out is not None
    new_answer, new_text = out
    n = [w for w in new_answer.split() if w.isdigit()][0]
    assert n != "12" and f"{n} layers" in new_text and "768" in new_text


@pytest.mark.skipif(not (rt.OUT / "sft_train.jsonl").exists(), reason="real-paper SFT data not built")
def test_built_sft_files_never_cross_the_document_boundary(env):
    manifest, chunks = env
    train = set(manifest["splits"]["train"])
    for name in ("train", "validation"):
        rows = rc.read_jsonl(rt.OUT / f"sft_{name}.jsonl")
        assert rows
        for r in rows:
            assert {chunks[i]["source_file"] for i in r["evidence_ids"]} <= train
            assert (r["doc_id"] == rt.VALIDATION_DOC) == (name == "validation") or r["kind"] == "abstain_out_of_corpus"
    man = json.loads((rt.OUT / "manifest.json").read_text())
    assert man["leakage"]["errors"] == []
    for rel, sha in man["files"].items():
        assert rc.sha256_file(rt.OUT / rel) == sha, rel
