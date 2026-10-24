"""End-to-end pipeline (retrieval -> prompt -> model -> checked answer) with a tiny index and scripted models."""

import json
import re
import struct

import numpy as np
import pytest

from ps3_llm import pipeline
from ps3_llm.adapters import SGLangAdapter, answer_from_text
from ps3_llm.prompting import format_target
from ps3_llm.retrieval import INDEX, METADATA, Gate, Retriever, coverage, read_flat_ip_index

DOCS = [
    ("bert.pdf", 3, "BERT-base has 12 Transformer layers, a hidden size of 768 and 12 attention heads."),
    ("bert.pdf", 4, "BERT is pre-trained with masked language modelling and next sentence prediction."),
    ("lora.pdf", 2, "LoRA freezes the pretrained weights and trains low-rank matrices of rank r = 4."),
]


class BagEncoder:
    """Deterministic hashed bag-of-words embedding (no model download)."""
    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            v = np.zeros(64, dtype="float32")
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[sum(map(ord, w)) % 64] += 1
            out.append(v / (np.linalg.norm(v) or 1))
        return np.array(out)


def _write_index(path, vectors):
    n, d = vectors.shape
    path.write_bytes(b"IxFI" + struct.pack("<iqqqBiQ", d, n, 1 << 20, 1 << 20, 1, 0, n * d)
                     + vectors.astype("<f4").tobytes())


@pytest.fixture()
def retriever(tmp_path):
    meta = [{"id": f"{d}::{i:04d}", "text": t, "source_file": d, "page_number": p, "chunk_index": i}
            for i, (d, p, t) in enumerate(DOCS)]
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    _write_index(tmp_path / "index.faiss", BagEncoder().encode([m["text"] for m in meta]))
    return Retriever(tmp_path / "index.faiss", tmp_path / "meta.json", encoder=BagEncoder())


class Scripted:
    """Returns a fixed raw output and records the prompts it was given."""
    name = "scripted"

    def __init__(self, raw):
        self.raw, self.queries = raw, []

    def answer_batch(self, queries):
        self.queries += queries
        raw = self.raw(queries[0]) if callable(self.raw) else self.raw
        return [answer_from_text(raw, q, self.name, 0.0, {}) for q in queries]


def test_supported_answer_keeps_ids_pages_and_citations(retriever):
    m = Scripted(lambda q: format_target("BERT-base has 12 layers.", [q.evidence[0].chunk_id], False))
    rec = pipeline.ask("How many layers does BERT-base have?", retriever, m, k=2)
    assert rec["retrieval"][0] == {"chunk_id": "bert.pdf::0000", "document_id": "bert.pdf", "page_number": 3,
                                   "score": rec["retrieval"][0]["score"]}
    r = rec["response"]
    assert not r["abstained"] and r["citations"] == ["bert.pdf::0000"]
    assert "bert.pdf p.3" in pipeline.render(rec)
    prompt = m.queries[0]
    assert [c.chunk_id for c in prompt.evidence] == [h["chunk_id"] for h in rec["retrieval"]]


def test_unanswerable_question_returns_clear_abstention(retriever):
    rec = pipeline.ask("What GPU was BERT trained on?", retriever, Scripted(format_target("", [], True)))
    assert rec["response"]["abstained"] and rec["response"]["abstain_reason"] == "model_abstained"
    assert pipeline.render(rec).startswith("ABSTAINED")


def test_conflicting_number_not_in_cited_passage_is_withheld(retriever):
    m = Scripted(lambda q: format_target("BERT-base has 24 layers.", [q.evidence[0].chunk_id], False))
    r = pipeline.ask("How many layers does BERT-base have?", retriever, m)["response"]
    assert r["abstained"] and r["abstain_reason"] == pipeline.REASON_NUMBERS
    assert any("unsupported_numbers" in w for w in r["warnings"])


def test_invalid_citation_ids_are_never_returned(retriever):
    m = Scripted(format_target("BERT-base has 12 layers.", ["made_up.pdf::0001"], False))
    r = pipeline.ask("How many layers does BERT-base have?", retriever, m)["response"]
    assert r["abstained"] and r["abstain_reason"] == "no_valid_citation" and r["citations"] == []
    m = Scripted(lambda q: format_target("12 layers.", [q.evidence[0].chunk_id, "made_up.pdf::0001"], False))
    r = pipeline.ask("How many layers does BERT-base have?", retriever, m)["response"]
    assert r["citations"] == ["bert.pdf::0000"] and "invalid_citation_ids" in r["warnings"]


def test_malformed_model_output_becomes_abstention(retriever):
    for raw in ("The answer is 12", '{"abstain": false, "answer": "12 layers", "citations": [', ""):
        r = pipeline.ask("How many layers does BERT-base have?", retriever, Scripted(raw))["response"]
        assert r["abstained"] and r["answer"] == ""


def test_retrieval_failure_and_server_unavailable(retriever, tmp_path):
    class Broken:
        def search(self, q, k):
            raise FileNotFoundError("index missing")
    m = Scripted(format_target("x", [], False))
    rec = pipeline.ask("q?", Broken(), m)
    assert rec["response"]["abstain_reason"].startswith(pipeline.REASON_RETRIEVAL) and m.queries == []
    down = SGLangAdapter(base_url="http://127.0.0.1:9/v1", timeout=2)
    r = pipeline.ask("How many layers does BERT-base have?", retriever, down)["response"]
    assert r["abstained"] and r["abstain_reason"].startswith("inference_error")


def test_gate_abstains_without_calling_the_model(retriever):
    never = Gate(weights=[1.0, 1.0], bias=-100.0, mean=[0, 0], std=[1, 1], threshold=0.5)
    m = Scripted(format_target("x", [], False))
    rec = pipeline.ask("How many layers does BERT-base have?", retriever, m, gate=never)
    assert rec["response"]["abstain_reason"] == pipeline.REASON_GATE and m.queries == [] and rec["gate"]["probability"] < 0.5


def test_no_evaluation_labels_reach_the_prompt(retriever):
    m = Scripted(format_target("", [], True))
    pipeline.ask("How many layers does BERT-base have?", retriever, m)
    q = m.queries[0]
    assert set(vars(q)) == {"question", "evidence", "query_id"}
    assert all(set(vars(c)) == {"chunk_id", "text", "source", "page", "score"} for c in q.evidence)


def test_coverage_feature():
    assert coverage("How many layers does BERT have?", "BERT has many layers") == 1.0
    assert coverage("What is ELECTRA?", "BERT has 12 layers") == 0.0


@pytest.mark.skipif(not INDEX.exists(), reason="vector store not present")
def test_real_vector_store_is_row_aligned_with_chunks():
    from ps3_llm import real_corpus as rc
    vecs = read_flat_ip_index(INDEX)
    meta = json.loads(METADATA.read_text())
    chunks = rc.load_chunks()
    assert vecs.shape == (len(chunks), 384) and np.allclose(np.linalg.norm(vecs[:50], axis=1), 1, atol=1e-4)
    assert all(m["id"] in chunks and chunks[m["id"]]["text"] == m["text"] for m in meta)
