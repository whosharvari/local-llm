import json
from pathlib import Path

from ps3_llm.adapters import MockAdapter
from ps3_llm.evaluate import load_examples, run
from ps3_llm.prompting import build_messages, format_target, parse_model_output
from ps3_llm.schemas import EvidenceChunk, GroundedQuery

ROOT = Path(__file__).resolve().parent.parent
IDS = {"a#1", "a#2"}
ADJ = ROOT / "data/mock/heldout_adjudication.json"


def test_target_roundtrip():
    t = format_target("15 days.", ["a#1"], abstain=False)
    assert parse_model_output(t, IDS)["citations"] == ["a#1"]
    assert parse_model_output(format_target("", [], abstain=True), IDS)["abstained"]


def test_messages_include_ids_and_question():
    q = GroundedQuery("How long?", [EvidenceChunk("a#1", "Fifteen days.", source="doc.pdf", page=3)])
    msgs = build_messages(q)
    assert msgs[0]["role"] == "system" and "[a#1] (doc.pdf, p.3)" in msgs[1]["content"]


def test_mock_end_to_end(tmp_path):
    data = ROOT / "data/mock/heldout_eval.jsonl"
    examples = load_examples(data)
    assert len(examples) == 40
    s1 = run(MockAdapter(), examples, tmp_path / "a", data, {}, ADJ, measure_server=False)
    s2 = run(MockAdapter(), examples, tmp_path / "b", data, {}, ADJ, measure_server=False)
    for m in ("serving", "intent"):
        for k in ("grounded_accuracy", "hallucinated_answer_rate", "unsupported_claim_rate"):
            assert s1[m][k] == s2[m][k]  # deterministic
    preds = [json.loads(line) for line in (tmp_path / "a/predictions.jsonl").read_text().splitlines()]
    assert {"answer", "citations", "abstained", "warnings"} <= preds[0].keys()


def test_run_refuses_to_overwrite_results(tmp_path):
    import pytest

    data = ROOT / "data/mock/heldout_eval.jsonl"
    examples = load_examples(data)
    run(MockAdapter(), examples, tmp_path / "a", data, {}, ADJ, measure_server=False)
    with pytest.raises(SystemExit):
        run(MockAdapter(), examples, tmp_path / "a", data, {}, ADJ, measure_server=False)


def test_no_heldout_leakage_into_training_data():
    """Held-out documents, chunk ids, chunk text and questions must not appear in any training artefact."""
    from ps3_llm.mockdata import load_heldout_chunks

    held = load_heldout_chunks()
    held_questions = [json.loads(line)["question"] for line in (ROOT / "data/mock/heldout_eval.jsonl").open()]
    train_docs = json.loads((ROOT / "data/mock/train_docs.json").read_text())
    train_sources = {d["source"] for k, d in train_docs.items() if not k.startswith("_")}
    assert not train_sources & {c["source"] for c in held.values()}

    artefacts = [ROOT / "data/mock/train_chunks.jsonl"] + sorted((ROOT / "data/curated").glob("*.jsonl")) \
        + sorted((ROOT / "data/curated/curatorkit").glob("*.jsonl"))
    for path in artefacts:
        blob = path.read_text()
        for cid, c in held.items():
            assert cid not in blob and c["text"][:60] not in blob, (path.name, cid)
        for q in held_questions:
            assert q not in blob, (path.name, q)


def test_train_dev_split_is_by_document():
    rows = {s: [json.loads(line) for line in (ROOT / f"data/curated/sft_{s}.jsonl").open()] for s in ("train", "dev")}
    assert rows["train"] and rows["dev"]
    assert not {r["doc_id"] for r in rows["train"]} & {r["doc_id"] for r in rows["dev"]}
    for r in rows["train"] + rows["dev"]:
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["system", "user", "assistant"]
        target = json.loads(r["messages"][2]["content"])
        assert set(target) == {"abstain", "answer", "citations"}
        if r["kind"] == "answerable":
            assert target["citations"] == [r["gold_chunk"]] and f"[{r['gold_chunk']}]" in r["messages"][1]["content"]
        else:
            assert target["abstain"] and f"[{r['gold_chunk']}]" not in r["messages"][1]["content"]


def test_strict_ground_gate():
    from ps3_llm.curate import strict_ground

    src = "Fire evacuation drills take place twice a year."
    assert strict_ground("Drills take place twice a year, as stated in the provided text.", src)[0] == \
        "Drills take place twice a year."
    assert strict_ground("Drills happen monthly to keep staff safe.", src)[0] is None
    out, note = strict_ground("Drills take place twice a year. This keeps everyone safe and prepared.", src)
    assert out == "Drills take place twice a year." and note == "dropped_unsupported_sentences"
