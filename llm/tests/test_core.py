import json
from pathlib import Path

from ps3_llm.adapters import MockAdapter
from ps3_llm.evaluate import load_examples, run
from ps3_llm.metrics import aggregate, answer_matches, claim_supported, score_example
from ps3_llm.prompting import build_messages, format_target, parse_model_output
from ps3_llm.schemas import EvalExample, EvidenceChunk, GroundedAnswer, GroundedQuery

ROOT = Path(__file__).resolve().parent.parent
IDS = {"a#1", "a#2"}


def test_parse_clean_json():
    out = parse_model_output('{"abstain": false, "answer": "15 days.", "citations": ["a#1"]}', IDS)
    assert out == {"answer": "15 days.", "citations": ["a#1"], "abstained": False, "parse_ok": True,
                   "invalid_citations": []}


def test_parse_fenced_json_drops_invented_ids():
    text = '```json\n{"abstain": false, "answer": "x", "citations": ["a#1", "z#9"]}\n```'
    out = parse_model_output(text, IDS)
    assert out["citations"] == ["a#1"] and out["invalid_citations"] == ["z#9"] and out["parse_ok"]


def test_parse_abstain_clears_answer_and_citations():
    out = parse_model_output('{"abstain": true, "answer": "maybe 5", "citations": ["a#1"]}', IDS)
    assert out["abstained"] and out["answer"] == "" and out["citations"] == []


def test_parse_fallback_prose():
    out = parse_model_output("The documents do not contain this information.", IDS)
    assert out["abstained"] and not out["parse_ok"]
    out = parse_model_output("It is 15 days [a#2].", IDS)
    assert not out["abstained"] and out["citations"] == ["a#2"] and not out["parse_ok"]


def test_target_roundtrip():
    t = format_target("15 days.", ["a#1"], abstain=False)
    assert parse_model_output(t, IDS)["citations"] == ["a#1"]
    assert parse_model_output(format_target("", [], abstain=True), IDS)["abstained"]


def test_messages_include_ids_and_question():
    q = GroundedQuery("How long?", [EvidenceChunk("a#1", "Fifteen days.", source="doc.pdf", page=3)])
    msgs = build_messages(q)
    assert msgs[0]["role"] == "system" and "[a#1] (doc.pdf, p.3)" in msgs[1]["content"]


def test_answer_matching_requires_numbers():
    assert answer_matches("At least 15 business days.", "At least 15 business days.")
    assert not answer_matches("At least 10 business days.", "At least 15 business days.")


def test_claim_support():
    ev = "The tender must remain open for a minimum of 15 business days."
    assert claim_supported("The tender must remain open 15 business days.", ev)
    assert not claim_supported("The tender must remain open 20 business days.", ev)


def _ex(answerable=True):
    q = GroundedQuery("How long must the tender stay open?",
                      [EvidenceChunk("a#1", "The tender must remain open for a minimum of 15 business days.")],
                      query_id="t1")
    return EvalExample(q, answerable, "15 business days." if answerable else "", ["a#1"] if answerable else [],
                       "answerable" if answerable else "distractor")


def test_score_hallucinated_answer_on_unanswerable():
    r = score_example(_ex(False), GroundedAnswer("20 days.", ["a#1"], False))
    assert r["hallucinated_answer"] and not r["grounded_correct"]
    assert r["n_unsupported_claims"] == 1 and r["unsupported_numbers"] == ["20"]


def test_score_grounded_correct():
    r = score_example(_ex(True), GroundedAnswer("15 business days.", ["a#1"], False))
    assert r["grounded_correct"] and r["citation_precision"] == 1.0
    s = aggregate([r])
    assert s["grounded_accuracy"] == 1.0 and s["unsupported_claim_rate"] == 0.0


def test_mock_end_to_end(tmp_path):
    data = ROOT / "data/mock/heldout_eval.jsonl"
    examples = load_examples(data)
    assert len(examples) == 40
    s1 = run(MockAdapter(), examples, tmp_path / "a", data, {})
    s2 = run(MockAdapter(), examples, tmp_path / "b", data, {})
    for k in ("grounded_accuracy", "hallucinated_answer_rate", "unsupported_claim_rate"):
        assert s1[k] == s2[k]  # deterministic
    preds = [json.loads(line) for line in (tmp_path / "a/predictions.jsonl").read_text().splitlines()]
    assert {"answer", "citations", "abstained"} <= preds[0].keys()


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


def test_cli_contract_roundtrip():
    import subprocess
    import sys

    req = (ROOT / "data/mock/example_request.json").read_text()
    out = subprocess.run([sys.executable, "-m", "ps3_llm.cli", "--adapter", "mock"], input=req, cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    ans = json.loads(out)
    ids = {e["chunk_id"] for e in json.loads(req)["evidence"]}
    assert ans["query_id"] == "demo-1" and set(ans["citations"]) <= ids
    assert ans["abstained"] is False and "15 business days" in ans["answer"]
    empty = json.dumps({"query_id": "demo-2", "question": "What is the dress code?", "evidence": []})
    ans2 = json.loads(subprocess.run([sys.executable, "-m", "ps3_llm.cli"], input=empty, cwd=ROOT,
                                     capture_output=True, text=True, check=True).stdout)
    assert ans2["abstained"] and ans2["answer"] == "" and ans2["citations"] == []
