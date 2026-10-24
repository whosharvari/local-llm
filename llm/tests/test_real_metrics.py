"""Real-corpus report metrics on hand-constructed examples with hand-computed expected values."""

import json

import pytest

from ps3_llm import real_corpus as rc
from ps3_llm.adapters import answer_from_text
from ps3_llm.evaluate import load_examples, run
from ps3_llm.prompting import format_target

EV = [{"chunk_id": "d.pdf::0001", "text": "The model has 42 layers and 7 heads.", "source": "d.pdf", "page": 1},
      {"chunk_id": "d.pdf::0002", "text": "Training used Adam.", "source": "d.pdf", "page": 2}]
ITEMS = [  # id, answerable, facts, scripted raw model output
    ("a1", True, [["42"]], format_target("It has 42 layers.", ["d.pdf::0001"], False)),
    ("a2", True, [["7"]], format_target("", [], True)),
    ("u1", False, [], format_target("It used 3 GPUs.", ["d.pdf::0002"], False)),
    ("u2", False, [], format_target("", [], True)),
    ("u3", False, [], "not json at all"),
    ("a3", True, [["adam"]], format_target("Adam.", ["fake::9999"], False)),
]


class Scripted:
    name = "scripted"

    def answer_batch(self, queries):
        raw = {i[0]: i[3] for i in ITEMS}
        return [answer_from_text(raw[q.query_id], q, self.name, 0.1, {"completion_tokens": 5, "prompt_tokens": 50})
                for q in queries]


@pytest.fixture()
def scored(tmp_path):
    data, adj, items = tmp_path / "d.jsonl", tmp_path / "adj.json", tmp_path / "items.jsonl"
    rc.write_jsonl(data, [{"query_id": q, "question": f"{q}?", "evidence": EV, "answerable": a, "gold_answer": "x" if a else "",
                           "gold_citations": ["d.pdf::0001"] if a else [], "category": "numerical" if a else "unanswerable",
                           "annotation_status": "machine_draft"} for q, a, _, _ in ITEMS])
    adj.write_text(json.dumps({"items": {q: {"facts": f} for q, a, f, _ in ITEMS if a}}))
    rc.write_jsonl(items, [{"id": q, "source_document": "d.pdf"} for q, *_ in ITEMS])
    run(Scripted(), load_examples(data), tmp_path / "run", data, {}, adj, measure_server=False)
    return rc.report(tmp_path / "run", items)


def test_rates_denominators_and_reasons(scored):
    s = scored["subsets"]["all"]
    r = s["serving"]["rates"]
    # answerable: a1 correct, a2 abstained, a3 cited only an invented id -> abstained (no_valid_citation)
    assert (r["answer_correct"]["k"], r["answer_correct"]["n"]) == (1, 3)
    assert (r["false_abstention"]["k"], r["false_abstention"]["n"]) == (2, 3)
    # unanswerable: u1 answered (hallucinated), u2 abstained, u3 unparseable -> abstained
    assert (r["hallucinated_answer"]["k"], r["hallucinated_answer"]["n"]) == (1, 3)
    # abstentions: a2, a3, u2, u3 -> 2 on unanswerable items
    assert (r["abstention_precision"]["k"], r["abstention_precision"]["n"]) == (2, 4)
    assert (r["abstention_recall"]["k"], r["abstention_recall"]["n"]) == (2, 3)
    assert r["abstention_f1"] == round(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3), 4)
    assert s["serving_outcome"] == {"answered": 2, "model_abstained": 2, "unparseable_output": 1, "no_valid_citation": 1}
    assert scored["invalid_citation_ids"] == 1 and scored["mean_completion_tokens"] == 5.0


def test_number_guard_withholds_numbers_absent_from_cited_passages(scored):
    g = scored["subsets"]["all"]["serving_guarded"]
    assert g["withheld_by_guard"] == 1                      # u1: "3" is not in d.pdf::0002
    assert g["rates"]["hallucinated_answer"]["k"] == 0
    assert g["rates"]["answer_correct"]["k"] == 1           # a1's 42 is in the cited passage, so it is kept
    assert (g["rates"]["abstention_precision"]["k"], g["rates"]["abstention_precision"]["n"]) == (3, 5)


def test_intent_reading_ignores_citation_policy(scored):
    r = scored["subsets"]["all"]["intent"]["rates"]
    assert r["answer_correct"]["k"] == 2                    # a3 "Adam." counts as an answer under intent


def test_wilson_interval():
    assert rc.wilson(5, 10) == [0.2366, 0.7634]
    assert rc.wilson(0, 10)[0] == 0.0 and rc.wilson(3, 0) is None


def test_breakdowns_and_provisional_flag(scored):
    assert scored["labels_provisional"] and scored["subsets"]["human_verified"] == {"n": 0}
    assert scored["by_document"]["d.pdf"]["n"] == 6
    assert scored["by_category"]["unanswerable"]["serving"]["answered"] == 1
