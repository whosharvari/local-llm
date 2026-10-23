"""v2 metrics: adjudicated correctness, refusal handling, ambiguous items, denominators."""

import json
from pathlib import Path

from ps3_llm.metrics import aggregate, claim_supported, facts_match, load_adjudication, score_example
from ps3_llm.schemas import EvalExample, EvidenceChunk, GroundedAnswer, GroundedQuery

ROOT = Path(__file__).resolve().parent.parent
ADJ = load_adjudication(ROOT / "data/mock/heldout_adjudication.json")
EV = "The tender must remain open for a minimum of 15 business days."


def _ex(qid, answerable, gold="", cits=(), category="answerable"):
    q = GroundedQuery("How long must the tender stay open?", [EvidenceChunk("a#1", EV)], query_id=qid)
    return EvalExample(q, answerable, gold, list(cits), category)


def _pred(raw):
    return GroundedAnswer(answer="", citations=[], abstained=False, raw_output=raw)


def test_claim_support():
    assert claim_supported("The tender must remain open 15 business days.", EV)
    assert not claim_supported("The tender must remain open 20 business days.", EV)


def test_facts_match_word_boundaries_and_reject():
    assert facts_match("Economy class must be booked.", [["economy"]], ["premium economy"])
    assert not facts_match("Premium economy", [["economy"]], ["premium economy"])  # audit h14
    assert facts_match("30 days", [["30 days"]])                                  # audit h17
    assert not facts_match("No", [["no"], ["client entertainment", "unless"]])    # audit h20
    assert not facts_match("Nothing is reimbursed", [["no"]])                     # 'no' is a whole word
    assert facts_match("EUR 1,800", [["1800"]])


def test_adjudication_covers_every_answerable_item_and_keeps_labels():
    rows = [json.loads(line) for line in (ROOT / "data/mock/heldout_eval.jsonl").read_text().splitlines()]
    answerable = {r["query_id"] for r in rows if r["answerable"]}
    assert answerable <= set(ADJ) and all(ADJ[q].get("facts") for q in answerable)
    assert ADJ["h30"]["category_override"] == "ambiguous"
    assert next(r for r in rows if r["query_id"] == "h30")["answerable"] is False  # original label preserved


def test_refusal_text_is_not_a_claim_or_hallucination():
    r = score_example(_ex("u1", False, category="distractor"),
                      _pred('{"abstain": false, "answer": "abstain", "citations": ["abstain"]}'), ADJ)
    for m in ("serving", "intent"):
        assert not r[m]["answered"] and r[m]["grounded"] and not r[m]["hallucinated"] and "n_claims" not in r[m]


def test_contradictory_flag_scored_by_intent_but_withheld_in_serving():
    adj = {**ADJ, "t1": {"facts": [["15"]]}}
    r = score_example(_ex("t1", True, "15 business days.", ["a#1"]),
                      _pred('{"abstain": "true", "answer": "15 business days.", "citations": ["a#1"]}'), adj)
    assert r["serving"]["inappropriate_abstention"] and not r["serving"]["correct"]
    assert r["intent"]["correct"] and r["intent"]["grounded"]


def test_hallucinated_answer_counts_unsupported_claim():
    r = score_example(_ex("u2", False, category="distractor"),
                      _pred('{"abstain": false, "answer": "It must stay open 20 days.", "citations": ["a#1"]}'), ADJ)
    assert r["serving"]["hallucinated"] and r["serving"]["n_unsupported"] == 1


def test_inference_error_is_counted_not_scored_as_answer():
    p = GroundedAnswer(answer="", citations=[], abstained=True, abstain_reason="inference_error: URLError: x",
                       parse_ok=False)
    r = score_example(_ex("u3", False, category="distractor"), p, ADJ)
    assert r["inference_error"] and not r["intent"]["answered"]
    assert aggregate([r])["inference_errors"] == 1


def test_ambiguous_item_accepts_abstention_or_supported_answer():
    q = GroundedQuery("Can I book premium economy for a 3-hour flight?",
                      [EvidenceChunk("travel#c1", "Economy class must be booked for all flights under 6 hours.")],
                      query_id="h30")
    ex = EvalExample(q, False, "", [], "distractor")
    ok = score_example(ex, _pred('{"abstain": false, "answer": "No, economy class is required for flights under 6 hours.", "citations": ["travel#c1"]}'), ADJ)
    ab = score_example(ex, _pred('{"abstain": true, "answer": "", "citations": []}'), ADJ)
    bad = score_example(ex, _pred('{"abstain": false, "answer": "Yes, if you pay the difference.", "citations": ["travel#c1"]}'), ADJ)
    assert ok["serving"]["grounded"] and ab["serving"]["grounded"] and not bad["serving"]["grounded"]
    s = aggregate([ok, bad])
    assert s["n_ambiguous"] == 2 and s["serving"]["hallucinated_answer_rate"] is None  # ambiguous excluded from U


def test_aggregate_denominators():
    adj = {**ADJ, "t1": {"facts": [["15"]]}}
    rows = [
        score_example(_ex("t1", True, "15 business days.", ["a#1"]),
                      _pred('{"abstain": false, "answer": "15 business days.", "citations": ["a#1"]}'), adj),
        score_example(_ex("u1", False, category="distractor"), _pred('{"abstain": true, "answer": "", "citations": []}'), adj),
    ]
    s = aggregate(rows)
    assert (s["n_answerable"], s["n_unanswerable"]) == (1, 1)
    assert s["serving"]["grounded_accuracy"] == 1.0 and s["serving"]["hallucinated_answer_rate"] == 0.0
    assert s["serving"]["unsupported_claim_rate"] == 0.0 and s["json_valid_rate"] == 1.0
