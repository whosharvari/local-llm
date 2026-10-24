"""Serving policy and intent reading of ps3_llm.prompting.parse_model_output (regression tests from the audit)."""

import pytest

from ps3_llm.prompting import is_refusal_text, parse_flag, parse_model_output

IDS = {"a#1", "a#2"}


def P(text):
    return parse_model_output(text, IDS)


def test_clean_answer():
    out = P('{"abstain": false, "answer": "15 days.", "citations": ["a#1"]}')
    assert (out["abstained"], out["answer"], out["citations"], out["abstain_reason"]) == (False, "15 days.", ["a#1"], "")
    assert out["parse_ok"] and out["schema_ok"] and out["violations"] == []
    assert out["intent_answered"] and out["intent_answer"] == "15 days."


def test_clean_abstention():
    out = P('{"abstain": true, "answer": "", "citations": []}')
    assert out["abstained"] and out["abstain_reason"] == "model_abstained" and out["violations"] == []
    assert not out["intent_answered"]


@pytest.mark.parametrize("raw,expected", [
    (True, True), (False, False), ("true", True), ("True ", True), ("false", False), ("FALSE", False),
    ("yes", True), ("no", False), (1, True), (0, False), ("abstain", True), (None, None), ("maybe", None), (2, None),
])
def test_parse_flag(raw, expected):
    assert parse_flag(raw) is expected


def test_string_false_flag_answers():
    # audit: a string "false" must not be truthy
    out = P('{"abstain": "false", "answer": "15 days.", "citations": ["a#1"]}')
    assert not out["abstained"] and "non_boolean_flag:'false'" in out["violations"] and not out["schema_ok"]


# --- bare refusals / refusal phrases (audit h33, h34, h37, h26) ---
@pytest.mark.parametrize("text", [
    "abstain\ncitations: []",                       # non-JSON bare refusal (h33/h34)
    "Abstain.",
    '{"abstain": "abstain", "answer": "abstain", "citations": []}',          # h37
    '{"abstain": false, "answer": "abstain", "citations": ["abstain"]}',     # h27
    '{"abstain": "abstain", "answer": "Current CISO not specified in the given evidence.", "citations": ["[a#1]"]}',  # h26
    '{"abstain": false, "answer": "The documents do not contain this information.", "citations": []}',
])
def test_refusals_never_count_as_answers(text):
    out = P(text)
    assert out["abstained"] and out["answer"] == "" and out["citations"] == []
    assert not out["intent_answered"] and out["intent_answer"] == ""


def test_refusal_text_detection_does_not_flag_real_negative_answers():
    assert not is_refusal_text("No, unless it is part of an approved client entertainment event.")
    assert not is_refusal_text("Alcohol is not reimbursable.")
    assert not is_refusal_text("No")
    assert is_refusal_text("abstain") and is_refusal_text("This is not specified in the passages.")


# --- contradictory flags (audit h10/h12/h18 and h21) ---
def test_flag_true_with_substantive_answer_is_withheld_but_intent_answers():
    out = P('{"abstain": "true", "answer": "4400", "citations": ["a#1"]}')
    assert out["abstained"] and out["abstain_reason"] == "contradictory_flag"
    assert "flag_true_but_substantive_answer" in out["violations"]
    assert out["intent_answered"] and out["intent_answer"] == "4400" and out["intent_citations"] == ["a#1"]


def test_flag_false_with_refusal_text_is_violation():
    out = P('{"abstain": false, "answer": "abstain", "citations": []}')
    assert out["abstained"] and "flag_false_but_refusal_or_empty" in out["violations"]


# --- malformed / missing ---
def test_malformed_json_is_unparseable_abstention():
    out = P('{"abstain": false, "answer": "15 days", "citations": ["a#1"]')   # truncated
    assert out["abstained"] and out["abstain_reason"] == "unparseable_output" and not out["parse_ok"]
    assert "not_json" in out["violations"]


def test_non_json_prose_answer_is_withheld_but_readable_as_intent():
    out = P("It is 15 days [a#2].")
    assert out["abstained"] and out["abstain_reason"] == "unparseable_output"
    assert out["intent_answered"] and out["intent_citations"] == ["a#2"]


def test_missing_flag_decided_by_content():
    out = P('{"answer": "15 days.", "citations": ["a#1"]}')
    assert not out["abstained"] and "flag_missing" in out["violations"] and "missing_keys:abstain" in out["violations"]
    out = P('{"answer": "", "citations": []}')
    assert out["abstained"]


def test_empty_output():
    out = P("")
    assert out["abstained"] and not out["intent_answered"]


# --- citations ---
def test_invented_ids_never_returned():
    out = P('{"abstain": false, "answer": "x", "citations": ["a#1", "z#9", "[a#2]", " a#2 "]}')
    assert out["citations"] == ["a#1", "a#2"] and out["invalid_citations"] == ["z#9"]
    assert "invalid_citation_ids" in out["violations"]


def test_answer_with_only_invalid_citations_abstains():
    out = P('{"abstain": false, "answer": "It is 20 days.", "citations": ["z#9"]}')
    assert out["abstained"] and out["abstain_reason"] == "no_valid_citation" and out["citations"] == []


def test_answer_without_citations_abstains():
    out = P('{"abstain": false, "answer": "It is 20 days.", "citations": []}')
    assert out["abstained"] and out["abstain_reason"] == "no_valid_citation"
    assert out["intent_answered"]  # still visible to scoring as an (uncited) answer


def test_citation_string_instead_of_list():
    out = P('{"abstain": false, "answer": "15 days.", "citations": "a#1"}')
    assert out["citations"] == ["a#1"] and not out["abstained"]


def test_think_block_and_fences_are_ignored():
    out = P('<think>hmm</think>```json\n{"abstain": false, "answer": "15 days.", "citations": ["a#1"]}\n```')
    assert not out["abstained"] and out["parse_ok"]


# --- JSON extraction (replaces the greedy {.*} match) ---
OK = '{"abstain": false, "answer": "15 days.", "citations": ["a#1"]}'


@pytest.mark.parametrize("text", [
    OK,
    "  \n\t" + OK + "\n\n  ",
    '{\n  "abstain": false,\n  "answer": "15 days.",\n  "citations": [\n    "a#1"\n  ]\n}',
])
def test_valid_json_with_whitespace_is_clean(text):
    out = P(text)
    assert (out["abstained"], out["answer"], out["citations"]) == (False, "15 days.", ["a#1"])
    assert out["parse_ok"] and out["schema_ok"] and out["violations"] == []


def test_braces_inside_answer_string():
    out = P('{"abstain": false, "answer": "Use the {project} code, then }.", "citations": ["a#1"]}')
    assert not out["abstained"] and out["answer"] == "Use the {project} code, then }." and out["violations"] == []


@pytest.mark.parametrize("text", [
    OK + "\nNote: ids look like {doc#chunk}.",                   # greedy match failed on this
    OK + " }",
    "Here is the answer:\n" + OK,
    "```json\n" + OK + "\n```\nHope this helps {:}",
])
def test_prose_around_valid_object_is_used_but_reported(text):
    out = P(text)
    assert (out["abstained"], out["answer"], out["citations"]) == (False, "15 days.", ["a#1"])
    assert out["parse_ok"] and out["schema_ok"] and out["violations"] == ["text_outside_json"]


@pytest.mark.parametrize("text", [
    OK + "\n" + '{"abstain": true, "answer": "", "citations": []}',
    OK + OK,
    OK + ' Also: {"note": 1}',
    '{"draft": 1} ' + OK,
])
def test_multiple_json_objects_are_ambiguous_and_rejected(text):
    out = P(text)
    assert out["abstained"] and out["abstain_reason"] == "unparseable_output"
    assert not out["parse_ok"] and not out["schema_ok"] and out["answer"] == "" and out["citations"] == []
    assert out["violations"][:2] == ["not_json", "multiple_json_objects"]


@pytest.mark.parametrize("text", [
    '{"abstain": false, "answer": "15 days" "citations": ["a#1"]}',          # missing comma
    '{"abstain": false, "answer": {"x": 1}',                                  # truncated; inner object valid
    "{'abstain': False, 'answer': '15 days.', 'citations': ['a#1']}",         # Python repr, not JSON
    "Answer {draft}: " + OK,                                                  # first brace is not JSON
    "   ",
])
def test_malformed_output_is_never_an_answer(text):
    out = P(text)
    assert out["abstained"] and out["abstain_reason"] == "unparseable_output" and not out["parse_ok"]
    assert out["answer"] == "" and out["citations"] == [] and "not_json" in out["violations"]


def test_missing_required_fields_are_not_invented():
    out = P('{"abstain": false, "answer": "15 days."}')                     # no citations key
    assert out["abstained"] and out["abstain_reason"] == "no_valid_citation" and out["citations"] == []
    assert "missing_keys:citations" in out["violations"] and not out["schema_ok"]
    out = P('{"abstain": false, "citations": ["a#1"]}')                      # no answer key
    assert out["abstained"] and out["abstain_reason"] == "model_abstained" and out["answer"] == ""
    assert "missing_keys:answer" in out["violations"] and not out["schema_ok"]
    out = P("{}")
    assert out["abstained"] and out["violations"][0] == "missing_keys:abstain,answer,citations"
