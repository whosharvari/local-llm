"""Prompt construction and output parsing shared by inference, training data, and eval.

The same SYSTEM_PROMPT / render_user_prompt are used to build SFT examples so that the
fine-tuned model sees exactly the format it is served with.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .schemas import GroundedQuery

ABSTAIN_ANSWER = "The provided documents do not contain enough information to answer this question."

SYSTEM_PROMPT = (
    "You are a document-grounded assistant. Answer ONLY using the evidence passages provided. "
    "Do not use outside knowledge.\n"
    "Rules:\n"
    "1. If the evidence fully supports an answer, answer concisely and cite the ids of the passages you used.\n"
    "2. If the evidence does not contain the answer, abstain: set \"abstain\" to true, leave \"citations\" empty, "
    "and do not guess.\n"
    "3. Every factual statement in the answer must be supported by a cited passage.\n"
    "Respond with a single JSON object and nothing else:\n"
    '{"abstain": <true|false>, "answer": "<answer text or empty>", "citations": ["<passage id>", ...]}'
)


def render_evidence(query: GroundedQuery) -> str:
    if not query.evidence:
        return "(no evidence passages were retrieved)"
    blocks = []
    for ch in query.evidence:
        loc = ch.source + (f", p.{ch.page}" if ch.page is not None else "")
        header = f"[{ch.chunk_id}]" + (f" ({loc})" if loc else "")
        blocks.append(f"{header}\n{ch.text.strip()}")
    return "\n\n".join(blocks)


def render_user_prompt(query: GroundedQuery) -> str:
    return f"Evidence passages:\n\n{render_evidence(query)}\n\nQuestion: {query.question.strip()}"


def build_messages(query: GroundedQuery) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(query)},
    ]


def format_target(answer: str, citations: list[str], abstain: bool) -> str:
    """Canonical assistant output, used as the SFT target."""
    if abstain:
        return json.dumps({"abstain": True, "answer": "", "citations": []}, ensure_ascii=False)
    return json.dumps({"abstain": False, "answer": answer, "citations": citations}, ensure_ascii=False)


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

# A refusal written as answer text. Bare words must be the whole answer; phrases may appear anywhere.
_REFUSAL_WORD = re.compile(r"^\W*(?:abstain(?:ed|ing)?|refuse[ds]?|no answer)\W*$", re.IGNORECASE)
_REFUSAL_PHRASES = (
    "do not contain", "does not contain", "doesn't contain", "don't contain", "not enough information",
    "insufficient information", "insufficient evidence", "cannot be answered", "can't be answered",
    "cannot be determined", "can't be determined", "no information", "no relevant information",
    "not mentioned", "not explicitly mentioned", "not provided in", "not specified", "not stated",
    "unable to answer", "cannot answer", "can't answer", "not found in the",
)
_TRUE_STR = {"true", "yes", "1", "abstain", "abstained"}
_FALSE_STR = {"false", "no", "0"}

# Serving decision reasons (GroundedAnswer.abstain_reason)
REASON_MODEL = "model_abstained"
REASON_UNPARSEABLE = "unparseable_output"
REASON_CONTRADICTION = "contradictory_flag"
REASON_NO_CITATION = "no_valid_citation"


def is_refusal_text(answer: str) -> bool:
    """True when the answer *text* is itself a refusal ("abstain", "not specified in the evidence")."""
    low = answer.strip().lower()
    return bool(_REFUSAL_WORD.match(low)) or any(p in low for p in _REFUSAL_PHRASES)


def parse_flag(value: Any) -> bool | None:
    """Interpret the "abstain" field. bool as-is; "true"/"yes"/"1"/"abstain" -> True;
    "false"/"no"/"0" -> False; 1/0 -> bool; anything else (missing, null, other strings) -> None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE_STR:
            return True
        if v in _FALSE_STR:
            return False
    return None


def _clean_id(c: Any) -> str:
    return re.sub(r"^\[(.*)\]$", r"\1", str(c).strip()).strip()


def parse_model_output(text: str, valid_ids: set[str]) -> dict[str, Any]:
    """Parse raw model text deterministically into a serving decision and an intent reading.

    Two decisions with different purposes:

    * Serving (`abstained`, `answer`, `citations`, `abstain_reason`) — what the API returns.
      Conservative: answer only if the output is valid JSON, the answer text is substantive,
      the flag does not say abstain, and at least one cited id exists in the request evidence.
        1. not JSON                                   -> abstain (unparseable_output)
        2. empty answer or refusal text               -> abstain (model_abstained)
        3. flag True + substantive answer             -> abstain (contradictory_flag)
        4. substantive, flag False/missing, no valid id -> abstain (no_valid_citation)
        5. otherwise                                  -> answer with the valid ids only
    * Intent (`intent_answered`, `intent_answer`) — retrospective scoring of what the model
      *meant*, independent of the flag: answered iff the answer text (or, for non-JSON output, the
      whole text minus a trailing "citations:" line) is substantive and not a refusal. Used to compare
      models whose flag usage differs; never used for serving.

    `violations` lists every contract breach (non-JSON, missing/non-boolean flag, flag/answer
    contradictions, invented citation ids, missing keys) so they are visible, not silently resolved.
    Citation ids are never returned unless they are in `valid_ids`.
    """
    cleaned = _THINK.sub("", text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    obj: dict[str, Any] | None = None
    m = _JSON_OBJ.search(cleaned)
    if m:
        try:
            cand = json.loads(m.group(0))
            obj = cand if isinstance(cand, dict) else None
        except json.JSONDecodeError:
            obj = None

    violations: list[str] = []
    if obj is None:
        violations.append("not_json")
        answer = re.sub(r"\n?\s*citations\s*:.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        raw_cits = re.findall(r"\[([^\[\]]+)\]", cleaned)
        flag = None
    else:
        missing = {"abstain", "answer", "citations"} - set(obj)
        if missing:
            violations.append("missing_keys:" + ",".join(sorted(missing)))
        raw_flag = obj.get("abstain")
        flag = parse_flag(raw_flag)
        if raw_flag is not None and not isinstance(raw_flag, bool):
            violations.append(f"non_boolean_flag:{raw_flag!r}")
        if flag is None:
            violations.append("flag_unrecognised" if "abstain" in obj else "flag_missing")
        answer = str(obj.get("answer") or "").strip()
        raw_cits = obj.get("citations") or []
        raw_cits = [raw_cits] if isinstance(raw_cits, str) else raw_cits
    cits = [_clean_id(c) for c in raw_cits]
    valid = list(dict.fromkeys(c for c in cits if c in valid_ids))
    invalid = [c for c in cits if c not in valid_ids]
    if invalid:
        violations.append("invalid_citation_ids")

    refusal = is_refusal_text(answer) if answer else False
    substantive = bool(answer) and not refusal
    if flag is True and substantive:
        violations.append("flag_true_but_substantive_answer")
    if flag is False and not substantive:
        violations.append("flag_false_but_refusal_or_empty")

    if obj is None:
        abstained, reason = True, REASON_UNPARSEABLE
    elif not substantive:
        abstained, reason = True, REASON_MODEL
    elif flag is True:
        abstained, reason = True, REASON_CONTRADICTION
    elif not valid:
        abstained, reason = True, REASON_NO_CITATION
    else:
        abstained, reason = False, ""

    return {
        "answer": "" if abstained else answer,
        "citations": [] if abstained else valid,
        "abstained": abstained,
        "abstain_reason": reason,
        "parse_ok": obj is not None,
        "schema_ok": obj is not None and not ({"abstain", "answer", "citations"} - set(obj)) and isinstance(obj.get("abstain"), bool),
        "invalid_citations": invalid,
        "violations": violations,
        "flag": flag,
        "intent_answered": substantive,
        "intent_answer": answer if substantive else "",
        "intent_citations": valid if substantive else [],
    }
