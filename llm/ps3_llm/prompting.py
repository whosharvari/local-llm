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
_ABSTAIN_PHRASES = (
    "do not contain", "does not contain", "not enough information", "cannot be answered",
    "can't be answered", "no information", "not mentioned", "not provided", "insufficient",
    "unable to answer", "cannot answer", "not stated", "not specified",
)
_ABSTAIN_WORD = re.compile(r"^\W*abstain\b", re.IGNORECASE)


def is_abstention_text(answer: str) -> bool:
    """True when the answer *text* is itself a refusal ("abstain", "not specified in the evidence")."""
    low = answer.strip().lower()
    return bool(_ABSTAIN_WORD.match(low)) or any(p in low for p in _ABSTAIN_PHRASES)


def parse_model_output(text: str, valid_ids: set[str], lenient_abstain: bool = False) -> dict[str, Any]:
    """Parse model text into {answer, citations, abstained, parse_ok}.

    Tolerates code fences, <think> blocks and trailing prose. When JSON parsing fails we fall
    back to treating the whole text as the answer and detect abstention heuristically; parse_ok
    is then False so the failure shows up in evaluation instead of being hidden.
    lenient_abstain=True also treats a JSON answer whose text is a refusal (e.g. "abstain",
    "not specified in the evidence") as an abstention even when the "abstain" flag is false/absent.
    Strict mode (default) is used for serving; lenient mode is an evaluation sensitivity check.
    Citations are filtered to ids present in the query evidence (hallucinated ids are dropped
    and counted).
    """
    cleaned = _THINK.sub("", text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    obj: dict[str, Any] | None = None
    m = _JSON_OBJ.search(cleaned)
    if m:
        try:
            cand = json.loads(m.group(0))
            if isinstance(cand, dict):
                obj = cand
        except json.JSONDecodeError:
            obj = None

    if obj is None:
        low = cleaned.lower()
        abstained = any(p in low for p in _ABSTAIN_PHRASES)
        cited = re.findall(r"\[([^\[\]]+)\]", cleaned)
        valid = [c for c in cited if c in valid_ids]
        return {
            "answer": "" if abstained else cleaned,
            "citations": list(dict.fromkeys(valid)),
            "abstained": abstained,
            "parse_ok": False,
            "invalid_citations": [c for c in cited if c not in valid_ids],
        }

    raw_cits = obj.get("citations") or []
    if isinstance(raw_cits, str):
        raw_cits = [raw_cits]
    raw_cits = [str(c).strip().strip("[]") for c in raw_cits]
    valid = [c for c in raw_cits if c in valid_ids]
    abstain_flag = obj.get("abstain")
    if isinstance(abstain_flag, str):
        abstain_flag = abstain_flag.strip().lower() == "true"
    answer = str(obj.get("answer") or "").strip()
    abstained = bool(abstain_flag) or (not answer) or (lenient_abstain and is_abstention_text(answer))
    return {
        "answer": "" if abstained else answer,
        "citations": [] if abstained else list(dict.fromkeys(valid)),
        "abstained": abstained,
        "parse_ok": True,
        "invalid_citations": [c for c in raw_cits if c not in valid_ids],
    }
