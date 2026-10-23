"""Integration contract between retrieval (Member 1, Papermind) and the model module (Member 2).

Contract version: 1.1 (see llm/INTEGRATION_GUIDE.md for examples and invocation). Field names are flat
and JSON-serialisable so either side can produce/consume them without importing this module.

Retrieval -> model:   GroundedQuery  (question + evidence chunks)
Model -> caller:      GroundedAnswer (answer text, citations, abstention decision, diagnostics)

Request:
    required  question            non-empty string
    required  evidence            list (may be empty) of chunks:
                  required  chunk_id   non-empty string, unique within the request, no leading/trailing
                                       whitespace, not wrapped in [ ]; stable across re-indexing
                  required  text       string
                  optional  source, page (int), score (float) — display/diagnostics only
    optional  query_id            string echoed back (recommended)

Response guarantees (serving policy, ps3_llm.prompting.parse_model_output):
    - citations is always a subset of the request's evidence chunk_ids; invented ids are never returned
      (they are listed in usage["invalid_citations"] and warnings);
    - abstained == True  <=>  answer == "" and citations == [];
    - abstained == False  =>  answer is non-empty and citations has >= 1 valid id
      (an answer without any valid citation is converted to an abstention, reason "no_valid_citation");
    - abstain_reason is one of:
        "model_abstained"      the model declined or produced refusal/empty text
        "contradictory_flag"   model set abstain=true but wrote an answer; withheld for safety
        "no_valid_citation"    answer cited no id from the evidence
        "unparseable_output"   model output was not JSON
        "inference_error: ..." server unavailable / HTTP error / timeout (parse_ok == False)
        "invalid_request: ..." request failed validation (no model call made)
    - empty evidence is valid and normally yields abstained=True;
    - warnings lists contract violations observed in the model output (diagnostic, never fatal).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CONTRACT_VERSION = "1.1"
INFERENCE_ERROR_PREFIX = "inference_error: "
INVALID_REQUEST_PREFIX = "invalid_request: "


@dataclass
class EvidenceChunk:
    chunk_id: str                 # stable id from retrieval, e.g. "handbook.pdf#p12-c3"
    text: str
    source: str = ""              # document name / path
    page: int | None = None
    score: float | None = None    # retrieval score, informational only

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvidenceChunk":
        return cls(
            chunk_id=str(d["chunk_id"]),
            text=d["text"],
            source=d.get("source", ""),
            page=d.get("page"),
            score=d.get("score"),
        )


def validate_request(d: Any) -> list[str]:
    """Return a list of contract violations in a request dict (empty list = valid)."""
    if not isinstance(d, dict):
        return ["request must be a JSON object"]
    errs = []
    q = d.get("question")
    if not isinstance(q, str) or not q.strip():
        errs.append("question must be a non-empty string")
    ev = d.get("evidence", None)
    if not isinstance(ev, list):
        errs.append("evidence must be a list (use [] when nothing was retrieved)")
        return errs
    seen = set()
    for i, c in enumerate(ev):
        if not isinstance(c, dict):
            errs.append(f"evidence[{i}] must be an object")
            continue
        cid = c.get("chunk_id")
        if not isinstance(cid, str) or not cid.strip():
            errs.append(f"evidence[{i}].chunk_id must be a non-empty string")
        else:
            if cid != cid.strip() or (cid.startswith("[") and cid.endswith("]")):
                errs.append(f"evidence[{i}].chunk_id {cid!r} has surrounding whitespace or brackets")
            if cid in seen:
                errs.append(f"evidence[{i}].chunk_id {cid!r} is duplicated")
            seen.add(cid)
        if not isinstance(c.get("text"), str):
            errs.append(f"evidence[{i}].text must be a string")
        if c.get("page") is not None and not isinstance(c.get("page"), int):
            errs.append(f"evidence[{i}].page must be an integer or omitted")
    if "query_id" in d and not isinstance(d["query_id"], (str, int)):
        errs.append("query_id must be a string")
    return errs


@dataclass
class GroundedQuery:
    question: str
    evidence: list[EvidenceChunk]
    query_id: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GroundedQuery":
        return cls(
            question=d["question"],
            evidence=[EvidenceChunk.from_dict(e) for e in d.get("evidence", [])],
            query_id=str(d.get("query_id", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GroundedAnswer:
    answer: str                   # empty string when abstaining
    citations: list[str]          # chunk_ids that support the answer; subset of query evidence ids
    abstained: bool
    abstain_reason: str = ""
    query_id: str = ""
    model: str = ""
    latency_s: float | None = None
    raw_output: str = ""          # unparsed model text, kept for error analysis
    parse_ok: bool = True         # False when the model output was not JSON or inference failed
    usage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)   # output-contract violations (diagnostic)
    contract_version: str = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalExample:
    """Held-out evaluation item: a query plus gold labels."""

    query: GroundedQuery
    answerable: bool
    gold_answer: str = ""
    gold_citations: list[str] = field(default_factory=list)
    category: str = ""            # e.g. "answerable", "missing_evidence", "distractor", "partial"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalExample":
        return cls(
            query=GroundedQuery.from_dict(d),
            answerable=bool(d["answerable"]),
            gold_answer=d.get("gold_answer", ""),
            gold_citations=list(d.get("gold_citations", [])),
            category=d.get("category", ""),
        )
