"""Integration contract between retrieval (Member 1) and the model module (Member 2).

Status: PROPOSED — pending sign-off from Member 1. Field names are kept flat and
JSON-serialisable so either side can produce/consume them without importing this module.

Retrieval -> model:   GroundedQuery  (question + ranked EvidenceChunk list)
Model -> caller:      GroundedAnswer (answer text, citations, abstention decision)

Request (what Member 1's retriever hands to the model adapter), as JSON:
    {"query_id": "q-17",
     "question": "How long must a tender stay open?",
     "evidence": [{"chunk_id": "procurement.pdf#p3-c2", "text": "...", "source": "procurement.pdf",
                   "page": 3, "score": 0.82}, ...]}

Response (what ps3_llm.adapters.*.answer() returns, GroundedAnswer.to_dict()):
    {"query_id": "q-17", "answer": "At least 15 business days.", "citations": ["procurement.pdf#p3-c2"],
     "abstained": false, "abstain_reason": "", "model": "sglang:<served id>", "latency_s": 2.1,
     "raw_output": "<model text>", "parse_ok": true, "usage": {...}}

Guarantees from the model side:
    - citations is always a subset of the request's evidence chunk_ids (invented ids are dropped and
      counted in usage["invalid_citations"]);
    - abstained=True implies answer == "" and citations == [];
    - an empty evidence list is valid and should produce an abstention;
    - inference errors return abstained=True, parse_ok=False, abstain_reason="inference_error: ...".

Open points needing Member 1's approval before integration:
    1. chunk_id format and stability: ids are shown to the model as "[<chunk_id>]" and returned as
       citations, so they must be unique per request, stable across re-indexing, and contain no "]".
    2. Evidence budget: the model was trained with 1-4 passages per question (~450 tokens max) and
       is served with a 4096-token context. Proposed: top-k <= 5, chunk text <= ~200 words.
    3. Ordering: evidence order is retrieval-rank order; the model does not rely on it, but `score`
       is informational only and is not used for abstention.
    4. `page` is optional and only used for display ("source, p.N") in the prompt.
    5. Abstention is decided by the model from the evidence; retrieval should NOT pre-filter low-score
       questions silently — pass an empty evidence list instead so abstention is logged consistently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
    parse_ok: bool = True         # False when the model output was not valid JSON
    usage: dict[str, Any] = field(default_factory=dict)

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
