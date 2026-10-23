"""Integration contract between retrieval (Member 1) and the model module (Member 2).

Status: PROPOSED — pending sign-off from Member 1. Field names are kept flat and
JSON-serialisable so either side can produce/consume them without importing this module.

Retrieval -> model:   GroundedQuery  (question + ranked EvidenceChunk list)
Model -> caller:      GroundedAnswer (answer text, citations, abstention decision)
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
