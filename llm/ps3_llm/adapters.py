"""Model adapters: one interface, a deterministic mock, and the SGLang-served model.

    adapter = get_adapter("mock")                                    # no server needed
    adapter = get_adapter("sglang", base_url="http://127.0.0.1:30000/v1")
    answer  = adapter.answer(GroundedQuery(...))                     # -> GroundedAnswer
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from .prompting import build_messages, parse_model_output
from .schemas import INFERENCE_ERROR_PREFIX, GroundedAnswer, GroundedQuery

ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "abstain": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["abstain", "answer", "citations"],
    "additionalProperties": False,
}


class ModelAdapter(Protocol):
    name: str

    def answer(self, query: GroundedQuery) -> GroundedAnswer: ...

    def answer_batch(self, queries: list[GroundedQuery]) -> list[GroundedAnswer]: ...


_WORD = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the of to in on for and or is are was were be been by with what which who whom when where why how "
    "do does did can could should would will may might must this that these those it its as at from our your "
    "their there than then into about over under per any all each".split()
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if t not in _STOP and len(t) > 1}


class MockAdapter:
    """Deterministic lexical-overlap 'model' for wiring and tests; no LLM involved.

    Picks the evidence sentence with the highest overlap with the question's content words.
    Abstains when overlap is below `min_overlap`. Good enough to exercise the pipeline and the
    metrics; not a quality baseline.
    """

    def __init__(self, min_overlap: float = 0.5, name: str = "mock-lexical"):
        self.min_overlap = min_overlap
        self.name = name

    def answer(self, query: GroundedQuery) -> GroundedAnswer:
        t0 = time.perf_counter()
        q = _content_tokens(query.question)
        best = (0.0, "", "")
        for ch in query.evidence:
            for sent in re.split(r"(?<=[.!?])\s+", ch.text):
                s = _content_tokens(sent)
                score = len(q & s) / max(len(q), 1)
                if score > best[0]:
                    best = (score, sent.strip(), ch.chunk_id)
        if best[0] < self.min_overlap:
            out = GroundedAnswer(answer="", citations=[], abstained=True,
                                 abstain_reason="model_abstained")
        else:
            out = GroundedAnswer(answer=best[1], citations=[best[2]], abstained=False)
        out.query_id, out.model = query.query_id, self.name
        out.latency_s = time.perf_counter() - t0
        out.raw_output = json.dumps({"abstain": out.abstained, "answer": out.answer, "citations": out.citations})
        return out

    def answer_batch(self, queries: list[GroundedQuery]) -> list[GroundedAnswer]:
        return [self.answer(q) for q in queries]


class SGLangAdapter:
    """Calls an SGLang server through its OpenAI-compatible /v1/chat/completions endpoint.

    `model` must match the served model id (SGLang reports it at /v1/models). With
    `constrained=True` the request carries a JSON schema via `response_format`, which SGLang
    enforces with grammar-constrained decoding (xgrammar) where the backend supports it.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000/v1",
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 384,
        constrained: bool = False,
        timeout: float = 180.0,
        concurrency: int = 4,
        api_key: str = "EMPTY",
        extra_body: dict | None = None,
        name: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self._model = model          # resolved lazily from /v1/models; never raises in __init__
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.constrained = constrained
        self.concurrency = concurrency
        self.extra_body = extra_body or {}
        self._name = name

    @property
    def model(self) -> str | None:
        return self._model

    @property
    def name(self) -> str:
        return self._name or f"sglang:{self._model or 'unresolved'}"

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _served_model(self) -> str:
        return self._request("/models")["data"][0]["id"]

    def _error_answer(self, query: GroundedQuery, err: BaseException, t0: float) -> GroundedAnswer:
        return GroundedAnswer(answer="", citations=[], abstained=True,
                              abstain_reason=f"{INFERENCE_ERROR_PREFIX}{type(err).__name__}: {err}",
                              query_id=query.query_id, model=self.name, parse_ok=False,
                              latency_s=time.perf_counter() - t0, warnings=["inference_error"])

    def answer(self, query: GroundedQuery) -> GroundedAnswer:
        t0 = time.perf_counter()
        if self._model is None:
            try:
                self._model = self._served_model()
            except Exception as e:  # server down, bad URL, malformed /v1/models response
                return self._error_answer(query, e, t0)
        payload = {
            "model": self._model,
            "messages": build_messages(query),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        if self.constrained:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "grounded_answer", "schema": ANSWER_JSON_SCHEMA},
            }
        try:
            resp = self._request("/chat/completions", payload)
            text = resp["choices"][0]["message"]["content"] or ""
            usage = resp.get("usage") or {}
        except Exception as e:  # HTTP/URL errors, timeouts, connection resets, malformed responses
            return self._error_answer(query, e, t0)
        latency = time.perf_counter() - t0
        return answer_from_text(text, query, self.name, latency, usage)

    def answer_batch(self, queries: list[GroundedQuery]) -> list[GroundedAnswer]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self.answer, queries))


def answer_from_text(text: str, query: GroundedQuery, model: str, latency_s: float | None,
                     usage: dict | None = None) -> GroundedAnswer:
    """Apply the serving policy (prompting.parse_model_output) to raw model text."""
    parsed = parse_model_output(text, {c.chunk_id for c in query.evidence})
    return GroundedAnswer(
        answer=parsed["answer"], citations=parsed["citations"], abstained=parsed["abstained"],
        abstain_reason=parsed["abstain_reason"], query_id=query.query_id, model=model, latency_s=latency_s,
        raw_output=text, parse_ok=parsed["parse_ok"], warnings=parsed["violations"],
        usage={**(usage or {}), "invalid_citations": parsed["invalid_citations"]},
    )


def get_adapter(kind: str, **kwargs) -> ModelAdapter:
    if kind == "mock":
        return MockAdapter(**kwargs)
    if kind == "sglang":
        return SGLangAdapter(**kwargs)
    raise ValueError(f"unknown adapter kind: {kind!r} (expected 'mock' or 'sglang')")
