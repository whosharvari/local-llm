"""Retrieval over Member 1's vector store (../vector_store), and a retrieval-side abstention gate.

The index is a FAISS IndexFlatIP of L2-normalised all-MiniLM-L6-v2 embeddings, one row per chunk of
data/processed/chunks.json (vector_store/metadata.json is row-aligned with it). It is read with numpy
(the flat format is a fixed header followed by float32 vectors), so faiss is not a dependency; inner
product over normalised vectors is exactly FAISS's IndexFlatIP search.

Gate: logistic regression on two features of the retrieval result,
    top1_score     cosine similarity of the best passage
    best_coverage  max over the top-k passages of the share of question content words in the passage
Member 1's calibration (vector_store/grounding_calibration.json) uses the same feature names, but its
coverage definition is in Papermind code that is not in this repository, and it was fitted on grounded_qa
questions that include dev/test-document questions. It is therefore not reused: the gate is refitted with
`python -m ps3_llm.retrieval fit-gate` on the leakage-free train-split questions only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import content_tokens

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
INDEX = REPO / "vector_store/index.faiss"
METADATA = REPO / "vector_store/metadata.json"
GATE = ROOT / "data/real_train/v1/retrieval_gate.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def read_flat_ip_index(path: Path) -> np.ndarray:
    """Vectors of a FAISS IndexFlatIP file ('IxFI', d:int32, ntotal:int64, 2x int64, is_trained:u8,
    metric:int32, n_floats:uint64, floats)."""
    raw = path.read_bytes()
    if raw[:4] != b"IxFI":
        raise ValueError(f"{path}: not a FAISS IndexFlatIP file (header {raw[:4]!r})")
    d = int.from_bytes(raw[4:8], "little")
    n = int.from_bytes(raw[8:16], "little")
    size = int.from_bytes(raw[37:45], "little")
    if size != n * d or len(raw) != 45 + 4 * size:
        raise ValueError(f"{path}: inconsistent sizes (d={d}, n={n}, floats={size}, bytes={len(raw)})")
    return np.frombuffer(raw[45:], dtype="<f4").reshape(n, d)


def coverage(question: str, text: str) -> float:
    q = set(content_tokens(question))
    return len(q & set(content_tokens(text))) / len(q) if q else 0.0


@dataclass
class Hit:
    chunk_id: str
    document_id: str
    page_number: int
    text: str
    score: float


class Retriever:
    def __init__(self, index: Path = INDEX, metadata: Path = METADATA, encoder=None):
        self.vectors = read_flat_ip_index(index)
        self.meta = json.loads(metadata.read_text())
        if len(self.meta) != len(self.vectors):
            raise ValueError(f"index has {len(self.vectors)} vectors but metadata has {len(self.meta)} rows")
        self._encoder = encoder

    def encode(self, text: str) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        return np.asarray(self._encoder.encode([text], normalize_embeddings=True)[0], dtype="float32")

    def search(self, question: str, k: int = 4) -> list[Hit]:
        scores = self.vectors @ self.encode(question)
        top = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.meta[i]["id"], self.meta[i]["source_file"], self.meta[i]["page_number"],
                    self.meta[i]["text"], float(scores[i])) for i in top]


def features(question: str, hits: list[Hit]) -> list[float]:
    return [hits[0].score if hits else 0.0, max((coverage(question, h.text) for h in hits), default=0.0)]


@dataclass
class Gate:
    weights: list[float]
    bias: float
    mean: list[float]
    std: list[float]
    threshold: float

    @classmethod
    def load(cls, path: Path = GATE) -> "Gate":
        d = json.loads(path.read_text())
        return cls(d["weights"], d["bias"], d["mean"], d["std"], d["threshold"])

    def probability(self, feats: list[float]) -> float:
        z = self.bias + sum(w * (f - m) / s for w, f, m, s in zip(self.weights, feats, self.mean, self.std))
        return 1 / (1 + math.exp(-z))

    def answerable(self, feats: list[float]) -> bool:
        return self.probability(feats) >= self.threshold


def fit_gate(rows: list[dict], retriever: Retriever, k: int = 4, threshold: float = 0.5) -> dict:
    """Fit on grounded_qa rows; report 5-fold cross-validated accuracy. The gate decides whether the CORPUS can
    answer, so a context-swap question (answerable from its own paper) is a positive and only out-of-corpus
    questions are negatives."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = np.array([features(r["question"], retriever.search(r["question"], k)) for r in rows])
    y = np.array([int(r["example_type"] != "unanswerable_out_of_corpus") for r in rows])
    mean, std = X.mean(0), X.std(0) + 1e-9
    Z = (X - mean) / std
    clf = LogisticRegression().fit(Z, y)
    cv = cross_val_predict(LogisticRegression(), Z, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           method="predict_proba")[:, 1] >= threshold
    return {"method": "logreg[top1_score, best_coverage]", "feature_names": ["top1_score", "best_coverage"],
            "weights": clf.coef_[0].tolist(), "bias": float(clf.intercept_[0]), "mean": mean.tolist(),
            "std": std.tolist(), "threshold": threshold, "top_k": k, "embedding_model": EMBEDDING_MODEL,
            "n": len(rows), "n_in_corpus": int(y.sum()),
            "cv5": {"in_corpus_kept": float(cv[y == 1].mean()), "out_of_corpus_abstained": float((~cv[y == 0]).mean())}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ps3_llm.retrieval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit-gate")
    f.add_argument("--rows", type=Path, default=ROOT / "data/real_train/v1/grounded_qa_train_clean.jsonl")
    f.add_argument("--out", type=Path, default=GATE)
    s = sub.add_parser("search")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=4)
    args = ap.parse_args(argv)
    r = Retriever()
    if args.cmd == "fit-gate":
        rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
        gate = {**fit_gate(rows, r), "trained_on": str(args.rows.relative_to(ROOT))}
        args.out.write_text(json.dumps(gate, indent=2) + "\n")
        print(json.dumps(gate, indent=2))
    else:
        for h in r.search(args.question, args.k):
            print(f"{h.score:.3f}  {h.chunk_id}  p.{h.page_number}  {h.text[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
