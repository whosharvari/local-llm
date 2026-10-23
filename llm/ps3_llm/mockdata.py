"""Build the deterministic held-out eval set from the mock spec (no model involved).

    python -m ps3_llm.mockdata  ->  data/mock/heldout_eval.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "mock"


def load_heldout_chunks(path: Path = DATA / "heldout_docs.json") -> dict[str, dict]:
    docs = json.loads(path.read_text())
    out = {}
    for key, doc in docs.items():
        if key.startswith("_"):
            continue
        for cid, text in doc["chunks"].items():
            out[cid] = {"chunk_id": cid, "text": text, "source": doc["source"]}
    return out


def build_heldout(spec_path: Path = DATA / "heldout_spec.json") -> list[dict]:
    chunks = load_heldout_chunks()
    rows = []
    for item in json.loads(spec_path.read_text()):
        evidence = [chunks[c] for c in item["ev"]]
        assert set(item["cit"]) <= set(item["ev"]), item["id"]
        rows.append({
            "query_id": item["id"],
            "question": item["q"],
            "evidence": evidence,
            "answerable": bool(item["cit"]),
            "gold_answer": item["gold"],
            "gold_citations": item["cit"],
            "category": item["category"],
        })
    return rows


def main() -> None:
    rows = build_heldout()
    out = DATA / "heldout_eval.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} examples -> {out}")


if __name__ == "__main__":
    main()
