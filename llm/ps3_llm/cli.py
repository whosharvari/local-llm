"""JSON-in / JSON-out entry point for integration (no Python import of this package needed).

    echo '<GroundedQuery JSON>' | python -m ps3_llm.cli --adapter mock
    python -m ps3_llm.cli --adapter sglang --base-url http://127.0.0.1:30000/v1 < request.json

Accepts one request object or a JSON list of them; prints GroundedAnswer JSON (object or list, same
order). Never raises on bad input or a down server: invalid requests get abstain_reason
"invalid_request: ...", server failures "inference_error: ...". Exit code is 0 when every item was
processed (including abstentions and inference errors), 2 when stdin is not JSON.
"""

from __future__ import annotations

import argparse
import json
import sys

from .adapters import get_adapter
from .schemas import INVALID_REQUEST_PREFIX, GroundedAnswer, GroundedQuery, validate_request


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["mock", "sglang"], default="mock")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default=None, help="served model id; default: discovered from /v1/models")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        json.dump({"error": f"stdin is not valid JSON: {e}"}, sys.stdout)
        sys.stdout.write("\n")
        return 2
    single = not isinstance(payload, list)
    items = [payload] if single else payload

    kwargs = {} if args.adapter == "mock" else {"base_url": args.base_url, "model": args.model,
                                                "timeout": args.timeout}
    adapter = get_adapter(args.adapter, **kwargs)
    results: list[dict | None] = [None] * len(items)
    valid_idx, queries = [], []
    for i, item in enumerate(items):
        errs = validate_request(item)
        if errs:
            qid = str(item.get("query_id", "")) if isinstance(item, dict) else ""
            results[i] = GroundedAnswer(answer="", citations=[], abstained=True, parse_ok=False, query_id=qid,
                                        abstain_reason=INVALID_REQUEST_PREFIX + "; ".join(errs),
                                        model=adapter.name).to_dict()
        else:
            valid_idx.append(i)
            queries.append(GroundedQuery.from_dict(item))
    for i, ans in zip(valid_idx, adapter.answer_batch(queries) if queries else []):
        results[i] = ans.to_dict()

    json.dump(results[0] if single else results, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
