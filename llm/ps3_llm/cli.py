"""JSON-in / JSON-out entry point for integration (no Python import of this package needed).

    echo '<GroundedQuery JSON>' | python -m ps3_llm.cli --adapter mock
    python -m ps3_llm.cli --adapter sglang --base-url http://127.0.0.1:30000/v1 < request.json

Accepts one request object or a JSON list of them; prints GroundedAnswer JSON (object or list).
"""

from __future__ import annotations

import argparse
import json
import sys

from .adapters import get_adapter
from .schemas import GroundedQuery


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["mock", "sglang"], default="mock")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    payload = json.load(sys.stdin)
    single = isinstance(payload, dict)
    queries = [GroundedQuery.from_dict(p) for p in ([payload] if single else payload)]
    kwargs = {} if args.adapter == "mock" else {"base_url": args.base_url, "model": args.model}
    answers = [a.to_dict() for a in get_adapter(args.adapter, **kwargs).answer_batch(queries)]
    json.dump(answers[0] if single else answers, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
