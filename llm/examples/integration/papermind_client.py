"""Minimal Papermind -> model-module client. Uses ONLY the JSON CLI contract (no ps3_llm imports).

    python papermind_client.py request.json [--adapter sglang|mock] [--base-url URL]

Set PS3_LLM_DIR to the model module directory (default: two levels up) and PS3_PYTHON to its venv python.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

LLM_DIR = Path(os.environ.get("PS3_LLM_DIR", Path(__file__).resolve().parents[2]))
PYTHON = os.environ.get("PS3_PYTHON", str(LLM_DIR.parent / ".venv" / "bin" / "python"))


def ask(requests, adapter="sglang", base_url="http://127.0.0.1:30000/v1", timeout=600):
    """Send one request dict or a list of them; returns GroundedAnswer dict(s) in the same order."""
    proc = subprocess.run([PYTHON, "-m", "ps3_llm.cli", "--adapter", adapter, "--base-url", base_url],
                          input=json.dumps(requests), capture_output=True, text=True, cwd=LLM_DIR, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"ps3_llm.cli exit {proc.returncode}: {proc.stdout}{proc.stderr}")
    return json.loads(proc.stdout)


def render(ans, evidence_by_id):
    """How Papermind should consume a response."""
    if ans["abstained"]:
        if ans["abstain_reason"].startswith(("inference_error", "invalid_request")):
            return f"[system error] {ans['abstain_reason']}"           # not a model decision: retry / alert
        return "The documents do not contain enough information to answer this question."
    sources = ", ".join(f"{c} ({evidence_by_id[c].get('source', '')})" for c in ans["citations"])
    return f"{ans['answer']}\nSources: {sources}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("--adapter", default="sglang")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    a = ap.parse_args()
    req = json.loads(Path(a.request).read_text())
    ans = ask(req, a.adapter, a.base_url)
    print(json.dumps(ans, indent=2))
    ev = {e["chunk_id"]: e for e in req.get("evidence", []) if isinstance(e, dict) and "chunk_id" in e}
    print("\n--- rendered ---\n" + render(ans, ev), file=sys.stderr)
