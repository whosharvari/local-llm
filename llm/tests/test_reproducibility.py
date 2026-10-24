"""Committed v2 results: replay is non-destructive, the comparison report builds, and the independent
verifier (no ps3_llm import) agrees with the reported metrics."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ps3_llm import compare
from ps3_llm.evaluate import ReplayAdapter, load_examples, run

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TAGS = ["baseline-qwen2.5-1.5b", "sft-qwen2.5-1.5b"]
pytestmark = pytest.mark.skipif(not (RES / "v2" / TAGS[0]).exists(), reason="committed v2 results not present")


def _digest(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_replay_matches_committed_v2_and_leaves_history_untouched(tmp_path):
    hist = sorted(p for p in (RES / TAGS[0]).rglob("*") if p.is_file())
    before = _digest(hist)
    examples = load_examples(ROOT / "data/mock/heldout_eval.jsonl")
    s = run(ReplayAdapter(RES / TAGS[0]), examples, tmp_path / "replay", ROOT / "data/mock/heldout_eval.jsonl",
            {}, ROOT / "data/mock/heldout_adjudication.json", measure_server=False)
    assert _digest(hist) == before
    committed = json.loads((RES / "v2" / TAGS[0] / "summary.json").read_text())
    for m in ("serving", "intent"):
        assert {k: v for k, v in s[m].items()} == committed[m]


def test_replay_refuses_existing_v2_dir():
    examples = load_examples(ROOT / "data/mock/heldout_eval.jsonl")
    with pytest.raises(SystemExit):
        run(ReplayAdapter(RES / TAGS[0]), examples, RES / "v2" / TAGS[0], ROOT / "data/mock/heldout_eval.jsonl",
            {}, ROOT / "data/mock/heldout_adjudication.json", measure_server=False)


def test_comparison_report_generation(tmp_path):
    out = compare.main("v2/" + TAGS[0], "v2/" + TAGS[1], tmp_path / "cmp.md")
    text = out.read_text()
    assert "## Item flips — intent" in text and "## Item flips — serving" in text
    assert "2 improved, 2 regressed" in text and "exact McNemar p = 1.000" in text
    assert compare.mcnemar_p(3, 1) == 0.625 and compare.mcnemar_p(0, 0) == 1.0


def test_independent_verifier_agrees():
    proc = subprocess.run([sys.executable, "-I", "verification/verify_v2.py", ".", *TAGS], cwd=ROOT,
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and "MISMATCHES: 0" in proc.stdout, proc.stdout + proc.stderr
