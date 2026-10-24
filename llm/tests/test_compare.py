"""ps3_llm.compare: --results-dir redirection (committed results untouched) and clear errors for bad run dirs."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ps3_llm import compare

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TAGS = ["baseline-qwen2.5-1.5b", "sft-qwen2.5-1.5b"]
COMMITTED_REPORT = RES / "v2" / "compare_baseline_vs_sft.md"
pytestmark = pytest.mark.skipif(not (RES / "v2" / TAGS[0]).exists(), reason="committed v2 results not present")


def _digest(root: Path) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def copied(tmp_path):
    """The two committed v2 runs copied into a scratch results dir under their original tag names."""
    res = tmp_path / "res"
    for t in TAGS:
        shutil.copytree(RES / "v2" / t, res / t)
    return res


def test_results_dir_reproduces_committed_report_without_touching_results(copied, capsys):
    before = _digest(RES)
    out = compare.main(*TAGS, results_dir=copied)
    capsys.readouterr()
    assert out == copied / f"compare_{TAGS[0]}_vs_{TAGS[1]}.md"     # default report lands in --results-dir
    assert out.read_text() == COMMITTED_REPORT.read_text()
    assert _digest(RES) == before


def test_default_results_dir_is_unchanged(tmp_path, capsys):
    out = compare.main("v2/" + TAGS[0], "v2/" + TAGS[1], tmp_path / "cmp.md")
    capsys.readouterr()
    assert out.read_text() == COMMITTED_REPORT.read_text()


def test_cli_results_dir_flag(copied):
    before = _digest(RES)
    p = subprocess.run([sys.executable, "-m", "ps3_llm.compare", *TAGS, "--results-dir", str(copied)],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert (copied / f"compare_{TAGS[0]}_vs_{TAGS[1]}.md").read_text() == COMMITTED_REPORT.read_text()
    assert _digest(RES) == before


def test_missing_run_dir(tmp_path):
    with pytest.raises(SystemExit, match="missing summary.json, per_example.jsonl"):
        compare.main("nope", "nope", results_dir=tmp_path)


def test_missing_single_file(copied):
    (copied / TAGS[1] / "errors.json").unlink()
    with pytest.raises(SystemExit, match=r"'sft-qwen2.5-1.5b'.*missing errors.json"):
        compare.main(*TAGS, results_dir=copied)


def test_corrupt_file(copied):
    (copied / TAGS[0] / "summary.json").write_text("{not json")
    with pytest.raises(SystemExit, match="unreadable result files"):
        compare.main(*TAGS, results_dir=copied)


def test_v1_run_is_rejected(copied):
    shutil.copytree(RES / TAGS[0], copied / "v1-run")             # results/<tag>/ holds v1 summaries
    with pytest.raises(SystemExit, match="not v2-scored"):
        compare.main("v1-run", TAGS[1], results_dir=copied)


def test_runs_on_different_items(copied):
    p = copied / TAGS[1] / "per_example.jsonl"
    p.write_text("".join(p.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(SystemExit, match="not on the same items"):
        compare.main(*TAGS, results_dir=copied)


def test_cli_error_exit_code(tmp_path):
    p = subprocess.run([sys.executable, "-m", "ps3_llm.compare", "a", "b", "--results-dir", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 1 and "missing summary.json" in p.stderr and "Traceback" not in p.stderr


def test_evaluate_results_dir_writes_only_there(tmp_path):
    before = _digest(RES)
    shutil.copytree(RES / TAGS[1], tmp_path / TAGS[1])           # --rescore-from reads from --results-dir too
    p = subprocess.run([sys.executable, "-m", "ps3_llm.evaluate", "--rescore-from", TAGS[1], "--tag", "re",
                        "--results-dir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    committed = json.loads((RES / "v2" / TAGS[1] / "summary.json").read_text())
    assert json.loads((tmp_path / "re" / "summary.json").read_text())["serving"] == committed["serving"]
    assert _digest(RES) == before
