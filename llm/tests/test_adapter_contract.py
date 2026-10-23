"""Adapter and JSON CLI contract: failures are structured answers, never exceptions."""

import json
import subprocess
import sys
from pathlib import Path

from ps3_llm.adapters import SGLangAdapter, answer_from_text
from ps3_llm.schemas import GroundedQuery, validate_request

ROOT = Path(__file__).resolve().parent.parent
DEAD = "http://127.0.0.1:9/v1"   # discard port: connection refused
REQ = {"query_id": "q1", "question": "How long?", "evidence": [{"chunk_id": "a#1", "text": "Fifteen days."}]}


def cli(payload, *args):
    proc = subprocess.run([sys.executable, "-m", "ps3_llm.cli", *args], input=payload, cwd=ROOT,
                          capture_output=True, text=True, timeout=60)
    return proc.returncode, proc.stdout, proc.stderr


def _check_error_answer(a, qid="q1"):
    assert a["abstained"] is True and a["answer"] == "" and a["citations"] == []
    assert a["abstain_reason"].startswith("inference_error: ") and a["parse_ok"] is False
    assert a["query_id"] == qid


def test_adapter_constructs_without_server_and_returns_error_answer():
    adapter = SGLangAdapter(base_url=DEAD, timeout=2)          # audit: used to raise URLError here
    _check_error_answer(adapter.answer(GroundedQuery.from_dict(REQ)).to_dict())


def test_adapter_with_explicit_model_returns_error_answer():
    adapter = SGLangAdapter(base_url=DEAD, model="m", timeout=2)
    _check_error_answer(adapter.answer(GroundedQuery.from_dict(REQ)).to_dict())


def test_cli_down_server_without_model_flag():
    code, out, err = cli(json.dumps(REQ), "--adapter", "sglang", "--base-url", DEAD, "--timeout", "2")
    assert code == 0 and "Traceback" not in err
    _check_error_answer(json.loads(out))


def test_answer_from_text_never_returns_invented_or_uncited():
    q = GroundedQuery.from_dict(REQ)
    a = answer_from_text('{"abstain": false, "answer": "Twenty days.", "citations": ["z#9"]}', q, "m", 0.1)
    assert a.abstained and a.citations == [] and a.abstain_reason == "no_valid_citation"
    assert a.usage["invalid_citations"] == ["z#9"] and "invalid_citation_ids" in a.warnings
    ok = answer_from_text('{"abstain": false, "answer": "Fifteen days.", "citations": ["a#1", "x"]}', q, "m", 0.1)
    assert not ok.abstained and ok.citations == ["a#1"]


def test_validate_request():
    assert validate_request(REQ) == []
    assert validate_request({**REQ, "evidence": []}) == []
    bad = {"question": "", "evidence": [{"chunk_id": "[a]", "text": 1}, {"chunk_id": "b", "text": "x"},
                                        {"chunk_id": "b", "text": "y"}]}
    errs = " | ".join(validate_request(bad))
    assert "question" in errs and "brackets" in errs and "text must be a string" in errs and "duplicated" in errs
    assert validate_request({"question": "q"}) == ["evidence must be a list (use [] when nothing was retrieved)"]


def test_cli_invalid_request_is_structured():
    code, out, _ = cli(json.dumps([REQ, {"query_id": "bad", "question": "x"}]), "--adapter", "mock")
    good, bad = json.loads(out)
    assert code == 0 and good["query_id"] == "q1"
    assert bad["abstained"] and bad["abstain_reason"].startswith("invalid_request: ") and bad["query_id"] == "bad"


def test_cli_non_json_stdin_exit_code():
    code, out, _ = cli("not json", "--adapter", "mock")
    assert code == 2 and "error" in json.loads(out)


def test_cli_contract_roundtrip():
    req = (ROOT / "data/mock/example_request.json").read_text()
    code, out, _ = cli(req, "--adapter", "mock")
    ans = json.loads(out)
    assert code == 0 and ans["query_id"] == "demo-1" and ans["contract_version"] == "1.1"
    assert set(ans["citations"]) <= {e["chunk_id"] for e in json.loads(req)["evidence"]}
    assert ans["abstained"] is False and "15 business days" in ans["answer"]
    empty = json.dumps({"query_id": "demo-2", "question": "What is the dress code?", "evidence": []})
    ans2 = json.loads(cli(empty)[1])
    assert ans2["abstained"] and ans2["answer"] == "" and ans2["citations"] == []
