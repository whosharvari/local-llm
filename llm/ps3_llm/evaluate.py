"""Run a model adapter over a held-out set and write reproducible results.

    python -m ps3_llm.evaluate --adapter mock --tag mock
    python -m ps3_llm.evaluate --adapter sglang --base-url http://127.0.0.1:30000/v1 --tag baseline-qwen2.5-1.5b

Outputs (results/<tag>/): predictions.jsonl, per_example.jsonl, summary.json, errors.json, run_info.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

from .adapters import get_adapter
from .metrics import aggregate, error_buckets, score_example
from .schemas import EvalExample

ROOT = Path(__file__).resolve().parent.parent


def load_examples(path: Path) -> list[EvalExample]:
    return [EvalExample.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def _sglang_server_mem_gb() -> float | None:
    """Memory of the SGLang scheduler process, which holds the weights and KV cache.

    On macOS uses `footprint`, which includes MLX's Metal (unified-memory) allocations that plain
    RSS misses. Elsewhere falls back to RSS from `ps`.
    """
    try:
        pids = subprocess.run(["pgrep", "-f", "sglang::scheduler"], capture_output=True, text=True).stdout.split()
        if not pids:
            return None
        if platform.system() == "Darwin":
            total_mb = 0.0
            for pid in pids:
                out = subprocess.run(["footprint", "-p", pid], capture_output=True, text=True).stdout
                for tok in out.split("Footprint:")[1:2]:
                    val, unit = tok.split()[:2]
                    total_mb += float(val) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}.get(unit, 1)
            return round(total_mb / 1024, 2) if total_mb else None
        out = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(pids)], capture_output=True, text=True).stdout
        return round(sum(int(x) for x in out.split()) / 1024 / 1024, 2)
    except (OSError, ValueError):
        return None


def run(adapter, examples: list[EvalExample], out_dir: Path, data_path: Path, adapter_args: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    preds = adapter.answer_batch([ex.query for ex in examples])
    wall = time.perf_counter() - t0

    rows = [score_example(ex, p) for ex, p in zip(examples, preds)]
    summary = aggregate(rows)
    summary["wall_time_s"] = round(wall, 2)
    summary["throughput_qps"] = round(len(examples) / wall, 3) if wall else None
    toks = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
    summary["mean_completion_tokens"] = round(sum(toks) / len(toks), 1) if toks else None
    summary["sglang_server_mem_gb"] = _sglang_server_mem_gb()

    (out_dir / "predictions.jsonl").write_text("".join(json.dumps(p.to_dict(), ensure_ascii=False) + "\n" for p in preds))
    (out_dir / "per_example.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "errors.json").write_text(json.dumps(error_buckets(rows), indent=2))
    (out_dir / "run_info.json").write_text(json.dumps({
        "adapter": adapter.name,
        "adapter_args": adapter_args,
        "data": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "n_examples": len(examples),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }, indent=2))
    return summary


class ReplayAdapter:
    """Re-parses the raw model outputs saved by an earlier run (no model call), e.g. to apply the
    lenient abstention parser identically to baseline and fine-tuned outputs."""

    def __init__(self, source_dir: Path, lenient_abstain: bool):
        self.saved = {p["query_id"]: p for p in map(json.loads, (source_dir / "predictions.jsonl").read_text().splitlines())}
        self.lenient = lenient_abstain
        self.name = f"replay:{source_dir.name}" + (":lenient" if lenient_abstain else "")

    def answer(self, query):
        from .prompting import parse_model_output
        from .schemas import GroundedAnswer

        s = self.saved[query.query_id]
        parsed = parse_model_output(s["raw_output"], {c.chunk_id for c in query.evidence}, self.lenient)
        return GroundedAnswer(
            answer=parsed["answer"], citations=parsed["citations"], abstained=parsed["abstained"],
            abstain_reason="model abstained" if parsed["abstained"] else "", query_id=query.query_id,
            model=s["model"], latency_s=s["latency_s"], raw_output=s["raw_output"], parse_ok=parsed["parse_ok"],
            usage={**(s.get("usage") or {}), "invalid_citations": parsed["invalid_citations"]},
        )

    def answer_batch(self, queries):
        return [self.answer(q) for q in queries]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", choices=["mock", "sglang"], default="mock")
    ap.add_argument("--data", type=Path, default=ROOT / "data/mock/heldout_eval.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    ap.add_argument("--model", default=None, help="served model id (default: first id from /v1/models)")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--constrained", action="store_true", help="JSON-schema constrained decoding via SGLang")
    ap.add_argument("--rescore-from", default=None, help="re-parse saved raw outputs of this results tag")
    ap.add_argument("--lenient-abstain", action="store_true", help="with --rescore-from: refusal text counts as abstain")
    args = ap.parse_args()

    examples = load_examples(args.data)
    if args.rescore_from:
        src = args.results_dir / args.rescore_from
        adapter_args = {"rescore_from": args.rescore_from, "lenient_abstain": args.lenient_abstain}
        summary = run(ReplayAdapter(src, args.lenient_abstain), examples, args.results_dir / args.tag,
                      args.data, adapter_args)
        # timing/resource numbers belong to the original generation run, not the replay
        orig = json.loads((src / "summary.json").read_text())
        for k in ("wall_time_s", "throughput_qps", "sglang_server_mem_gb", "latency_p50_s", "latency_p95_s",
                  "latency_mean_s", "mean_completion_tokens"):
            summary[k] = orig.get(k)
        (args.results_dir / args.tag / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return
    if args.adapter == "mock":
        adapter_args: dict = {}
    else:
        adapter_args = dict(base_url=args.base_url, model=args.model, max_tokens=args.max_tokens,
                            concurrency=args.concurrency, constrained=args.constrained, temperature=0.0)
    adapter = get_adapter(args.adapter, **adapter_args)
    summary = run(adapter, examples, args.results_dir / args.tag, args.data, adapter_args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
