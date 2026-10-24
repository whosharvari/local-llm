"""Stage 2 — LoRA SFT with AlignTune (TRL backend), then merge for SGLang serving.

    python -m ps3_llm.train --config configs/sft_qwen2.5-1.5b.yaml
    python -m ps3_llm.train --config configs/sft_qwen2.5-1.5b.yaml --merge-only   # re-merge an existing adapter

The SFT rows from ps3_llm.curate are chat `messages` and are passed through as-is with
`assistant_only_loss=True`, so the loss covers only the assistant answer, not the long evidence
prompt (TRL swaps in its Qwen2.5 training chat template with {% generation %} markers for this).
Note: we deliberately avoid TRL's conversational prompt/completion columns here — AlignTune
0.1.28's conversational normaliser stringifies a list-valued `prompt` into the user turn. The trained LoRA
adapter is merged into the base weights and saved as a plain HF checkpoint that SGLang loads
directly (scripts/serve_sglang.sh <merged dir>).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def to_messages(src: Path, dst: Path) -> int:
    """Keep only the `messages` column (drops kind/doc_id bookkeeping fields)."""
    rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
    return len(rows)


def train(cfg: dict) -> Path:
    from aligntune.core.backend_factory import create_sft_trainer

    out = ROOT / cfg["output_dir"]
    data_dir = out / "data"
    n_train = to_messages(ROOT / cfg["train_file"], data_dir / "train.jsonl")
    n_dev = to_messages(ROOT / cfg["dev_file"], data_dir / "validation.jsonl") if cfg.get("dev_file") else 0
    print(f"train={n_train} dev={n_dev}")

    t0 = time.time()
    trainer = create_sft_trainer(
        model_name=cfg["base_model"],
        dataset_name=str(data_dir),          # local folder with train.jsonl (+ validation.jsonl)
        backend="trl",                       # Unsloth requires CUDA; TRL runs on Apple MPS
        output_dir=str(out / "adapter"),
        num_epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        max_seq_length=cfg["max_seq_length"],
        use_peft=True,
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        lora_target_modules=cfg.get("lora_target_modules"),
        precision=cfg.get("precision"),
        gradient_accumulation_steps=cfg["grad_accum"],
        device_map=cfg.get("device_map"),
        use_gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        assistant_only_loss=True,
        seed=cfg["seed"],
        split="train",
        report_to="none",
        **cfg.get("extra", {}),
    )
    result = trainer.train()
    elapsed = time.time() - t0
    param = next(trainer.model.parameters())
    print(f"[ps3] trained model dtype={param.dtype} device={param.device}", flush=True)
    (out / "train_summary.json").write_text(json.dumps(
        {"elapsed_s": round(elapsed, 1), "n_train": n_train, "n_dev": n_dev, "config": cfg,
         "model_dtype": str(param.dtype), "device": str(param.device),
         "result": {k: v for k, v in (result or {}).items() if isinstance(v, (int, float, str))}},
        indent=2, default=str))
    return out / "adapter"


def merge(cfg: dict) -> Path:
    """Merge the LoRA adapter into the base model -> plain HF checkpoint for SGLang."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = ROOT / cfg["output_dir"]
    adapter = out / "adapter"
    merged = out / "merged"
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    model.save_pretrained(str(merged), safe_serialization=True)
    AutoTokenizer.from_pretrained(cfg["base_model"]).save_pretrained(str(merged))
    print(f"merged checkpoint -> {merged}")
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    # Cap the MPS caching allocator (fraction of the recommended max working set) so it releases
    # cached blocks instead of growing into swap on a 16 GB machine. Must be set before torch import.
    if cfg.get("mps_high_watermark_ratio") is not None:
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", str(cfg["mps_high_watermark_ratio"]))
        # the low watermark (default 1.4) must not exceed the high one
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(min(0.5, cfg["mps_high_watermark_ratio"])))
    if not args.merge_only:
        train(cfg)
    if not args.no_merge:
        merge(cfg)


if __name__ == "__main__":
    main()
