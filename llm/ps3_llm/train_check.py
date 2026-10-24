"""Pre- and post-training checks for the AlignTune SFT run.

    python -m ps3_llm.train_check data --config configs/sft_real_v1_qwen2.5-1.5b.yaml
        tokenises every train/validation example with TRL's training chat template (the one used for
        assistant_only_loss), and checks that none exceeds max_seq_length (no truncation) and that the loss
        mask covers exactly the assistant JSON target
    python -m ps3_llm.train_check weights --config configs/sft_real_v1_qwen2.5-1.5b.yaml
        compares the merged checkpoint with the base model tensor by tensor (which weights changed, by how much)
    python -m ps3_llm.train_check valloss --config configs/sft_real_v1_qwen2.5-1.5b.yaml
        mean assistant-token cross-entropy on the validation file for the base and the merged model (AlignTune's
        generation SFT trainer does not evaluate in-loop, so this is the held-out loss comparison)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def check_data(cfg: dict) -> dict:
    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    template = get_training_chat_template(tok)
    out = {}
    for name in ("train_file", "dev_file"):
        if not cfg.get(name):
            continue
        rows = [json.loads(line) for line in (ROOT / cfg[name]).read_text().splitlines() if line.strip()]
        lengths, masked, bad_mask = [], [], []
        for i, r in enumerate(rows):
            enc = tok.apply_chat_template(r["messages"], chat_template=template, tokenize=True, return_dict=True,
                                          return_assistant_tokens_mask=True)
            ids, mask = enc["input_ids"], enc["assistant_masks"]
            lengths.append(len(ids))
            target_ids = [t for t, m in zip(ids, mask) if m]
            masked.append(len(target_ids))
            decoded = tok.decode(target_ids).replace("<|im_end|>", "").strip()
            if decoded != r["messages"][-1]["content"].strip():
                bad_mask.append(i)
        out[name] = {"n": len(rows), "max_tokens": max(lengths), "mean_tokens": round(sum(lengths) / len(lengths), 1),
                     "over_max_seq_length": sum(n > cfg["max_seq_length"] for n in lengths),
                     "mean_loss_tokens": round(sum(masked) / len(masked), 1),
                     "loss_tokens_share": round(sum(masked) / sum(lengths), 4),
                     "mask_mismatch_rows": bad_mask[:10], "n_mask_mismatch": len(bad_mask)}
    return out


def check_weights(cfg: dict) -> dict:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    merged = load_file(str(ROOT / cfg["output_dir"] / "merged" / "model.safetensors"))
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16).state_dict()
    changed, unchanged, rel = [], 0, {}
    for k, v in merged.items():
        b = base.get(k)
        if b is None:
            continue
        d = (v.float() - b.float()).norm() / (b.float().norm() + 1e-12)
        if d > 0:
            changed.append(k)
            rel[k.split(".")[-2]] = max(rel.get(k.split(".")[-2], 0.0), round(float(d), 6))
        else:
            unchanged += 1
    targets = set(cfg.get("lora_target_modules") or [])
    unexpected = [k for k in changed if k.split(".")[-2] not in targets]
    return {"tensors_changed": len(changed), "tensors_unchanged": unchanged,
            "max_relative_change_by_module": rel, "changed_outside_lora_targets": unexpected[:10]}


def validation_loss(cfg: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    template = get_training_chat_template(tok)
    rows = [json.loads(line) for line in (ROOT / cfg["dev_file"]).read_text().splitlines() if line.strip()]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    out = {"n": len(rows), "file": cfg["dev_file"]}
    for name, path in (("base", cfg["base_model"]), ("fine_tuned", str(ROOT / cfg["output_dir"] / "merged"))):
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for r in rows:
                enc = tok.apply_chat_template(r["messages"], chat_template=template, tokenize=True, return_dict=True,
                                              return_assistant_tokens_mask=True)
                ids = torch.tensor([enc["input_ids"]], device=device)
                mask = torch.tensor(enc["assistant_masks"], device=device)[1:].bool()
                logits = model(ids).logits[0, :-1].float()
                nll = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="none")[mask]
                total, count = total + float(nll.sum()), count + int(mask.sum())
        out[name] = round(total / count, 4)
        del model
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ps3_llm.train_check")
    ap.add_argument("what", choices=["data", "weights", "valloss"])
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    res = {"data": check_data, "weights": check_weights, "valloss": validation_loss}[args.what](cfg)
    print(json.dumps(res, indent=2))
    if args.what == "data":
        return 1 if any(v["over_max_seq_length"] or v["n_mask_mismatch"] for v in res.values()) else 0
    if args.what == "weights":
        return 1 if not res["tensors_changed"] or res["changed_outside_lora_targets"] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
