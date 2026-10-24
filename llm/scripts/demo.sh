#!/usr/bin/env bash
# End-to-end demo: serve the final checkpoint with SGLang (localhost only), ask questions through
# retrieval -> prompt -> model -> checked answer, log every run record, stop the server.
#   scripts/demo.sh [checkpoint dir] [log file]
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-$PWD/outputs/sft-real-v1-qwen2.5-1.5b/merged}"
LOG="${2:-results/demo/demo_$(date +%Y%m%d_%H%M%S).jsonl}"
NAME=ps3-qwen2.5-1.5b-grounded
PY=../.venv/bin/python
[[ -f "$CKPT/model.safetensors" ]] || { echo "checkpoint not found: $CKPT (train it or pass a path)" >&2; exit 1; }
mkdir -p "$(dirname "$LOG")"
SERVE_LOG="$(mktemp -t ps3_demo.XXXXXX)"
scripts/serve_sglang.sh "$CKPT" --served-model-name "$NAME" >"$SERVE_LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
scripts/wait_for_server.sh http://127.0.0.1:30000 900 >/dev/null
echo "checkpoint: $CKPT  sha256: $(shasum -a 256 "$CKPT/model.safetensors" | cut -c1-16)…"
while IFS= read -r q; do
  echo; echo "Q: $q"
  HF_HUB_OFFLINE=1 $PY -m ps3_llm.pipeline "$q" --model "$NAME" --gate data/real_train/v1/retrieval_gate.json --log "$LOG" 2>/dev/null
done <<'EOF'
How many layers does the BERT-base model have?
How does the Mamba architecture make state space model parameters input-dependent?
How many GPUs were used to train the ImageNet classification models in the ResNet paper?
EOF
echo; echo "run records: $LOG"
