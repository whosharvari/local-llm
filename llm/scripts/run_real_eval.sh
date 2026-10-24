#!/usr/bin/env bash
# Serve one model with SGLang, evaluate it on a real-corpus split with the locked settings, write the
# real-corpus report, record the checkpoint hash, and stop the server (also on error).
#   scripts/run_real_eval.sh <model path or HF id> <served name> <results tag> <data.jsonl> <adjudication.json>
# Locked settings (identical for every model): temperature 0, max_tokens 384, concurrency 4, 4 passages per item,
# context length 4096, SGLang defaults otherwise. Results go to results/<tag> (evaluate refuses existing dirs).
set -euo pipefail
MODEL="${1:?model}"; NAME="${2:?served name}"; TAG="${3:?results tag}"; DATA="${4:?data}"; ADJ="${5:?adjudication}"
cd "$(dirname "$0")/.."
PY=../.venv/bin/python
PORT="${PORT:-30000}"
[[ -e "results/$TAG" ]] && { echo "results/$TAG already exists" >&2; exit 1; }
LOG="$(mktemp -t ps3_serve.XXXXXX)"
PORT="$PORT" scripts/serve_sglang.sh "$MODEL" --served-model-name "$NAME" >"$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null || true' EXIT
scripts/wait_for_server.sh "http://127.0.0.1:$PORT" 900 >/dev/null
$PY -m ps3_llm.evaluate --adapter sglang --base-url "http://127.0.0.1:$PORT/v1" --model "$NAME" \
    --max-tokens 384 --concurrency 4 --data "$DATA" --adjudication "$ADJ" --tag "$TAG" >/dev/null
$PY -m ps3_llm.real_corpus report "results/$TAG" >/dev/null
if [[ -f "$MODEL/model.safetensors" ]]; then
  shasum -a 256 "$MODEL/model.safetensors" | awk '{print $1"  model.safetensors"}' >"results/$TAG/checkpoint_sha256.txt"
else
  echo "$MODEL (Hugging Face Hub id; weights downloaded by SGLang)" >"results/$TAG/checkpoint_sha256.txt"
fi
cp "$LOG" "results/$TAG/serve.log"
echo "wrote results/$TAG"
