#!/usr/bin/env bash
# Serve a model with SGLang (OpenAI-compatible API at http://$HOST:$PORT/v1).
#   scripts/serve_sglang.sh <hf-model-id-or-local-path> [extra sglang args...]
# On Apple Silicon this uses SGLang's MLX runtime (SGLANG_USE_MLX=1); on Linux/CUDA the
# MLX variable is not set and SGLang uses its default CUDA path.
set -euo pipefail
MODEL="${1:?usage: serve_sglang.sh <model> [extra args]}"; shift || true
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${SGLANG_PYTHON:-$HERE/.venv-sglang/bin/python}"
HOST="${HOST:-127.0.0.1}"; PORT="${PORT:-30000}"; CTX="${CTX:-4096}"
EXTRA=()
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  export SGLANG_USE_MLX=1
  EXTRA+=(--disable-cuda-graph)
fi
exec "$PY" -m sglang.launch_server --model-path "$MODEL" --host "$HOST" --port "$PORT" \
  --context-length "$CTX" "${EXTRA[@]}" "$@"
