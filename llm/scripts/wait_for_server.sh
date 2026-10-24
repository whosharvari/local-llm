#!/usr/bin/env bash
# Block until the SGLang server answers /v1/models (default timeout 1200s).
URL="${1:-http://127.0.0.1:30000}"; TIMEOUT="${2:-1200}"
for ((i=0; i<TIMEOUT; i+=5)); do
  curl -sf "$URL/v1/models" >/dev/null && { curl -s "$URL/v1/models"; echo; exit 0; }
  sleep 5
done
echo "server at $URL not ready after ${TIMEOUT}s" >&2; exit 1
