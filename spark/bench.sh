#!/usr/bin/env bash
# Quick llama-bench numbers for a profile: prompt processing and generation
# speed at an empty context and at 8k depth. Stop the server first; both will
# not fit at once for the big models.
#   spark/bench.sh [model-key]
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_env
key=${1:-$MODEL}
load_model "$key"
gguf=$("$ROOT/spark/resolve-gguf.sh" "$key")
bench=$(llama_bin llama-bench)
log "benchmarking $key ($gguf)"
"$bench" -m "$gguf" -ngl 999 -fa 1 -p 512 -n 128 -d 0,8192 -r 2 \
    -ctk "$KV_TYPE" -ctv "$KV_TYPE" 2>&1 | tee -a "$LOG_DIR/bench-$key.log"
