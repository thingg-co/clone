#!/usr/bin/env bash
# Fallback path: Ollama from the release tarball, kept under .tools/ollama with
# models under models/ollama. No sudo, nothing installed system-wide.
#   spark/ollama.sh install
#   spark/ollama.sh serve        # foreground, :11434, OpenAI API at /v1
#   spark/ollama.sh pull gpt-oss:120b
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_env
OLLAMA_DIR="$TOOLS_DIR/ollama"
export OLLAMA_MODELS="$MODELS_DIR/ollama"
export OLLAMA_HOST=${OLLAMA_HOST:-0.0.0.0:11434}
export OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH:-131072}
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=${OLLAMA_KV_CACHE_TYPE:-q8_0}
export OLLAMA_KEEP_ALIVE=-1
case "${1:-}" in
    install)
        mkdir -p "$OLLAMA_DIR" "$OLLAMA_MODELS"
        log "downloading ollama (linux arm64) into $OLLAMA_DIR"
        curl -L --progress-bar https://ollama.com/download/ollama-linux-arm64.tgz | tar -xz -C "$OLLAMA_DIR"
        "$OLLAMA_DIR/bin/ollama" --version
        ;;
    serve) exec "$OLLAMA_DIR/bin/ollama" serve ;;
    pull)  "$OLLAMA_DIR/bin/ollama" pull "${2:-gpt-oss:120b}" ;;
    *)     die "usage: ollama.sh install|serve|pull [model]" ;;
esac
