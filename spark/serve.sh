#!/usr/bin/env bash
# Run llama-server in the foreground. systemd, `service.sh bg` and `make serve` all end up here.
#
#   spark/serve.sh            # main instance: MODEL on PORT from config/spark.env
#   spark/serve.sh sidekick   # second instance: SIDEKICK_MODEL on SIDEKICK_PORT
#   spark/serve.sh glm-4.5-air          # any profile on PORT (or PORT=8090 spark/serve.sh glm-4.5-air)
#   FORCE=1 spark/serve.sh    # skip the free-memory check
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_env

role=${1:-main}
case "$role" in
    main)     key=$MODEL ;;
    sidekick) key=${SIDEKICK_MODEL:-}; PORT=${SIDEKICK_PORT:-8081}
              [ -n "$key" ] || die "SIDEKICK_MODEL is not set in config/spark.env" ;;
    *)        key=$role ;;
esac
load_model "$key"
gguf=$("$ROOT/spark/resolve-gguf.sh" "$key")
server=$(llama_bin llama-server)

have=$(mem_available_gb)
if [ "$have" -lt "$MIN_FREE_GB" ] && [ "${FORCE:-0}" != 1 ]; then
    "$ROOT/spark/preflight.sh" "$key" || true
    die "only ${have} GB available, $key wants ${MIN_FREE_GB} GB. Stop whatever is using memory or run with FORCE=1."
fi

# CTX in the profile is per slot. llama-server's -c is the total, divided between -np slots.
# CTX=fit leaves -c out: llama.cpp starts from the model's trained window and shrinks it
# until it fits, keeping FIT_MARGIN_MB free for everything else on the machine.
if [ "$CTX" = fit ]; then
    ctx_args=(--fit on --fit-ctx "${CTX_MIN:-65536}" --fit-target "${FIT_MARGIN_MB:-8192}")
else
    ctx_args=(-c $((CTX * SLOTS)))
fi
args=(
    -m "$gguf" --alias "$ALIAS"
    --host "${HOST:-0.0.0.0}" --port "$PORT"
    -ngl 999
    "${ctx_args[@]}" -np "$SLOTS"
    -fa on
    -b "${BATCH:-2048}" -ub "${UBATCH:-512}"
    --cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE"
    --jinja
    --cache-reuse 256
    --metrics --slots
)
[ -n "${API_KEY:-}" ] && args+=(--api-key "$API_KEY")
# EXTRA_ARGS (profile) and LLAMA_EXTRA (spark.env) are word-split on purpose
# shellcheck disable=SC2206
args+=(${EXTRA_ARGS:-} ${LLAMA_EXTRA:-})

log "$key on :$PORT ($(du -ch "$(dirname "$gguf")"/*.gguf 2>/dev/null | tail -1 | cut -f1) of weights, ${CTX} ctx x ${SLOTS} slot(s), kv $KV_TYPE, ${have} GB free)"
log "llama-server ${args[*]}"
exec "$server" "${args[@]}"
