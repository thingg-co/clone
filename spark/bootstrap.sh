#!/usr/bin/env bash
# One-shot setup on the Spark. Idempotent, no sudo, everything under this tree:
# tools in .tools/, llama.cpp in llama.cpp/, weights in models/, logs in logs/.
#
#   spark/bootstrap.sh                # tools, build, pull the configured models, install and start services
#   spark/bootstrap.sh --no-pull      # skip downloads (resume them later with pull-model.sh / vllm.sh pull)
#   spark/bootstrap.sh --no-build     # skip the llama.cpp build
#   spark/bootstrap.sh --no-start     # install services but do not start them
#   spark/bootstrap.sh --agent        # also create the agent's venv here (to run fable on the Spark itself)
set -euo pipefail
. "$(dirname "$0")/common.sh"

do_pull=1 do_build=1 do_start=1 do_agent=0
for a in "$@"; do
    case "$a" in
        --no-pull) do_pull=0 ;;
        --no-build) do_build=0 ;;
        --no-start) do_start=0 ;;
        --agent) do_agent=1 ;;
        *) die "unknown flag $a" ;;
    esac
done

# 1. config/spark.env
if [ ! -f "$ROOT/config/spark.env" ]; then
    cp "$ROOT/config/spark.env.example" "$ROOT/config/spark.env"
    key=$(python3 -c 'import secrets; print(secrets.token_hex(16))')
    sed -i "s/^API_KEY=.*/API_KEY=$key/; s/^SPARK_HOST=.*/SPARK_HOST=$(hostname).local/; s/^SPARK_USER=.*/SPARK_USER=$USER/" "$ROOT/config/spark.env"
    log "wrote config/spark.env with a fresh API key"
fi
load_env
BACKEND=${BACKEND:-vllm}
log "backend=$BACKEND model=$MODEL vllm_profile=${VLLM_PROFILE:-} sidekick=${SIDEKICK_MODEL:-none} port=$PORT"

# 2. uv, hf and ninja under .tools (no sudo, nothing outside this tree)
mkdir -p "$TOOLS_DIR/bin"
if [ ! -x "$TOOLS_DIR/bin/uv" ]; then
    log "installing uv into $TOOLS_DIR/bin"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$TOOLS_DIR/bin" INSTALLER_NO_MODIFY_PATH=1 sh
fi
uv tool install --quiet --upgrade "huggingface_hub[cli,hf_xet]"
uv tool install --quiet --upgrade ninja
log "uv $(uv --version | cut -d' ' -f2), hf $(hf version 2>/dev/null | tail -1 || echo '?')"

# 3. llama.cpp (needed for GGUF profiles and the sidekick; ~15 min first time)
if [ "$do_build" = 1 ]; then
    "$ROOT/spark/build-llama.sh"
fi

# 4. weights
if [ "$do_pull" = 1 ]; then
    if [ "$BACKEND" = vllm ]; then
        "$ROOT/spark/vllm.sh" pull "${VLLM_PROFILE:-qwen3.8-27b}"
    else
        "$ROOT/spark/pull-model.sh" "$MODEL"
    fi
    if [ -n "${SIDEKICK_MODEL:-}" ]; then "$ROOT/spark/pull-model.sh" "$SIDEKICK_MODEL"; fi
fi

# 5. services
"$ROOT/spark/service.sh" install
if [ "$BACKEND" = vllm ]; then VLLM_PROFILE="${VLLM_PROFILE:-qwen3.8-27b}" "$ROOT/spark/vllm.sh" install; fi

systemctl --user restart clone-panel.service && log "control page on http://$(hostname).local:${PANEL_PORT:-8088} (make panel on the Mac)"

# 6. agent venv on the Spark (optional)
if [ "$do_agent" = 1 ]; then
    (cd "$ROOT/agent" && uv sync --quiet) && log "agent ready: $ROOT/agent/.venv/bin/fable"
fi

# 7. start and check
if [ "$do_start" = 1 ]; then
    if [ "$BACKEND" = vllm ]; then systemctl --user start vllm.service; else systemctl --user start llama-server.service; fi
    if [ -n "${SIDEKICK_MODEL:-}" ]; then systemctl --user start llama-sidekick.service; fi
    log "waiting for :$PORT (model load can take several minutes)"
    if wait_healthy "$PORT" 1800; then
        log "healthy; running the tool-calling smoke test"
        python3 -I "$ROOT/scripts/smoke-test.py" --base-url "http://127.0.0.1:$PORT/v1" --api-key "$API_KEY" || warn "smoke test failed; see logs"
    else
        warn "not healthy after 30 min. Logs: spark/service.sh logs   or   spark/vllm.sh logs"
    fi
fi
log "done. Agent from the Mac: make tunnel, then cd agent && uv run fable"
