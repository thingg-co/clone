#!/usr/bin/env bash
# Long-context backend: NVIDIA's vLLM container for DGX Spark, profiles in
# config/vllm/. Weights are kept in a Hugging Face cache under models/hf-cache
# in this tree and mounted into the container, so nothing is downloaded twice.
#
#   spark/vllm.sh pull [profile]        # fetch weights into models/hf-cache
#   spark/vllm.sh serve [profile]       # foreground on PORT (default from spark.env)
#   VLLM_LONG=1 spark/vllm.sh serve     # YaRN-extended context (LONG_MAX_LEN, about 1M)
#   VLLM_LONG=524288 spark/vllm.sh serve  # or a token count between MAX_LEN and LONG_MAX_LEN
#   spark/vllm.sh switch <profile> [long] # write VLLM_PROFILE and VLLM_LONG to spark.env, restart
#   spark/vllm.sh install               # systemd user unit vllm.service for VLLM_PROFILE
#   spark/vllm.sh image                 # show which container tag will be used
#   spark/vllm.sh stop                  # stop the running container
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_env
VLLM_PROFILE=${VLLM_PROFILE:-qwen3.8-27b}
VLLM_PORT=${VLLM_PORT:-$PORT}
HF_CACHE="$MODELS_DIR/hf-cache"
NAME=vllm-main

load_vllm_profile() {
    local key=$1 f="$ROOT/config/vllm/$1.env"
    [ -f "$f" ] || die "no vLLM profile $f (known: $(ls "$ROOT/config/vllm" | sed 's/\.env$//' | tr '\n' ' '))"
    unset TITLE DETAIL HF_REPO EXTRA_REPOS SERVED_NAME MAX_LEN LONG_MAX_LEN ROPE_ORIG ROPE_STYLE TOOL_PARSER REASONING_PARSER REASONING_PLUGIN EFFORT_ALIAS KV_DTYPE MEM_UTIL MIN_FREE_GB DOCKER_ENV EXTRA_ARGS RUNNER PULL PREPARE
    set -a; . "$f"; set +a
    PROFILE_KEY=$key
}

# Newest nvcr.io/nvidia/vllm:YY.MM-py3 that exists, probed with docker, cached in run/.
find_image() {
    if [ -n "${VLLM_IMAGE:-}" ]; then echo "$VLLM_IMAGE"; return; fi
    local cache="$RUN_DIR/vllm-image"
    if [ -f "$cache" ] && [ "$(find "$cache" -mtime -7)" ]; then cat "$cache"; return; fi
    local y m tag i
    y=$(date +%y); m=$(date +%m | sed 's/^0//')
    for i in $(seq 0 18); do
        tag=$(printf 'nvcr.io/nvidia/vllm:%02d.%02d-py3' "$y" "$m")
        if docker manifest inspect "$tag" >/dev/null 2>&1; then echo "$tag" | tee "$cache"; return; fi
        m=$((m - 1)); if [ "$m" -eq 0 ]; then m=12; y=$((y - 1)); fi
    done
    die "no nvcr.io/nvidia/vllm tag found in the last 18 months; set VLLM_IMAGE explicitly (check ngc.nvidia.com)"
}

cmd=${1:-serve}
case "$cmd" in
    image) find_image ;;
    pull)
        load_vllm_profile "${2:-$VLLM_PROFILE}"
        if [ -n "${PULL:-}" ]; then   # a model whose runner keeps its weights its own way
            # shellcheck disable=SC2086
            exec "$ROOT"/$PULL
        fi
        command -v hf >/dev/null || die "hf CLI missing; run spark/bootstrap.sh"
        # Under set -e a failed curl would end the script without a word, so say what happened.
        code=$(curl -s --retry 3 --retry-delay 10 --retry-all-errors -o /dev/null -w '%{http_code}' "https://huggingface.co/api/models/$HF_REPO") || true
        [ "$code" != 000 ] || die "could not reach huggingface.co to check $HF_REPO"
        [ "$code" = 200 ] || die "HF repo $HF_REPO returned HTTP $code; fix HF_REPO in config/vllm/$PROFILE_KEY.env"
        mkdir -p "$HF_CACHE"
        log "pulling $HF_REPO into $HF_CACHE (cache layout, used by the container)"
        HF_HOME="$HF_CACHE" hf download "$HF_REPO"
        for extra in ${EXTRA_REPOS:-}; do   # draft models for speculative decoding and the like
            log "pulling $extra"
            HF_HOME="$HF_CACHE" hf download "$extra"
        done
        log "pulling container image $(find_image)"
        docker pull "$(find_image)"
        ;;
    serve)
        load_vllm_profile "${2:-$VLLM_PROFILE}"
        if [ -n "${RUNNER:-}" ]; then   # a model the stock image cannot serve brings its own launcher
            [ -x "$ROOT/$RUNNER" ] || die "$PROFILE_KEY is served by $RUNNER, which is missing or not executable"
            exec "$ROOT/$RUNNER" serve "$PROFILE_KEY"
        fi
        image=$(find_image)
        if [ "${FORCE:-0}" != 1 ] && ! have=$(settled_mem_gb "$MIN_FREE_GB"); then
            die "$PROFILE_KEY wants ${MIN_FREE_GB} GB available and steady for 30 s; ${have} GB is available (FORCE=1 to override)"
        fi
        max_len=$MAX_LEN
        overrides=()
        long=${VLLM_LONG:-0}
        if [ "$long" != 0 ] && [ -z "${LONG_MAX_LEN:-}" ]; then
            log "$PROFILE_KEY serves its full context natively; ignoring VLLM_LONG=$long"
            long=0
        fi
        if [ "$long" != 0 ]; then
            if [ "$long" = 1 ]; then max_len=$LONG_MAX_LEN; else max_len=$long; fi
            [ "$max_len" -le "$LONG_MAX_LEN" ] || die "VLLM_LONG=$long is past LONG_MAX_LEN=$LONG_MAX_LEN for $PROFILE_KEY"
        fi
        # A window vLLM refused on an earlier start, with its own estimate of what does fit
        # (see the end of this branch). Delete the file to try the profile's full window again.
        fit="$RUN_DIR/vllm-ctx-$PROFILE_KEY"
        if [ -s "$fit" ] && [ "$(cat "$fit")" -lt "$max_len" ]; then
            max_len=$(cat "$fit")
            log "$PROFILE_KEY: using the $max_len-token window that fit last time ($fit)"
        fi
        if [ "$max_len" -gt "$MAX_LEN" ] && [ -n "${ROPE_ORIG:-}" ]; then
            # YaRN factor as the model cards give it: 2.0 for 524k, 4.0 for 1M
            factor=$(python3 -c "import math; print(float(math.ceil($max_len/$ROPE_ORIG)))")
            rope="{\"rope_type\":\"yarn\",\"factor\":$factor,\"original_max_position_embeddings\":$ROPE_ORIG}"
            # older vLLM builds take --rope-scaling instead (VLLM_ROPE_FLAG=rope-scaling in spark.env)
            if [ "${ROPE_STYLE:-}" = qwen35 ]; then   # Qwen3.5 and later keep it under text_config.rope_parameters
                overrides=(--hf-overrides "{\"text_config\":{\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"rope_type\":\"yarn\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"factor\":$factor,\"original_max_position_embeddings\":$ROPE_ORIG}}}")
            elif [ "${VLLM_ROPE_FLAG:-hf-overrides}" = rope-scaling ]; then overrides=(--rope-scaling "$rope"); else overrides=(--hf-overrides "{\"rope_scaling\":$rope}"); fi
        fi
        args=(vllm serve "$HF_REPO"
            --served-model-name "$SERVED_NAME"
            --host 0.0.0.0 --port "$VLLM_PORT"
            --max-model-len "$max_len"
            --gpu-memory-utilization "$MEM_UTIL"
            --kv-cache-dtype "$KV_DTYPE"
            --max-num-seqs "${VLLM_SEQS:-4}"
            --enable-prefix-caching
            --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER"
        )
        if [ -n "${REASONING_PLUGIN:-}" ]; then   # a parser shipped inside the model repo
            plugin=$(find "$HF_CACHE/hub/models--${HF_REPO//\//--}/snapshots" -name "$REASONING_PLUGIN" 2>/dev/null | head -1)
            [ -n "$plugin" ] || die "$REASONING_PLUGIN not found in the $HF_REPO snapshot; pull the profile first"
            args+=(--reasoning-parser-plugin "/root/.cache/huggingface${plugin#"$HF_CACHE"}")
        fi
        [ -n "${REASONING_PARSER:-}" ] && args+=(--reasoning-parser "$REASONING_PARSER")
        envs=()
        for kv in ${DOCKER_ENV:-}; do envs+=(-e "$kv"); done
        mounts=()
        if [ "${EFFORT_ALIAS:-0}" = 1 ]; then
            # Qwen3.8 templates take reasoning_effort xhigh, medium or low and reject the rest with
            # a 400, including the "high" Claude Code sends. Serve a copy whose one effort line maps
            # high and max to xhigh and minimal to low first (the rewrite the qwen3.8-Flash-DGX
            # recipe uses); every other value renders exactly as before.
            tpl=$(find "$HF_CACHE/hub/models--${HF_REPO//\//--}/snapshots" -name chat_template.jinja 2>/dev/null | head -1)
            alias_tpl="$RUN_DIR/chat-template-$PROFILE_KEY.jinja"
            if [ -n "$tpl" ] && python3 -I - "$tpl" "$alias_tpl" <<'PY'
import sys
old = "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
new = "{%- set resolved_reasoning_effort = {'high': 'xhigh', 'max': 'xhigh', 'minimal': 'low'}.get(reasoning_effort|default('xhigh'), reasoning_effort|default('xhigh')) %}"
text = open(sys.argv[1], encoding="utf-8").read()
if text.count(old) != 1:
    sys.exit(1)
open(sys.argv[2], "w", encoding="utf-8").write(text.replace(old, new))
PY
            then
                mounts+=(-v "$alias_tpl:/clone/chat_template.jinja:ro")
                args+=(--chat-template /clone/chat_template.jinja)
            else
                warn "EFFORT_ALIAS: the template's effort line is not the expected one; serving it unchanged (reasoning_effort high will be refused)"
            fi
        fi
        [ -n "${API_KEY:-}" ] && args+=(--api-key "$API_KEY")
        # shellcheck disable=SC2206
        args+=("${overrides[@]}" ${EXTRA_ARGS:-} ${VLLM_EXTRA:-})
        mkdir -p "$HF_CACHE"
        log "$PROFILE_KEY on :$VLLM_PORT, max_len $max_len, image $image"
        log "${args[*]}"
        docker rm -f "$NAME" >/dev/null 2>&1 || true
        set +e
        docker run --rm --name "$NAME" --gpus all --ipc=host --network host \
            -v "$HF_CACHE:/root/.cache/huggingface" "${mounts[@]}" \
            -e HF_HUB_OFFLINE=1 -e HF_TOKEN="${HF_TOKEN:-}" -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 "${envs[@]}" \
            "$image" "${args[@]}" 2>&1 | tee -p >(head -n 5000 > "$LOG_DIR/vllm-start.log")   # the start-up part only
        code=${PIPESTATUS[0]}
        set -e
        sleep 1
        # vLLM refuses a window its cache cannot hold and says what would fit. Take 97% of
        # that and start again, instead of failing the same way on every restart.
        est=$(sed -n 's/.*estimated maximum model length is \([0-9]*\).*/\1/p' "$LOG_DIR/vllm-start.log" | tail -1)
        if [ -n "$est" ] && [ "$est" -lt "$max_len" ]; then
            echo $(( est * 97 / 100 / 1000 * 1000 )) > "$fit"
            log "a $max_len-token window does not fit; vLLM estimates $est. Starting again with $(cat "$fit")"
            exec "$0" serve "$PROFILE_KEY"
        fi
        exit "$code"
        ;;
    switch)
        key=${2:?usage: vllm.sh switch <profile> [0|1|tokens]}
        long=${3:-${VLLM_LONG:-0}}
        load_vllm_profile "$key"   # validates the profile
        f="$ROOT/config/spark.env"
        for kv in "BACKEND=vllm" "VLLM_PROFILE=$key" "VLLM_LONG=$long"; do
            if grep -q "^${kv%%=*}=" "$f"; then sed -i "s/^${kv%%=*}=.*/$kv/" "$f"; else echo "$kv" >> "$f"; fi
        done
        "$0" install
        log "restarting vllm with $key, VLLM_LONG=$long (loading the weights takes several minutes)"
        systemctl --user restart vllm.service
        wait_healthy "$VLLM_PORT" 1800 || die "not healthy after 30 min; check: spark/vllm.sh logs"
        log "healthy on :$VLLM_PORT"
        check_margin vllm.service || exit 1
        ;;
    stop) docker stop "$NAME" ;;
    logs) docker logs -f --tail 200 "$NAME" ;;
    install)
        mkdir -p "$HOME/.config/systemd/user"
        cat > "$HOME/.config/systemd/user/vllm.service" <<UNIT
[Unit]
Description=vLLM main (clone)
After=network-online.target docker.service

[Service]
Type=simple
WorkingDirectory=$ROOT
Environment=VLLM_PROFILE=$VLLM_PROFILE
Environment=VLLM_LONG=${VLLM_LONG:-0}
ExecStart=/usr/bin/env bash $ROOT/spark/vllm.sh serve
# A runner without a container (ds4) has nothing for docker to stop; systemd then ends it
ExecStop=-/usr/bin/docker stop $NAME
# Never enabled and never restarted, like llama-server.service: a server that died on the
# Spark usually died of memory, and starting it again stalled the machine.
Restart=no
TimeoutStopSec=90

[Install]
WantedBy=default.target
UNIT
        systemctl --user daemon-reload
        systemctl --user disable vllm.service >/dev/null 2>&1 || true
        log "installed vllm.service for profile $VLLM_PROFILE; start with: systemctl --user start vllm"
        ;;
    *) die "usage: vllm.sh pull|serve|switch|stop|logs|install|image [profile]" ;;
esac
