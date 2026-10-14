#!/usr/bin/env bash
# Qwen3.8-Flash-Next through the qwen3.8-Flash-DGX recipe
# (https://github.com/blazux/qwen3.8-Flash-DGX, Apache-2.0).
#
# The NVFP4 checkpoint is 133 GB, more than this machine's memory, so the
# stock vLLM image cannot load it. The recipe patches the official vLLM 0.31
# image to read the 48 GiB n-gram table from the NVMe through mmap, which
# leaves about 75 GiB resident, and fixes prefix caching and nondeterministic
# top-k on the GB10 along the way. It is cloned under third_party/ at a pinned
# commit, so this never follows a moving branch; review the diff before
# raising FLASH_REF.
#
#   spark/vllm.sh pull qwen3.8-flash-next   # the weights, into models/hf-cache
#   spark/flash-next.sh setup               # clone, build the image, prepare the hybrid layout
#   spark/flash-next.sh serve               # setup if needed, then foreground; vllm.sh serve calls this
#   spark/flash-next.sh build               # just the image, e.g. while the weights download
#   spark/flash-next.sh refit               # forget the fitted window and try the full one again
#   spark/flash-next.sh status | logs | stop
#
# VLLM_LONG=0 asks for 500k tokens, VLLM_LONG=1 for 1M, both with the FP8 cache
# and the recipe's hybrid layout. What this machine can give is less than 1M:
# after 75 GiB of weights, 0.80 of its 121.7 GiB leaves about 15 GiB of cache,
# and vLLM refuses a window that does not fit ("the estimated maximum model
# length is 576000" on the first try here). When that happens the launcher
# writes 97% of vLLM's estimate to run/flash-next-ctx and starts again with it,
# and later starts go straight to that window. Delete the file to have it try
# the full window again, for instance after freeing memory. At 16 bits the
# cache holds half as much, which is under 300k here, so FP8 is not optional.
#
# The recipe's containers run as root, so the hybrid layout it writes under
# models/hf-cache is owned by root; remove it with `spark/flash-next.sh clean`,
# not rm.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_env
FLASH_REPO=https://github.com/blazux/qwen3.8-Flash-DGX
FLASH_REF=${FLASH_REF:-7d0798e8c3aaacee8c72afb302aa4247fe4277a7}
DIR="$ROOT/third_party/qwen3.8-Flash-DGX"
HF_CACHE="$MODELS_DIR/hf-cache"
NAME=vllm-main
MODEL_REPO=nvidia/Qwen3.8-Flash-Next-NVFP4

FIT="$RUN_DIR/flash-next-ctx"
profile() { echo default; }   # hybrid layout, YaRN; the window and cache type are set below
window() {   # what was asked for, capped by what vLLM last said fits
    local want=500000 fitted
    [ "${VLLM_LONG:-0}" = 0 ] || want=1000000
    if [ -s "$FIT" ]; then read -r fitted < "$FIT"; [ "$fitted" -lt "$want" ] && want=$fitted; fi
    echo "$want"
}
flash() { (cd "$DIR" && env HF_CACHE="$HF_CACHE" NAME="$NAME" MODEL="$MODEL_REPO" "$@"); }
checkout() {
    if [ ! -d "$DIR/.git" ]; then
        mkdir -p "$(dirname "$DIR")"
        git clone --quiet "$FLASH_REPO" "$DIR"
    fi
    if [ "$(git -C "$DIR" rev-parse HEAD)" != "$FLASH_REF" ]; then
        git -C "$DIR" fetch --quiet origin
        git -C "$DIR" checkout --quiet --detach "$FLASH_REF"
    fi
}

cmd=${1:-serve}
case "$cmd" in
    build)   # the image only; safe to run while the weights are still downloading
        checkout
        log "recipe at $(git -C "$DIR" rev-parse --short HEAD); building qwen38-flash-dgx:v0.31"
        docker build -t qwen38-flash-dgx:v0.31 "$DIR"
        ;;
    setup)
        checkout
        log "recipe at $(git -C "$DIR" rev-parse --short HEAD); building the image and preparing the checkpoint"
        flash ./flash setup "$(profile)"
        ;;
    refit) rm -f "$FIT"; log "the next start will try the full window again" ;;
    serve)
        # Every setup step is skipped when already done, so this costs seconds after the first
        # time. The first time it downloads the base image and takes a good twenty minutes.
        checkout
        flash ./flash setup "$(profile)"
        have=$(mem_available_gb)
        if [ "$have" -lt 100 ] && [ "${FORCE:-0}" != 1 ]; then
            die "only ${have} GB available, qwen3.8-flash-next wants 100 GB (FORCE=1 to override)"
        fi
        ctx=$(window)
        # The recipe always stretches positions by 4. Qwen's card says to use 2 when the window
        # is 524,288 or less, and it matters: at factor 4 the model returned the neighbouring
        # record instead of the one asked for once the prompt passed 500k. A later
        # --hf-overrides replaces the recipe's own.
        factor=${FLASH_YARN_FACTOR:-}
        if [ -z "$factor" ]; then if [ "$ctx" -le 524288 ]; then factor=2.0; else factor=4.0; fi; fi
        extra="${API_KEY:+--api-key $API_KEY}"
        if [ "$factor" != 4.0 ]; then
            extra="$extra --hf-overrides {\"text_config\":{\"rope_parameters\":{\"mrope_interleaved\":true,\"mrope_section\":[11,11,10],\"rope_type\":\"yarn\",\"rope_theta\":10000000,\"partial_rotary_factor\":0.25,\"factor\":$factor,\"original_max_position_embeddings\":262144}}}"
        fi
        log "qwen3.8-flash-next on :$PORT, window $ctx, YaRN factor $factor, FP8 cache"
        docker rm -f "$NAME" >/dev/null 2>&1 || true
        flash env PORT="$PORT" SERVED_MODEL_NAME=qwen3.8-flash-next SEQS="${VLLM_SEQS:-4}" \
            CTX="$ctx" YARN=1 KV_DTYPE=fp8_e4m3 \
            EXTRA="$extra" ./flash serve "$(profile)"
        docker update --restart=no "$NAME" >/dev/null   # the systemd unit decides about restarts
        docker logs -f "$NAME" || true
        code=$(docker inspect -f '{{.State.ExitCode}}' "$NAME" 2>/dev/null || echo 1)
        est=$(docker logs "$NAME" 2>&1 | sed -n 's/.*estimated maximum model length is \([0-9]*\).*/\1/p' | tail -1)
        if [ -n "$est" ] && [ "$est" -lt "$ctx" ]; then
            echo $(( est * 97 / 100 / 1000 * 1000 )) > "$FIT"
            log "a $ctx-token window does not fit; vLLM estimates $est. Starting again with $(cat "$FIT")"
            exec "$0" serve
        fi
        exit "$code"
        ;;
    status) flash env PORT="$PORT" ./flash status ;;
    logs) docker logs -f --tail 200 "$NAME" ;;
    stop) docker stop "$NAME" ;;
    clean)   # the root-owned hybrid layout, through a container since rm as this user cannot
        docker run --rm -v "$HF_CACHE:/hf" --entrypoint bash qwen38-flash-dgx:v0.31 -c \
            "rm -rf /hf/hub/models--${MODEL_REPO//\//--}/snapshots/*-fp8hybrid* /hf/qwen38-flash-dgx"
        ;;
    *) die "usage: flash-next.sh build|setup|serve|refit|status|logs|stop|clean" ;;
esac
