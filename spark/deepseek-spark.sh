#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 in 0xSero's build for one DGX Spark
# (https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer, Apache-2.0; weights
# 0xSero/deepseek-v4-flash-0731-spark, MIT). The experts are pruned (REAP) and the weights
# quantized to 3 bits (EXL3), 107 GB; a patched vLLM with the sparkinfer kernels serves them
# with DeepSeek's DSpark speculative decoding. The build is validated at 262,144 tokens.
#
# The image's own start-up downloads the weights, repacks them for one GPU and builds the
# draft model, which takes well over an hour the first time. Here those are separate steps,
# so a start only loads:
#
#   spark/deepseek-spark.sh fetch      # the image and the weights (vllm.sh pull calls this)
#   spark/deepseek-spark.sh prepare    # repack for one GPU and build the draft; needs the memory free
#   spark/deepseek-spark.sh serve      # foreground on PORT; vllm.sh serve calls this
#   spark/deepseek-spark.sh refit | stop | logs
#
# The image and the weights are pinned by digest and revision. Its own memory setting is
# 0.9465 of the GPU, more than this machine has free with a desktop session running, so the
# launcher gives it what is free less 4 GiB. When vLLM then says the 262k window does not
# fit, the launcher keeps 97% of vLLM's estimate in run/deepseek-spark-ctx and starts again.
# The container runs as root, so the repacked files under models/ belong to root.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_env
IMAGE=ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4
MODEL_REPO=0xSero/deepseek-v4-flash-0731-spark
MODEL_REV=22f28d32b9b29b4352eaa380ff8c2c170b2847ab
DATA="$MODELS_DIR/deepseek-v4-flash-spark"     # source/ as downloaded, tp1/ repacked, dspark-draft-k64/
NAME=vllm-main                                 # vllm.service stops the container by this name
FIT="$RUN_DIR/deepseek-spark-ctx"
PY=/opt/runtime-venv/bin/python
SCRIPTS=/opt/recipe/scripts

cmd=${1:-serve}
case "$cmd" in
    fetch)
        command -v hf >/dev/null || die "hf CLI missing; run spark/bootstrap.sh"
        log "pulling the runtime image (pinned by digest)"
        docker pull "$IMAGE"
        mkdir -p "$DATA/source"
        log "fetching $MODEL_REPO at $MODEL_REV into $DATA/source"
        hf download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$DATA/source"
        ;;
    prepare)
        [ -d "$DATA/source" ] || die "no weights in $DATA/source; run $0 fetch first"
        if [ -f "$DATA/tp1/rank-sliced-tp1-manifest.json" ] && [ -f "$DATA/dspark-draft-k64/model.safetensors.index.json" ]; then
            log "already repacked, and the draft is built"
            exit 0
        fi
        log "repacking the weights for one GPU and building the draft model (CPU and disk; this takes a while)"
        docker run --rm -v "$DATA:/models" --entrypoint bash "$IMAGE" -c "
            set -euo pipefail
            if [ ! -f /models/tp1/rank-sliced-tp1-manifest.json ]; then
                $PY $SCRIPTS/coalesce_rank_sliced_exl3.py --input-dir /models/source --output-dir /models/tp1 \
                    --link-carried --reuse-complete --workers 2
            fi
            $PY $SCRIPTS/verify_tp1_manifest.py /models/tp1
            if [ ! -f /models/dspark-draft-k64/model.safetensors.index.json ]; then
                $PY $SCRIPTS/build_dspark_draft.py --source /models/tp1 --output /models/dspark-draft-k64 \
                    --experts 64 --structured-per-category 32
            fi"
        log "ready to serve"
        ;;
    refit) rm -f "$FIT"; log "the next start will try the full 262k window again" ;;
    serve)
        [ -f "$DATA/tp1/rank-sliced-tp1-manifest.json" ] || die "the weights are not repacked yet; run $0 prepare"
        max_len=262144
        if [ -s "$FIT" ] && [ "$(cat "$FIT")" -lt "$max_len" ]; then max_len=$(cat "$FIT"); fi
        # The share is worked out from what is free, so wait for the last server's memory to
        # come back. It leaves SAFETY_GB plus 1 for the container's own memory.
        avail=$(settled_mem_gb 0) || true
        util=$(python3 -c "print(round(min(0.9465, ($avail - $SAFETY_GB - 1) / 121.7), 4))")
        log "DeepSeek V4 Flash, single-Spark build, on :$PORT: window $max_len, memory share $util"
        docker rm -f "$NAME" >/dev/null 2>&1 || true
        # The image's own defaults are thinking off and effort high, which adds no instruction.
        # EXTRA_VLLM_ARGS lands after its flags and wins, so a request that names no effort
        # gets thinking at max; one that names an effort still gets that.
        set +e
        docker run --rm --name "$NAME" --gpus all --network host --ipc host --shm-size 16g \
            -e HOST=0.0.0.0 -e PORT="$PORT" -e SERVED_MODEL_NAME=deepseek-v4-flash-spark \
            -e MODEL_REPO="$MODEL_REPO" -e MODEL_REVISION="$MODEL_REV" \
            -e MAX_MODEL_LEN="$max_len" -e MAX_NUM_SEQS=4 -e MAX_NUM_BATCHED_TOKENS=8224 \
            -e MODE=dspark -e DSPARK_TOKENS=5 -e DSPARK_CAPACITY=0 -e DSPARK_DYNAMIC_DRAFT_DEPTH=0 \
            -e DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW=8 -e DSPARK_DRAFT_EXPERTS=64 -e DSPARK_STRUCTURED_EXPERTS_PER_CATEGORY=32 \
            -e VLLM_USE_B12X_WO_PROJECTION=1 -e GPU_MEMORY_UTILIZATION="$util" -e VERIFY_MODEL_CHECKSUMS=0 \
            -e EXTRA_VLLM_ARGS="--default-chat-template-kwargs.thinking=true --default-chat-template-kwargs.reasoning_effort=max" \
            -v "$DATA:/models" -v "$DATA/cache:/cache" \
            "$IMAGE" 2>&1 | tee -p >(head -n 5000 > "$LOG_DIR/deepseek-spark-start.log")
        code=${PIPESTATUS[0]}
        set -e
        sleep 1
        est=$(sed -n 's/.*estimated maximum model length is \([0-9]*\).*/\1/p' "$LOG_DIR/deepseek-spark-start.log" | tail -1)
        if [ -n "$est" ] && [ "$est" -lt "$max_len" ]; then
            echo $(( est * 97 / 100 / 1000 * 1000 )) > "$FIT"
            log "a $max_len-token window does not fit; vLLM estimates $est. Starting again with $(cat "$FIT")"
            exec "$0" serve
        fi
        exit "$code"
        ;;
    stop) docker stop "$NAME" ;;
    logs) docker logs -f --tail 200 "$NAME" ;;
    *) die "usage: deepseek-spark.sh fetch|prepare|serve|refit|stop|logs" ;;
esac
