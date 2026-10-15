#!/usr/bin/env bash
# DeepSeek V4 Flash 0731 on ds4 (DwarfStar 4), antirez's C and CUDA engine written for this
# one model, in Entrpi's fork tuned for the GB10 (https://github.com/Entrpi/ds4, MIT) at the
# tag Entrpi's ds4-on-spark installer pins. It runs only antirez's own GGUF builds
# (https://huggingface.co/antirez/deepseek-v4-gguf, MIT). This one has the routed experts at
# 2 bits (IQ2_XXS and Q2_K) and attention, shared experts and output at Q8_0, 81 GiB. The
# DSpark draft (6.5 GiB) goes beside it, where the server finds it and attaches it. The cache
# costs about 4.3 KiB a token, so a full 1M window takes about 4 GiB.
#
#   spark/ds4.sh fetch     # the weights, resumable (vllm.sh pull calls this)
#   spark/ds4.sh build     # clone under third_party/ at the pinned tag and build
#   spark/ds4.sh serve     # foreground on PORT; vllm.sh serve calls this
#   spark/ds4.sh stop | logs
#
# The server takes no API key and trusts whatever reaches it, which on this LAN is fine. It
# has no /health either: it opens its port once the weights are loaded, and wait_healthy
# takes /v1/models instead.
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_env
DS4_REPO=https://github.com/Entrpi/ds4.git
DS4_REF=v0.6.5
DIR="$ROOT/third_party/ds4"
DATA="$MODELS_DIR/deepseek-v4-flash-ds4"
BASE_REPO=antirez/deepseek-v4-gguf
BASE_REV=f71f23d552d664e523b422157b2befbf74040380
BASE=DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
DRAFT_REPO=bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF
DRAFT_REV=57fae967d0c5dac1f32a444ea8884ec46459247b
DRAFT=DSpark-drafter-Q2K-Q8-0731.gguf
CTX=${DS4_CTX:-1048576}

fetch_one() {  # repo revision file
    local dest="$DATA/$3" size sha try
    read -r size sha < <(curl -s --retry 3 --retry-delay 10 --retry-all-errors "https://huggingface.co/api/models/$1/tree/$2" |
        python3 -I -c '
import json, sys
for f in json.load(sys.stdin):
    if f.get("path") == sys.argv[1]:
        print(f["size"], (f.get("lfs") or {}).get("oid", ""))' "$3") || true
    [ -n "${size:-}" ] || die "could not find $3 in $1 at $2"
    if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ]; then
        log "$3 is here"
        return
    fi
    log "fetching $3 ($((size / 1073741824)) GiB) from $1"
    # curl picks up a partial file where it stopped; the hf CLI would start over
    for try in 1 2 3 4 5 6; do
        curl -L --fail --silent --show-error -C - -o "$dest.part" "https://huggingface.co/$1/resolve/$2/$3" && break
        [ "$try" = 6 ] && die "download of $3 failed six times; run $0 fetch again to resume"
        warn "download of $3 stopped; resuming in a minute"
        sleep 60
    done
    [ "$(stat -c %s "$dest.part")" = "$size" ] || die "$3 came out at $(stat -c %s "$dest.part") bytes, not $size; delete $dest.part and fetch again"
    if [ -n "$sha" ]; then
        log "checking $3"
        [ "$(sha256sum "$dest.part" | cut -d' ' -f1)" = "$sha" ] || die "$3 does not match its sha256; delete $dest.part and fetch again"
    fi
    mv "$dest.part" "$dest"
}

cmd=${1:-serve}
case "$cmd" in
    fetch)
        mkdir -p "$DATA"
        # The bake fetches before it serves, maybe while an earlier fetch is still running: wait
        # for that one instead of writing the same file twice.
        exec 9> "$RUN_DIR/ds4-fetch.lock"
        flock -n 9 || { log "another fetch is running; waiting for it"; flock 9; }
        fetch_one "$BASE_REPO" "$BASE_REV" "$BASE"
        fetch_one "$DRAFT_REPO" "$DRAFT_REV" "$DRAFT"
        log "weights ready in $DATA"
        ;;
    build)
        if [ ! -d "$DIR/.git" ]; then
            git clone --quiet --depth 1 --branch "$DS4_REF" "$DS4_REPO" "$DIR"
        elif [ "$(git -C "$DIR" describe --tags --exact-match 2>/dev/null)" != "$DS4_REF" ]; then
            git -C "$DIR" fetch --quiet --depth 1 origin tag "$DS4_REF"
            git -C "$DIR" checkout --quiet --detach "$DS4_REF"
        fi
        if [ -x "$DIR/ds4-server" ] && [ "$(cat "$RUN_DIR/ds4-built" 2>/dev/null)" = "$DS4_REF" ]; then
            log "ds4 $DS4_REF is built"
            exit 0
        fi
        log "building ds4 $DS4_REF for the GB10"
        make -C "$DIR" -j "${BUILD_JOBS:-$(nproc)}" cuda-spark
        "$DIR/ds4-server" --version
        echo "$DS4_REF" > "$RUN_DIR/ds4-built"
        ;;
    serve)
        [ -x "$DIR/ds4-server" ] || die "ds4 is not built; run $0 build"
        [ -f "$DATA/$BASE" ] && [ -f "$DATA/$DRAFT" ] || die "the weights are not all in $DATA; run $0 fetch"
        want=${MIN_FREE_GB:-100}
        if [ "${FORCE:-0}" != 1 ] && ! have=$(settled_mem_gb "$want"); then
            die "ds4 wants $want GB available and steady for 30 s; $have GB is available (FORCE=1 to override)"
        fi
        log "DeepSeek V4 Flash on ds4 $DS4_REF, :$PORT, window $CTX"
        # Thinking is on by default. Effort max with --reasoning-effort-native adds DeepSeek's own
        # instruction to reason at length, as reasoning_effort max does in the llama.cpp profiles;
        # without that flag ds4 maps every effort to the level that adds nothing. The memory floor
        # is the margin every model here keeps.
        exec "$DIR/ds4-server" --cuda -m "$DATA/$BASE" -c "$CTX" --host 0.0.0.0 --port "$PORT" \
            --reasoning-effort max --reasoning-effort-native --mem-floor-gb "$SAFETY_GB" --no-update-check
        ;;
    stop) systemctl --user stop vllm.service ;;
    logs) journalctl --user -u vllm.service -f -o cat ;;
    *) die "usage: ds4.sh fetch|build|serve|stop|logs" ;;
esac
