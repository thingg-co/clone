#!/usr/bin/env bash
# Measure models one after another and file the results for the control page.
# For each item: download it if needed, serve it, wait until it answers, then run
# scripts/bench.py. Items are backend:profile[:long], with long as for vllm.sh switch.
#
#   spark/bake.sh llama:deepseek-v4-flash vllm:qwen3.6-35b-a3b:1 vllm:qwen3.5-122b
#   FINAL=vllm:qwen3.8-27b-fp8:1 spark/bake.sh ...   # serve this one when the queue is done
#   AFTER=<pid> spark/bake.sh ...                    # let a running test finish first
#   echo vllm:qwen3.8-27b-fp8:1 > run/bake.final     # or decide the final model later
#
# Progress goes to logs/bake.log, each model's run to logs/bench-<profile>.log and the
# results to run/results.jsonl. run/bake.state names the model being measured, for the
# control page. A model that fails to download or start is logged and skipped.
set -uo pipefail
. "$(dirname "$0")/common.sh"
load_env

[ "$#" -gt 0 ] || die "usage: spark/bake.sh backend:profile[:long] ..."
STATE="$RUN_DIR/bake.state"
exec > >(tee -a "$LOG_DIR/bake.log") 2>&1
# Another queue may be waiting its turn; only clear the state file if this one wrote it.
trap 'grep -q " $$\$" "$STATE" 2>/dev/null && rm -f "$STATE"' EXIT

profile_file() { if [ "$1" = vllm ]; then echo "$ROOT/config/vllm/$2.env"; else echo "$ROOT/config/models/$2.env"; fi; }
field() {  # backend profile NAME: one field of a profile, empty when unset
    (set -a; . "$(profile_file "$1" "$2")" >/dev/null 2>&1; eval "printf '%s' \"\${$3:-}\"")
}
served() {  # the model ids the main port lists
    curl -s -m 5 -H "Authorization: Bearer ${API_KEY:-}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
        python3 -I -c 'import sys, json; print(" ".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null
}
note_for() {  # backend profile long: how it was served, for the results table
    local kv orig
    if [ "$1" = vllm ]; then
        kv=$(field vllm "$2" KV_DTYPE); orig=$(field vllm "$2" ROPE_ORIG)
        case "$kv" in fp8*) kv="FP8 cache" ;; nvfp4*) kv="FP4 cache" ;; *) kv="16-bit cache" ;; esac
        if [ "$3" != 0 ] && [ -n "$orig" ] && [ -n "$(field vllm "$2" LONG_MAX_LEN)" ]; then
            echo "vLLM, stretched from $((orig / 1024))k, $kv"
        else
            echo "vLLM, $kv"
        fi
    else
        kv=$(field llama "$2" KV_TYPE)
        case "$kv" in f16|bf16) kv="16-bit cache" ;; q8_0) kv="8-bit cache" ;; *) kv="$kv cache" ;; esac
        echo "llama.cpp, $kv"
    fi
}

serve() {  # backend profile long
    if [ "$1" = vllm ]; then
        systemctl --user disable --now llama-server.service 2>/dev/null || true
        "$ROOT/spark/vllm.sh" switch "$2" "$3"
    else
        systemctl --user disable --now vllm.service 2>/dev/null || true
        if grep -q '^BACKEND=' "$ROOT/config/spark.env"; then
            sed -i 's/^BACKEND=.*/BACKEND=llama/' "$ROOT/config/spark.env"
        else
            echo BACKEND=llama >> "$ROOT/config/spark.env"
        fi
        "$ROOT/spark/service.sh" install
        "$ROOT/spark/service.sh" switch "$2"
    fi
    local want i=0
    if [ "$1" = vllm ]; then want=$(field vllm "$2" SERVED_NAME); else want=$(field llama "$2" ALIAS); fi
    want=${want:-$2}
    until served | tr ' ' '\n' | grep -qx "$want" ||
          { [ "$(field "$1" "$2" KIND)" = decision ] && curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; }; do
        [ "$i" -ge 60 ] && return 1
        sleep 10; i=$((i + 1))
    done
}

if [ -n "${AFTER:-}" ]; then
    log "waiting for process $AFTER to finish"
    while kill -0 "$AFTER" 2>/dev/null; do sleep 30; done
fi

n=0
for item in "$@"; do
    n=$((n + 1))
    IFS=: read -r backend key long <<< "$item"
    long=${long:-0}
    case "$backend" in vllm|llama) ;; *) warn "skipping $item: backend must be vllm or llama"; continue ;; esac
    [ -f "$(profile_file "$backend" "$key")" ] || { warn "skipping $item: no profile"; continue; }
    echo "$item $n/$# $(date +%s) $$" > "$STATE"

    log "[$n/$#] $item: download"
    if [ "$backend" = vllm ]; then "$ROOT/spark/vllm.sh" pull "$key"; else "$ROOT/spark/pull-model.sh" "$key"; fi \
        || { warn "$item: download failed, skipping"; continue; }

    prepare=$(field "$backend" "$key" PREPARE)
    if [ -n "$prepare" ]; then   # a step that needs the memory to itself, such as repacking weights
        log "[$n/$#] $item: stopping the server to run $prepare"
        systemctl --user stop vllm.service llama-server.service 2>/dev/null || true
        # shellcheck disable=SC2086
        "$ROOT"/$prepare > "$LOG_DIR/bake-$key-prepare.log" 2>&1 || { warn "$item: $prepare failed, skipping"; continue; }
    fi

    log "[$n/$#] $item: start"
    if ! serve "$backend" "$key" "$long"; then
        warn "$item: did not start; the last lines of its log are in $LOG_DIR/bake-$key-start.log"
        if [ "$backend" = vllm ]; then cp "$LOG_DIR/vllm-start.log" "$LOG_DIR/bake-$key-start.log" 2>/dev/null
        else journalctl --user -u llama-server.service -n 300 --no-pager -o cat > "$LOG_DIR/bake-$key-start.log" 2>&1; fi
        continue
    fi
    if [ "$backend" = llama ]; then  # the window and memory llama.cpp settled on
        journalctl --user -u llama-server.service -n 400 --no-pager -o cat 2>/dev/null |
            grep -i -E "n_ctx|fit|KV buffer|compute buffer|model buffer|failed" | tail -20 > "$LOG_DIR/bake-$key-start.log"
    fi
    log "[$n/$#] $item: serving, $(mem_available_gb) GB of memory left; measuring (log: logs/bench-$key.log)"
    if [ "$(field "$backend" "$key" KIND)" = decision ]; then   # writes no text: the checker version of the test
        python3 -I -u "$ROOT/scripts/clef-test.py" --base-url "http://127.0.0.1:$PORT" --api-key "${API_KEY:-local}" \
            --model "$(field "$backend" "$key" ALIAS)" --label "$key" --title "$(field "$backend" "$key" TITLE)" \
            --detail "$(field "$backend" "$key" DETAIL)" > "$LOG_DIR/bench-$key.log" 2>&1
    else
        python3 -I -u "$ROOT/scripts/bench.py" --base-url "http://127.0.0.1:$PORT/v1" --api-key "${API_KEY:-local}" \
            --label "$key" --title "$(field "$backend" "$key" TITLE)" --detail "$(field "$backend" "$key" DETAIL)" \
            --note "$(note_for "$backend" "$key" "$long")" > "$LOG_DIR/bench-$key.log" 2>&1
    fi
    log "[$n/$#] $item: $(tail -n 1 "$LOG_DIR/bench-$key.log")"
done

# run/bake.final, when present, names the model to serve afterwards and wins over FINAL,
# so the choice can be changed while the queue runs.
[ -s "$RUN_DIR/bake.final" ] && FINAL=$(head -n 1 "$RUN_DIR/bake.final")
if [ -n "${FINAL:-}" ]; then
    IFS=: read -r backend key long <<< "$FINAL"
    echo "$FINAL final $(date +%s) $$" > "$STATE"
    log "queue done; serving $FINAL"
    serve "$backend" "$key" "${long:-0}" || warn "$FINAL did not start"
fi
log "bake-off finished"
