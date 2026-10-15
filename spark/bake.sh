#!/usr/bin/env bash
# Measure models one after another and file the results for the control page.
# For each item: download it if needed, serve it, wait until it answers, then run
# scripts/bench.py. Items are backend:profile[:long], with long as for vllm.sh switch.
#
#   spark/bake.sh llama:deepseek-v4-flash vllm:qwen3.6-35b-a3b:1 vllm:qwen3.5-122b
#   FINAL=vllm:qwen3.8-27b-fp8:1 spark/bake.sh ...   # serve this one when the queue is done
#   AFTER=<pid> spark/bake.sh ...                    # let a running test finish first
#   echo vllm:qwen3.8-27b-fp8:1 > run/bake.final     # or decide the final model later
#   echo llama:glm-5.3-flash >> run/bake.next        # add an item to run next, while the queue runs
#   echo llama:seed-oss-36b >> run/bake.later        # or one to run after the rest
#
# Progress goes to logs/bake.log, each model's run to logs/bench-<profile>.log and the
# results to run/results.jsonl. run/bake.state names the model being measured, for the
# control page. A download is tried three times; a model that still fails to download, or
# fails to start, is logged and skipped. The bake gives way to the other jobs on the box
# (give_way and make_room below), and measures again a model it stopped for them.
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
        effort=$(field llama "$2" TEMPLATE_KWARGS | sed -n 's/.*"reasoning_effort" *: *"\([a-z]*\)".*/\1/p')
        echo "llama.cpp, $kv${effort:+, reasoning effort $effort}"
    fi
}

serve() {  # backend profile long
    # switch fails when the server does not come up or leaves less than SAFETY_GB free
    if [ "$1" = vllm ]; then
        systemctl --user disable --now llama-server.service 2>/dev/null || true
        "$ROOT/spark/vllm.sh" switch "$2" "$3" || return 1
    else
        systemctl --user disable --now vllm.service 2>/dev/null || true
        if grep -q '^BACKEND=' "$ROOT/config/spark.env"; then
            sed -i 's/^BACKEND=.*/BACKEND=llama/' "$ROOT/config/spark.env"
        else
            echo BACKEND=llama >> "$ROOT/config/spark.env"
        fi
        "$ROOT/spark/service.sh" install
        "$ROOT/spark/service.sh" switch "$2" || return 1
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

# The other jobs on this box (steel, and flint on its weekday timer) look at the free memory
# once when they start and allocate it minutes later, after compiling. A model loaded in
# that gap runs the Spark out of memory when they do: on 2026-10-15 at 06:45 steel's first
# training step hung the Spark with GLM-5.3-Flash loaded, and the hang was blamed on GLM at
# first. Steel also restarts itself every 30 s until it finds the memory free. So the bake
# gives way: while one of them is up it stops its server and loads nothing.
# NEIGHBOURS="" turns this off.
neighbours_up() {  # the units in NEIGHBOURS that are running or about to
    local u
    for u in ${NEIGHBOURS-steel.service flint-run.service}; do
        case "$(systemctl --user is-active "$u" 2>/dev/null)" in
            active|activating|reloading) printf '%s ' "$u" ;;
        esac
    done
}
give_way() {  # what is waiting: returns once no neighbour is up, with the server stopped if one was
    local busy
    busy=$(neighbours_up)
    [ -n "$busy" ] || return 0
    log "$1: ${busy% } is running; stopping the server until it is done"
    systemctl --user stop llama-server.service vllm.service 2>/dev/null || true
    while [ -n "$(neighbours_up)" ]; do sleep 60; done
    log "$1: ${busy% } is done; going on"
}

# Flint starts from a timer on weekdays (19:45) and needs most of the memory, so the bake
# makes room LEAD_MIN minutes before a timer in NEIGHBOUR_TIMERS fires: it stops its server,
# stopping a measurement if one is running (that model is measured again afterwards), lets
# the timer fire and then gives way to what it started. NEIGHBOUR_TIMERS="" turns this off.
timer_due() {  # prints "epoch timer" for the first timer that fires within LEAD_MIN minutes
    local t next at now
    now=$(date +%s)
    for t in ${NEIGHBOUR_TIMERS-flint.timer}; do
        next=$(systemctl --user show "$t" -p NextElapseUSecRealtime --value 2>/dev/null)
        [ -n "$next" ] || continue   # no such timer, or it is stopped
        at=$(date -d "$next" +%s 2>/dev/null) || continue
        if [ "$at" -gt "$now" ] && [ $((at - now)) -le $((${LEAD_MIN:-5} * 60)) ]; then
            echo "$at $t"
            return 0
        fi
    done
    return 1
}
make_room() {  # what is waiting
    local due at
    due=$(timer_due) || return 0
    at=${due%% *}
    log "$1: ${due#* } fires at $(date -d "@$at" +%H:%M); stopping the server until it has run"
    systemctl --user stop llama-server.service vllm.service 2>/dev/null || true
    while [ "$(date +%s)" -lt $((at + 120)) ]; do sleep 15; done   # what it starts checks the memory within seconds
    give_way "$1"
}

# While a model is measured: stop the measurement and the server when a timer is about to
# fire (above), and stop the servers if free memory stays under GUARD_GB for 4 s. Each model
# starts with SAFETY_GB free, but a long prompt can need more memory than it held at idle.
guard() {  # item bench-pid
    local low=0 i=0 kids
    while sleep 2; do
        i=$((i + 1))
        if [ $((i % 15)) = 0 ] && timer_due >/dev/null; then
            warn "$1: a timer is about to fire; stopping the measurement to make room"
            touch "$RUN_DIR/bake.interrupted"
            kids=$(pgrep -P "$2")
            # shellcheck disable=SC2086
            kill "$2" $kids 2>/dev/null   # bench.py writes its result at the end, so nothing is recorded
            systemctl --user stop llama-server.service vllm.service 2>/dev/null
            return
        fi
        if [ "$(mem_available_gb)" -lt "${GUARD_GB:-3}" ]; then
            low=$((low + 1))
            if [ "$low" -ge 2 ]; then
                warn "$1: under ${GUARD_GB:-3} GB free while measuring; stopping the server"
                systemctl --user stop llama-server.service vllm.service 2>/dev/null
                return
            fi
        else
            low=0
        fi
    done
}

if [ -n "${AFTER:-}" ]; then
    log "waiting for process $AFTER to finish"
    while kill -0 "$AFTER" 2>/dev/null; do sleep 30; done
fi

take() {  # file: its non-empty lines into $got, and the file removed
    got=()
    [ -s "$1" ] || return 1
    mapfile -t got < <(grep -v '^[[:space:]]*$' "$1")
    rm -f "$1"
    [ "${#got[@]}" -gt 0 ]
}

items=("$@")
n=0
while :; do
    if take "$RUN_DIR/bake.next"; then items=("${items[@]:0:n}" "${got[@]}" "${items[@]:n}"); log "added next: ${got[*]}"; fi
    if take "$RUN_DIR/bake.later"; then items+=("${got[@]}"); log "added at the end: ${got[*]}"; fi
    [ "$n" -lt "${#items[@]}" ] || break
    item=${items[$n]}
    n=$((n + 1))
    of="$n/${#items[@]}"
    IFS=: read -r backend key long <<< "$item"
    long=${long:-0}
    case "$backend" in vllm|llama) ;; *) warn "skipping $item: backend must be vllm or llama"; continue ;; esac
    [ -f "$(profile_file "$backend" "$key")" ] || { warn "skipping $item: no profile"; continue; }
    echo "$item $of $(date +%s) $$" > "$STATE"

    log "[$of] $item: download"
    pulled=0
    for try in 1 2 3; do   # the line is busy with other downloads, and a check can time out
        if [ "$backend" = vllm ]; then "$ROOT/spark/vllm.sh" pull "$key"; else "$ROOT/spark/pull-model.sh" "$key"; fi && { pulled=1; break; }
        [ "$try" = 3 ] || { warn "$item: download attempt $try did not work; trying again in a minute"; sleep 60; }
    done
    [ "$pulled" = 1 ] || { warn "$item: download failed, skipping"; continue; }
    make_room "[$of] $item"
    give_way "[$of] $item"

    prepare=$(field "$backend" "$key" PREPARE)
    if [ -n "$prepare" ]; then   # a step that needs the memory to itself, such as repacking weights
        log "[$of] $item: stopping the server to run $prepare"
        systemctl --user stop vllm.service llama-server.service 2>/dev/null || true
        # shellcheck disable=SC2086
        "$ROOT"/$prepare > "$LOG_DIR/bake-$key-prepare.log" 2>&1 || { warn "$item: $prepare failed, skipping"; continue; }
    fi

    log "[$of] $item: start"
    if ! serve "$backend" "$key" "$long"; then
        warn "$item: did not start; the last lines of its log are in $LOG_DIR/bake-$key-start.log"
        if [ "$backend" = vllm ]; then cp "$LOG_DIR/vllm-start.log" "$LOG_DIR/bake-$key-start.log" 2>/dev/null
        else journalctl --user -u llama-server.service -n 300 --no-pager -o cat > "$LOG_DIR/bake-$key-start.log" 2>&1; fi
        continue
    fi
    # One that started while the model loaded saw the memory free and takes it once it has
    # compiled. Flint refuses within seconds when it does not see enough, so allow a minute.
    if [ -n "$(neighbours_up)" ] && sleep 60 && [ -n "$(neighbours_up)" ]; then
        n=$((n - 1))   # measure this one again afterwards
        give_way "[$of] $item"
        continue
    fi
    if [ "$backend" = llama ]; then  # the window and memory llama.cpp settled on
        journalctl --user -u llama-server.service -n 400 --no-pager -o cat 2>/dev/null |
            grep -i -E "n_ctx|fit|KV buffer|compute buffer|model buffer|failed" | tail -20 > "$LOG_DIR/bake-$key-start.log"
    fi
    log "[$of] $item: serving, $(mem_available_gb) GB of memory left; measuring (log: logs/bench-$key.log)"
    rm -f "$RUN_DIR/bake.interrupted"
    if [ "$(field "$backend" "$key" KIND)" = decision ]; then   # writes no text: the checker version of the test
        python3 -I -u "$ROOT/scripts/clef-test.py" --base-url "http://127.0.0.1:$PORT" --api-key "${API_KEY:-local}" \
            --model "$(field "$backend" "$key" ALIAS)" --label "$key" --title "$(field "$backend" "$key" TITLE)" \
            --detail "$(field "$backend" "$key" DETAIL)" > "$LOG_DIR/bench-$key.log" 2>&1 &
    else
        python3 -I -u "$ROOT/scripts/bench.py" --base-url "http://127.0.0.1:$PORT/v1" --api-key "${API_KEY:-local}" \
            --label "$key" --title "$(field "$backend" "$key" TITLE)" --detail "$(field "$backend" "$key" DETAIL)" \
            --note "$(note_for "$backend" "$key" "$long")" > "$LOG_DIR/bench-$key.log" 2>&1 &
    fi
    bench_pid=$!
    guard "$item" "$bench_pid" &
    guard_pid=$!
    wait "$bench_pid"
    kill "$guard_pid" 2>/dev/null; wait "$guard_pid" 2>/dev/null
    if [ -f "$RUN_DIR/bake.interrupted" ]; then
        rm -f "$RUN_DIR/bake.interrupted"
        log "[$of] $item: stopped to make room; it is measured again afterwards"
        n=$((n - 1))
        make_room "[$of] $item"
        continue
    fi
    log "[$of] $item: $(tail -n 1 "$LOG_DIR/bench-$key.log")"
done

# run/bake.final, when present, names the model to serve afterwards and wins over FINAL,
# so the choice can be changed while the queue runs.
[ -s "$RUN_DIR/bake.final" ] && FINAL=$(head -n 1 "$RUN_DIR/bake.final")
if [ -n "${FINAL:-}" ]; then
    IFS=: read -r backend key long <<< "$FINAL"
    echo "$FINAL final $(date +%s) $$" > "$STATE"
    log "queue done; serving $FINAL"
    give_way "$FINAL"
    serve "$backend" "$key" "${long:-0}" || warn "$FINAL did not start"
fi
log "bake-off finished"
