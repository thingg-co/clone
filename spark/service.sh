#!/usr/bin/env bash
# Manage the llama-server instances as systemd user services (survives logout:
# linger is already enabled on the Spark). `bg` is a no-systemd fallback that
# uses nohup and a pid file under run/.
#
#   spark/service.sh install            # write units into ~/.config/systemd/user and enable them
#   spark/service.sh start|stop|restart|status|logs [main|sidekick]
#   spark/service.sh switch glm-4.5-air # change MODEL in config/spark.env and restart main
#   spark/service.sh bg [main|sidekick] # run detached without systemd
#   spark/service.sh bg-stop [main|sidekick]
set -euo pipefail
. "$(dirname "$0")/common.sh"
load_env

cmd=${1:-status}
inst=${2:-main}
unit_name() { case "$1" in main) echo llama-server;; sidekick) echo llama-sidekick;; *) die "instance must be main or sidekick";; esac; }
has_sidekick() { [ -n "${SIDEKICK_MODEL:-}" ]; }

case "$cmd" in
    install)
        mkdir -p "$HOME/.config/systemd/user"
        for u in llama-server llama-sidekick clone-panel; do
            sed "s#@ROOT@#$ROOT#g" "$ROOT/spark/$u.service" > "$HOME/.config/systemd/user/$u.service"
        done
        systemctl --user daemon-reload
        if [ "${BACKEND:-vllm}" = llama ]; then
            systemctl --user enable llama-server.service >/dev/null
        else
            systemctl --user disable llama-server.service >/dev/null 2>&1 || true
            log "BACKEND=vllm: llama-server main unit installed but not enabled (vllm.sh install handles the main port)"
        fi
        systemctl --user enable clone-panel.service >/dev/null
        if has_sidekick; then systemctl --user enable llama-sidekick.service >/dev/null; else systemctl --user disable llama-sidekick.service >/dev/null 2>&1 || true; fi
        loginctl enable-linger "$USER" 2>/dev/null || warn "could not enable linger (try: sudo loginctl enable-linger $USER); services stop at logout without it"
        log "installed; start with: spark/service.sh start"
        ;;
    uninstall)
        systemctl --user disable --now llama-server.service llama-sidekick.service clone-panel.service 2>/dev/null || true
        rm -f "$HOME/.config/systemd/user/llama-server.service" "$HOME/.config/systemd/user/llama-sidekick.service" "$HOME/.config/systemd/user/clone-panel.service"
        systemctl --user daemon-reload
        ;;
    start|stop|restart)
        if [ "$#" -ge 2 ]; then
            systemctl --user "$cmd" "$(unit_name "$inst").service"
        else
            systemctl --user "$cmd" llama-server.service
            if has_sidekick; then systemctl --user "$cmd" llama-sidekick.service; fi
        fi
        ;;
    status)
        systemctl --user --no-pager status llama-server.service llama-sidekick.service 2>&1 | grep -E 'service|Active|Main PID' || true
        for p in "$PORT" "${SIDEKICK_PORT:-}"; do
            [ -n "$p" ] || continue
            if curl -sf "http://127.0.0.1:$p/health" >/dev/null 2>&1; then
                printf ':%s healthy, models: %s\n' "$p" "$(curl -s ${API_KEY:+-H "Authorization: Bearer $API_KEY"} "http://127.0.0.1:$p/v1/models" | python3 -I -c 'import sys,json;print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null)"
            else
                printf ':%s not answering\n' "$p"
            fi
        done
        ;;
    logs)
        journalctl --user -u "$(unit_name "$inst").service" -f -n 100
        ;;
    switch)
        key=${2:?usage: service.sh switch <model-key>}
        load_model "$key"   # validates the profile
        f="$ROOT/config/spark.env"
        if grep -q '^MODEL=' "$f"; then sed -i "s/^MODEL=.*/MODEL=$key/" "$f"; else echo "MODEL=$key" >> "$f"; fi
        log "MODEL=$key written to config/spark.env; restarting main"
        systemctl --user restart llama-server.service
        wait_healthy "$PORT" 900 && log "healthy on :$PORT" || warn "not healthy after 15 min; check: spark/service.sh logs"
        ;;
    bg)
        pidf="$RUN_DIR/$inst.pid"
        if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then die "$inst already running (pid $(cat "$pidf"))"; fi
        nohup setsid bash "$ROOT/spark/serve.sh" "$inst" > "$LOG_DIR/$inst.log" 2>&1 &
        echo $! > "$pidf"
        log "$inst started, pid $(cat "$pidf"), log $LOG_DIR/$inst.log"
        ;;
    bg-stop)
        pidf="$RUN_DIR/$inst.pid"
        [ -f "$pidf" ] || die "no pid file for $inst"
        kill -INT "$(cat "$pidf")" && rm -f "$pidf" && log "$inst stopped"
        ;;
    *) die "unknown command $cmd" ;;
esac
