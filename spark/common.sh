# Shared helpers for spark/*.sh. Source it, don't run it.
# Everything lives under $ROOT (this tree), including tools, llama.cpp and models.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
TOOLS_DIR="$ROOT/.tools"
export PATH="$TOOLS_DIR/bin:/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export UV_INSTALL_DIR="$TOOLS_DIR/bin" UV_TOOL_DIR="$TOOLS_DIR/uv-tools" UV_TOOL_BIN_DIR="$TOOLS_DIR/bin"
export UV_CACHE_DIR="$TOOLS_DIR/uv-cache" UV_PYTHON_INSTALL_DIR="$TOOLS_DIR/python"

log() { printf '\033[1;34m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# Load config/spark.env (if present) and apply defaults.
load_env() {
    if [ -f "$ROOT/config/spark.env" ]; then
        set -a; . "$ROOT/config/spark.env"; set +a
    fi
    MODEL=${MODEL:-gpt-oss-120b}
    PORT=${PORT:-8080}
    SLOTS=${SLOTS:-1}
    MODELS_DIR=${MODELS_DIR:-$ROOT/models}
    LLAMA_DIR=${LLAMA_DIR:-$ROOT/llama.cpp}
    LOG_DIR=${LOG_DIR:-$ROOT/logs}
    RUN_DIR=${RUN_DIR:-$ROOT/run}
    # Memory (GB) every model must leave free once loaded. With 3 to 5 GB left, long prompts
    # ran the Spark out of memory more than once.
    SAFETY_GB=${SAFETY_GB:-10}
    mkdir -p "$LOG_DIR" "$RUN_DIR"
}

# Load config/models/<key>.env into the environment and set MODEL_KEY.
load_model() {
    MODEL_KEY=${1:?model key required}
    local f="$ROOT/config/models/$MODEL_KEY.env"
    [ -f "$f" ] || die "no model profile at $f (known: $(ls "$ROOT/config/models" | sed 's/\.env$//' | tr '\n' ' '))"
    # reset fields a previous profile may have set
    unset TITLE DETAIL KIND HF_REPO HF_INCLUDE GGUF_MATCH MODEL_DIR ALIAS CTX CTX_MIN FIT_MARGIN_MB KV_TYPE MIN_FREE_GB BATCH UBATCH EXTRA_ARGS TEMPLATE_KWARGS
    set -a; . "$f"; set +a
    ALIAS=${ALIAS:-$MODEL_KEY}
    CTX=${CTX:-32768}
    KV_TYPE=${KV_TYPE:-f16}
    MIN_FREE_GB=${MIN_FREE_GB:-0}
}

# Path to a built llama.cpp binary, or die with a hint.
llama_bin() {
    local b="$LLAMA_DIR/build/bin/$1"
    [ -x "$b" ] || die "$1 not found at $b; run spark/build-llama.sh"
    echo "$b"
}

mem_available_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
mem_total_gb() { awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo; }

# Available memory in GB once it has stopped moving. A server that was just stopped or killed
# gives its memory back over many seconds, and three times a model that started loading within
# six seconds of the last one's exit stalled the whole Spark for hours. Waits until the readings
# over 30 s stay within 1 GB of each other and reach $1 GB, for at most $2 seconds (default
# 300). Prints the last reading; returns 1 if memory never got there.
settled_mem_gb() {
    local want=$1 secs=${2:-300} t=0 n=0 now lo hi
    lo=$(mem_available_gb); hi=$lo
    while :; do
        sleep 5; t=$((t + 5))
        now=$(mem_available_gb)
        if [ $((now - lo)) -gt 1 ] || [ $((hi - now)) -gt 1 ]; then
            lo=$now; hi=$now; n=0   # still moving: start the 30 s over
        else
            [ "$now" -lt "$lo" ] && lo=$now
            [ "$now" -gt "$hi" ] && hi=$now
            n=$((n + 1))
        fi
        if [ "$n" -ge 6 ] && [ "$now" -ge "$want" ]; then echo "$now"; return 0; fi
        if [ "$t" -ge "$secs" ]; then echo "$now"; return 1; fi
    done
}

# Once a server answers: stop its unit ($1) if it leaves less than SAFETY_GB free.
check_margin() {
    local left
    left=$(settled_mem_gb 0 120) || true
    if [ "$left" -lt "$SAFETY_GB" ]; then
        systemctl --user stop "$1"
        warn "only ${left} GB free with the model loaded, under the ${SAFETY_GB} GB margin (SAFETY_GB); stopped $1"
        return 1
    fi
    log "${left} GB free with the model loaded (margin ${SAFETY_GB} GB)"
}

# Wait for an llama-server health endpoint. $1 = port, $2 = seconds (default 600).
wait_healthy() {
    local port=$1 secs=${2:-600} i=0
    while [ "$i" -lt "$secs" ]; do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then return 0; fi
        sleep 2; i=$((i + 2))
    done
    return 1
}
