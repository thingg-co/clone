# Point Claude Code at the Spark. Source it, then run `claude`:
#
#   source claude-code/env.sh            # model name taken from config/spark.env
#   source claude-code/env.sh gpt-oss-120b
#   CLAUDE_LOCAL_HOST=localhost source claude-code/env.sh    # through scripts/tunnel.sh
#
# Needs a server that speaks the Anthropic Messages API on /v1/messages (vLLM
# does; a recent llama.cpp does, build-llama.sh reports it). Otherwise run the
# LiteLLM proxy described in claude-code/README.md and source this with
# CLAUDE_LOCAL_HOST=localhost CLAUDE_LOCAL_PORT=4000.
_src=${BASH_SOURCE[0]:-$0}
_root=$(cd "$(dirname "$_src")/.." && pwd)
if [ ! -f "$_root/config/spark.env" ]; then echo "no config/spark.env (make config)" >&2; return 1 2>/dev/null || exit 1; fi
set -a; . "$_root/config/spark.env"; set +a

_model=${1:-}
if [ -z "$_model" ]; then
    if [ "${BACKEND:-vllm}" = vllm ]; then
        _model=$(sed -n 's/^SERVED_NAME=//p' "$_root/config/vllm/${VLLM_PROFILE:-qwen3.8-27b}.env")
    else
        _model=$(sed -n 's/^ALIAS=//p' "$_root/config/models/${MODEL}.env")
    fi
fi
_host=${CLAUDE_LOCAL_HOST:-$SPARK_HOST}
_port=${CLAUDE_LOCAL_PORT:-$PORT}

export ANTHROPIC_BASE_URL="http://${_host}:${_port}"
export ANTHROPIC_AUTH_TOKEN="${CLAUDE_LOCAL_TOKEN:-$API_KEY}"
export ANTHROPIC_MODEL="$_model"
export ANTHROPIC_SMALL_FAST_MODEL="$_model"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$_model"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$_model"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$_model"
export CLAUDE_CODE_SUBAGENT_MODEL="$_model"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export DISABLE_PROMPT_CACHING=1
export DISABLE_TELEMETRY=1
export API_TIMEOUT_MS=3600000
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384
echo "claude -> $ANTHROPIC_BASE_URL as $ANTHROPIC_MODEL"
unset _src _root _model _host _port
