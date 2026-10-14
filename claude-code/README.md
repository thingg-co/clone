# Claude Code on the Spark

Claude Code reads a handful of environment variables to talk to a different
backend. Point it at the Spark and it drives the local model with its own
tools and UI. Two routes, depending on what the server speaks.

## Route A: the server has /v1/messages

vLLM has an Anthropic-compatible endpoint, and recent llama.cpp builds do too
(`spark/build-llama.sh` prints which; it also leaves `run/has-anthropic-api`
on the Spark when the build has it). Check from the Mac:

    curl -s -X POST http://omega.local:8080/v1/messages \
      -H "Authorization: Bearer $API_KEY" -H "content-type: application/json" \
      -d '{"model":"qwen3.8-27b","max_tokens":2000,"messages":[{"role":"user","content":"hi"}]}'

vLLM reads the key from the `Authorization` header only, which is also where
Claude Code puts `ANTHROPIC_AUTH_TOKEN`; an `x-api-key` header gets a 401. A
JSON reply means it works (the control page has a button for the same test):

    source claude-code/env.sh        # or: source claude-code/env.sh gpt-oss-120b
    claude

Through the tunnel instead of the LAN: `CLAUDE_LOCAL_HOST=localhost source claude-code/env.sh`.

## Route B: LiteLLM proxy

When the server only has /v1/chat/completions, run LiteLLM on the Mac. It
translates the Messages API, including tool use and streaming.

    export FABLE_OPENAI_BASE=http://omega.local:8080/v1
    export FABLE_SIDEKICK_BASE=http://omega.local:8081/v1
    export FABLE_API_KEY=<API_KEY from config/spark.env>
    uvx --from 'litellm[proxy]' litellm --config claude-code/litellm.yaml --port 4000

    CLAUDE_LOCAL_HOST=localhost CLAUDE_LOCAL_PORT=4000 CLAUDE_LOCAL_TOKEN=sk-local \
      source claude-code/env.sh local
    claude

Edit the `model:` lines in `litellm.yaml` when the served model name changes.

## What to expect

- Every model slot (main, small-fast, subagents) is the same local model
  unless you run a second server and route the small one through LiteLLM.
- Prompt caching is disabled through `DISABLE_PROMPT_CACHING`; llama.cpp and
  vLLM cache prefixes on their own anyway.
- Claude Code assumes Anthropic-sized context windows. With a 500k or 1M
  vLLM server that is conservative, and auto-compaction will fire early. Fine.
- Models that reason natively return their reasoning as a thinking block.
- Claude Code asks for `high` effort. Qwen3.8 chat templates only know `xhigh`,
  `medium` and `low` and answer anything else with a 400. Profiles with
  `EFFORT_ALIAS=1` serve a copy of the template that maps `high` and `max` to
  `xhigh` first, so this works without touching Claude Code.
- A cold prompt of several hundred thousand tokens takes many minutes to read
  in. `API_TIMEOUT_MS` in `env.sh` is an hour for that reason.
- If `claude` reports an auth error, the token is wrong: `ANTHROPIC_AUTH_TOKEN`
  must equal `API_KEY` on the Spark (or `sk-local` for the proxy).
