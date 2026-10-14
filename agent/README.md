# fable

The terminal agent. It talks to any OpenAI-compatible chat endpoint that
supports tool calling (llama.cpp, vLLM, Ollama) and gives the model the tools
a coding agent needs: a shell with a persistent working directory, file read,
write and edit, glob and grep, web fetch, a task list and read-only subagents.

From a checkout:

    uv sync
    uv run fable --doctor            # models, context window, speed, tool-call check
    uv run fable                     # interactive session in the current directory
    uv run fable -p "explain the build"            # one shot
    git diff | uv run fable -p "review this"       # piped input
    uv run fable -y "fix the failing test"         # auto-approve tool calls

Without a checkout, on any machine that can reach the Spark, the control page
gives a command that installs fable from the Spark itself with `uvx`. It only
needs `FABLE_BASE_URL` and `FABLE_API_KEY`: the model defaults to the first
one the server lists, and the kind of server is detected.

Permissions: read-only tools always run. In `ask` mode a shell command runs
without a prompt only if it is obviously harmless (ls, git status, rg and so
on, or a prefix listed in `allowed_commands`); everything else asks, with an
"always for this tool" answer that lasts the session. `-y` approves everything,
`--readonly` removes the writing tools entirely. Non-interactive runs deny what
they cannot ask about, so pass `-y` when piping.

Thinking: `--effort` (or `/effort` in a session) takes `off`, `low`, `medium`,
`high` or `max`; the default is `max`. What that does depends on the model.
A model with a reasoning mode of its own (Qwen3.8, Laguna, Nemotron, gpt-oss)
thinks natively and the server returns it as `reasoning_content`. Each family
names its levels differently, so the agent translates: Qwen3.8 gets `xhigh`
for `high` and `max`, gpt-oss gets `high`, and models that only switch
thinking on or off get no level at all. If a server still refuses the value,
the agent drops it for that model and carries on with the model's default. A
model without a reasoning mode gets a section in the system prompt that asks
for a `<think>` block at the start of every response, with instructions that
get more demanding at each level, and the client lifts the block out of the
stream. Either way the thinking is hidden unless you pass `--think`, it comes
before tool calls, and it stays in the history the model sees. `thinking` in
the config forces one kind when the guess from the model name is wrong, and
`--doctor` checks that thinking and a tool call arrive in the same response.

Context: the agent reads the context window from the server and scales to it.
One tool result may take about a tenth of the window, so with 1M tokens whole
files and long logs come back uncut. When usage passes `compact_threshold` the
agent asks for a continuation summary at the end of the real conversation,
where the server already has the prefix cached, and replaces the history with
it. The system prompt is built once per session for the same reason: anything
that changes it, including `/effort` on a model that is prompted to think,
makes the server process the whole history again.

Sessions are JSONL files under `.fable/sessions/`; `-c` resumes the latest for
the current directory and `-r ID` a specific one. `--sessions` lists them.

Prompts live in `fable_local/prompts/`: `system.md` (the agent's standing
orders, with `{placeholders}` for the environment), `user.md` (the
user's standing instructions; put your own in `user.local.md`, which replaces
it and is ignored by git), `thinking.md` (the `<think>` protocol and one section
per effort level), `subagent.md`, `compact.md` and `tools.md` (one section
per tool, parsed into the tool descriptions the model sees). Edit them freely;
nothing else needs to change. Project files named `AGENTS.md`, `CLAUDE.md` or
`FABLE.md` in the working directory or its parents are appended at startup.

Slash commands in the session: `/help`, `/compact`, `/clear`, `/tokens`,
`/model`, `/effort`, `/think`, `/yolo`, `/tools`, `/todos`, `/cwd`, `/system`.

Licensed under the PolyForm Noncommercial License 1.0.0; see `LICENSE.md` in the repository root.
