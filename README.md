# 👽 clone

A local coding agent on the DGX Spark. Two parts: a model server on the Spark
with as much context as the box can hold, and `fable`, a terminal agent with
tool calling that works the way Claude Code does. The target is three things
at once: a context window of 500k tokens or more, deep thinking before the
model acts, and reliable tool calls. Everything stays inside this
directory on both machines: tools, the llama.cpp build, weights, logs, the
agent's venv and sessions.

What it is not: a frontier model. On their vendors' own numbers the best
models that fit this box score around 60 on SWE-bench Pro, and the comparison
table on Poolside's Laguna card puts the hosted model this is modelled on at
80. It is slower and less sharp, and the prompts are written to make it
careful.

## Hardware, and what follows from it

The Spark is sold as 128 GB. The system reports 121.7 GiB of unified memory,
at 273 GB/s. Two things follow from those numbers.

Decode speed is roughly bandwidth divided by the bytes of weights read per
token. A sparse model with a few billion active parameters runs well however
large it is in total, and a dense model reads all of itself for every token.
Measured here: 49 tok/s for a 125B model with 6B active, 12 tok/s for a dense
27B.

Context comes out of what the weights leave. vLLM takes a fixed share of the
121.7 GiB (`MEM_UTIL` in a profile), loads the weights and gives the rest to
the KV cache, so a model with 93 GiB of weights has room for about 350k tokens
of cache whatever window it was trained for. Reserving more than about 0.80
of the box does not work: at that setting with a large cache the GPU driver
ran out of memory during warm-up and the server hung.

## What was measured

`scripts/context-test.py` is the acceptance test. It fills the prompt with
numbered records ("Record 9337: the access code for locker 9337 is 836244."),
asks for three of the codes, one near the start, one in the middle and one
near the end, and wants them back through a tool call. A size passes when the
call is made and all three codes are right. Thousands of nearly identical
lines make it a harsh test. It is also the property a long window is for, and
one model passes it.

Results from 2026-10-14, all on this Spark:

| model | weights | window served | decode | recall |
|---|---|---|---|---|
| Qwen3.8-27B, dense | 22 GB | 950k, stretched from 256k | 12 tok/s | 3/3 at 97k, 245k and 507k |
| Qwen3.8-Flash-Next, 125B with 6B active | 133 GB, 75 GiB resident | 524k, stretched from 256k | 49 tok/s | 3/3 at 93k and 253k, 2/3 at 411k, 505k and 516k |
| Nemotron 3 Super, 120B with 12B active | 80 GB | 1M, native | 16 tok/s | 2/3 at 93k, 1/3 at 235k and 489k |
| Nemotron 3.5 Lightning, 30B with 3B active | 22 GB | 1M, native | 136 tok/s | 3/3 at 102k, 1/3 at 232k, 482k and 986k |
| Qwen3-Coder-Next, 80B with 3B active | 80 GB | 256k, native | 45 tok/s | not run |

Every row passes `fable --doctor` at the window listed: thinking and a tool
call in one response, then the tool loop. The builds are NVFP4, except
Coder-Next, which was FP8 and has since been deleted.

How they fail matters as much as where. The Nemotron models return a code
from the wrong part of the prompt. Flash-Next past its trained length returns
the neighbouring record: asked for locker 9337 it gave locker 9338's code, and
for locker 137 it gave locker 37's, each time quoting the line it believed it
had read. An agent that is nearly right about what it read is worse than a
slow one.

Every Qwen window past 256k is stretched. The models are trained to 262,144
tokens and their cards document YaRN for more, which rescales positions for
every prompt, long or short. What that does depends on the model, so it has
to be tested each time:

- Qwen3.8-27B, with dense attention in a quarter of its layers, stayed exact
  at 507k with a factor of 4.
- Qwen3.8-Flash-Next attends sparsely and loses its place past the trained
  length. A factor of 2 with a 524k window does better than a factor of 4 with
  558k (2/3 against 0/3 at about 517k). Neither is exact.
- Qwen3-Coder-Next, an older model whose card only claims 256k, stopped
  calling tools altogether at a factor of 4.

Long prompts cost time. Flash-Next reads about 2,500 tok/s, so a cold 500k
prompt takes under four minutes. The dense 27B reads 900 tok/s for the first
100k and under 300 past 250k, and the same prompt takes 22 minutes. The
server caches what it has read, and a request that extends the previous
prompt only pays for the new part. For these hybrid models the cache resumes
from the end of an earlier prompt, so anything that changes the start of the
conversation, or shortens it, means reading it all again. The agent keeps its
system prompt fixed for that reason.

Where that leaves the three targets: Qwen3.8-27B with its stretched window is
the only configuration that has been exact past 500k, with thinking and tool
calls, and it is slow. Flash-Next is the model that suits the hardware and is
exact inside 256k. Nothing tested so far is exact at 1M.

## Profiles

A profile in `config/vllm/` names the weights, the window, the parsers and the
memory share. `VLLM_PROFILE` and `VLLM_LONG` in `config/spark.env` pick one.

| profile | what it is | windows |
|---|---|---|
| `qwen3.8-27b` | the one that passes the recall test past 500k | 256k native; `VLLM_LONG=1` or a token count stretches it |
| `qwen3.8-flash-next` | fastest and strongest coder, served through a patched vLLM (below) | `VLLM_LONG=0` asks for 500k, `1` for the most that fits |
| `laguna-s-2.1` | Poolside's 118B coding model, trained to 1M | what its 93 GiB of weights leave room for |
| `nemotron-3-super`, `nemotron-3.5-lightning` | native 1M windows that fail the recall test | 1M |
| `gpt-oss-120b` | 128k | 128k |

When vLLM refuses a window because the cache cannot hold it, it says what
would fit. The launchers take 97% of that figure, write it to
`run/vllm-ctx-<profile>` (`run/flash-next-ctx` for Flash-Next) and start again
with it, and later starts go straight there. Delete the file to have the full
window tried again.

Three ways to serve, one port:

- vLLM in NVIDIA's DGX Spark container (`spark/vllm.sh`). Proper tool-call
  and reasoning parsers, prefix caching, and an Anthropic-style
  `/v1/messages` endpoint for Claude Code. The default.
- Qwen3.8-Flash-Next through `spark/flash-next.sh`. Its checkpoint is larger
  than the machine's memory, so the stock image cannot load it. The launcher
  uses the [qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
  recipe (Apache-2.0), cloned under `third_party/` at a pinned commit, which
  patches vLLM 0.31 to read the 48 GiB n-gram table from the NVMe. It sets
  itself up on first use: the image build and checkpoint preparation take
  about twenty minutes once.
- llama.cpp built from source (`spark/serve.sh`, GGUF profiles in
  `config/models/`). 128k windows, good for gpt-oss, and it can run a small
  sidekick model on a second port when the main profile leaves memory for
  one. None of the vLLM profiles above do.

## Layout

    config/spark.env.example   hostname, backend, model, port, API key (copy to spark.env)
    config/vllm/*.env          vLLM profiles: HF repo, window, parsers, memory share
    config/models/*.env        llama.cpp profiles: GGUF repo, context, KV type
    spark/bootstrap.sh         one-shot setup on the Spark; idempotent; no sudo
    spark/build-llama.sh       llama.cpp with CUDA for sm_121, into llama.cpp/
    spark/pull-model.sh        GGUF download into models/<key>/ (checks the repo first)
    spark/vllm.sh              pull | serve | switch | install | stop | logs | image
    spark/flash-next.sh        Qwen3.8-Flash-Next through the pinned third-party recipe
    spark/serve.sh             llama-server for a profile (main, sidekick, or any key)
    spark/service.sh           systemd user units for the llama.cpp instances
    spark/panel.py             control page: status, profile and context switch, connection strings
    spark/preflight.sh         memory, GPU processes, ports, does the profile fit
    spark/bench.sh             llama-bench at 0 and 8k depth
    spark/ollama.sh            fallback: Ollama from the tarball, in-tree
    scripts/deploy.sh          rsync this tree to the Spark (and run bootstrap); leaves its spark.env alone
    scripts/tunnel.sh          ssh -L for the model ports and the control page
    scripts/status.sh          what the Spark is doing
    scripts/smoke-test.py      tool-calling check, standard library only
    scripts/context-test.py    recall across the window through a tool call; the acceptance test
    agent/                     fable, the agent (see agent/README.md)
    docs/                      the project page, served by GitHub Pages
    claude-code/               run Claude Code against the Spark
    Makefile                   the above as targets; make help
    LICENSE.md                 PolyForm Noncommercial 1.0.0

## First run

On the Mac:

    make config                       # writes config/spark.env with a fresh API key
    $EDITOR config/spark.env          # SPARK_HOST, SPARK_USER, BACKEND, VLLM_PROFILE or MODEL
    make deploy                       # rsync to ~/clone on the Spark (or copy the tree however you like)

On the Spark, in the tree:

    spark/bootstrap.sh                # tools into .tools/, llama.cpp build, weights, services, smoke test
    spark/bootstrap.sh --no-pull      # same without the downloads

The bootstrap builds llama.cpp (about 15 minutes on the 20 cores), pulls the
container image (37 GB) and the weights of the profile in `config/spark.env`
(22 GB for Qwen3.8-27B, 133 GB for Flash-Next), installs systemd user units,
starts them, waits for the health check and runs the smoke test. Nothing
needs sudo; linger is already on, so the services outlive the SSH session.
Rerunning it is safe. On Wi-Fi the downloads are the slow part: this Spark
managed about 30 MB/s on a 2.4 GHz network, and a cable is worth finding.

Back on the Mac:

    make panel                        # control page at http://omega.local:8088
    make tunnel &                     # localhost:8080 (and :8081) -> Spark
    make agent-install
    make doctor                       # models, context window, tok/s, thinking and tool-call check
    make agent                        # interactive session in the current directory

Then the test that decides whether a profile is worth using, on the Spark so
it does not depend on the laptop staying awake:

    python3 scripts/context-test.py --base-url http://127.0.0.1:8080/v1 --api-key KEY --sizes 100,250,500

The agent runs anywhere with network access to the Spark, and does not need
this checkout there. The control page prints the command, which looks like
this:

    FABLE_BASE_URL=http://omega.local:8080/v1 \
    FABLE_API_KEY=... \
    uvx --from "fable-local @ http://omega.local:8088/dist/<hash>/fable_local-0.2.1.tar.gz" fable

To run it on the Spark itself: `spark/bootstrap.sh --agent`, then
`agent/.venv/bin/fable`.

## Control page

`spark/panel.py` is a small page served from the Spark, standard library only,
installed as the user unit `clone-panel` by the bootstrap. It is on the LAN at
http://omega.local:8088 (`make panel` opens it). There is no login.

It shows what is being served, the context window and memory, and has buttons
for start, stop and restart. Downloads in progress get a bar each, with speed
and a finish time. The profile table lists every vLLM and llama.cpp profile
with its window range, how it thinks, its tool parser and whether the weights
are on disk; pick one, pick a window and it rewrites `config/spark.env` on the
Spark, reinstalls the unit and restarts. Weights can be pulled from the same
table. The Connect box prints the full command for Claude Code and for fable
against whichever host name you choose, with copy buttons, and two tests: one
request to `/v1/messages`, which is what Claude Code needs, and the
tool-calling check. The host list has every name the Spark answers to
(`omega.local` on the LAN, the bare name for Tailscale or a DNS entry of your
own, its addresses) and a box for any other. Both commands work on any
machine that can reach the Spark: the page hands fable out as an installable
archive, so its command is two variables and one `uvx` line. Output of whatever ran last is at the bottom, next to the
service logs. The page updates itself over server-sent events.

`PANEL_HOST=0.0.0.0` in `config/spark.env` is what puts it on the LAN. Set it
to `127.0.0.1` to keep the page to the SSH tunnel (http://localhost:8088 with
`make tunnel` running). The page shows the API key and can restart services,
and anyone on the network can open it.
A profile switched from the page changes `config/spark.env` on the Spark,
which is the copy that counts: `make deploy` sends the Mac's only when the
Spark has none. `make vllm-switch` changes both.

## Day to day

    make status                       # memory, GPU, services, endpoints
    make logs                         # follow the main server
    make restart | stop | start
    make smoke                        # tool-calling check through the tunnel
    make switch MODEL=glm-4.5-air     # llama.cpp backend: change profile and restart
    make test                         # agent unit tests

vLLM profile changes go through one target, which edits `config/spark.env`
here and on the Spark, reinstalls the unit and restarts:

    make vllm-switch PROFILE=qwen3.8-27b LONG=1               # exact past 500k, slow
    make vllm-switch PROFILE=qwen3.8-27b LONG=0               # the same model at its native 256k
    make vllm-switch PROFILE=qwen3.8-flash-next LONG=0        # fast, exact inside 256k

Each profile is its own download; the bootstrap pulls only the one in
`config/spark.env`, so run `spark/vllm.sh pull <profile>` on the Spark, or use
the page, before switching to another. To try a profile in the foreground
first: `systemctl --user stop vllm; spark/vllm.sh serve qwen3.8-27b`.

Memory is checked before a model starts (`MIN_FREE_GB` in each profile).
`spark/preflight.sh` shows what is using the box if a start is refused;
`FORCE=1` overrides.

## The agent

`fable` is a Python package in `agent/`. It streams from any OpenAI-compatible
endpoint and gives the model: `bash` with a persistent working directory,
`read_file`, `write_file`, `edit_file` (exact-match replace with a diff),
`glob_files`, `grep_files`, `list_dir`, `web_fetch`, `todo_write` and
`spawn_subagent` (a fresh read-only agent whose report comes back as the tool
result, optionally on the sidekick model). Permissions ask before anything
that changes state, with an allowlist for obviously safe commands. The
context window is read from the server, tool output limits scale with it, and
the history is summarised when it fills. Sessions are saved and resumable.

Thinking has five levels (`--effort off|low|medium|high|max`, default `high`).
On a model that reasons natively the agent translates the level into whatever
that family calls it (Qwen3.8 says `xhigh`, some models only switch thinking
on or off) and keeps the reasoning in the history. On one that does not, the
system prompt asks for a `<think>` block before every answer and tool call,
and the levels change how much work that block has to do. Details in
`agent/README.md`.

The prompts are in `agent/fable_local/prompts/` as Markdown: the system
prompt (identity, how the session works, delivering work, working with code,
writing as the user, writing for the user, planning), the user's standing
instructions, the thinking protocol, the subagent brief, the compaction
brief, and the tool descriptions. A file named `<name>.local.md` replaces
`<name>.md` and is ignored by git, so `user.local.md` is the place for
instructions of your own that should travel with the agent. They are modeled on the instructions Claude Code runs under.

## Claude Code against the Spark

`claude-code/README.md`. Short version: `source claude-code/env.sh && claude`
when the server has `/v1/messages` (vLLM does), or run the LiteLLM proxy from
`claude-code/litellm.yaml` when it does not. The control page prints the same
variables as one command to copy.

## License

PolyForm Noncommercial License 1.0.0. The full terms are in `LICENSE.md` and at
<https://polyformproject.org/licenses/noncommercial/1.0.0>. Noncommercial use,
changes and redistribution are allowed; commercial use is not.

Required Notice: Copyright 2026 thingg.co

The license covers this repository. The models, llama.cpp, vLLM and the
container image it downloads onto the Spark are separate works under their own
licenses.

## Open ends

- A window of 1M that is also exact. Qwen3.8-27B with a 16-bit cache passed at
  507k, but the cache that fits at that precision (950k tokens) has no
  headroom over a prompt that fills the window: vLLM read a 935k prompt to
  the end, ran out of blocks and started over from its last checkpoint. The
  profile therefore stops at 900k, and nothing above 507k has been checked.
  An FP8 cache, which NVIDIA's recipe for the checkpoint uses, would make room
  for the full 1.01M. Whether recall survives FP8 at that length is the next
  thing to measure.
- Laguna S 2.1. The installed vLLM supports it and the weights are on the
  Spark. What window its 93 GiB leaves, and whether it is exact there, has
  not been measured.
- Flash-Next with a 16-bit cache. At that precision only about 290k fits, so
  the stretched results above are all FP8. Whether the cache or the sparse
  attention is what loses its place past 256k is not settled.
- The llama.cpp profiles other than gpt-oss-20b have not been run, and their
  Hugging Face repo names are unchecked. `pull-model.sh` checks the repo
  before downloading and says which line to fix.
- Memory headroom. `MEM_UTIL` values are what worked on the night, with a
  desktop session and a download running. `spark/preflight.sh` and the
  control page show what is in use.
