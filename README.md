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
Measured here: 44 to 49 tok/s for a 125B model with 6B active, 12 tok/s for a
dense 27B.

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
lines make it a harsh test. It is also the property a long window is for.

`scripts/bench.py` wraps it. It measures decode speed on a short prompt,
checks that the model thinks, calls a tool and uses the result, and then runs
the recall test at 100k, 250k, 500k and 1M, or as far as the window goes.
`spark/bake.sh` does that for a list of profiles in turn, downloading and
switching as it goes, and appends a line per model to `run/results.jsonl`.
The control page ranks everything measured so far. `scripts/results.py
--publish` folds the runs into `docs/results.json` and rewrites the tables
here and on the project page.

The ranking puts first the largest prompt answered exactly with every smaller
one exact as well, then the share of codes right, then decode speed. All on
this Spark, with thinking on where the model has it:

<!-- results:start -->
| | model | window served | exact up to | codes right, by prompt size | thinks, tools | decode | read-in |
|---|---|---|---|---|---|---|---|
| 1 | Qwen3.8-27B, FP8 cache (dense 27B, NVFP4, 22 GB) | 1.01M, vLLM, stretched from 256k, FP8 cache | 519k | 93k 3/3, 253k 3/3, 519k 3/3, 986k 2/3 | yes | 12 tok/s | 918 tok/s |
| 2 | Qwen3.8-27B, 16-bit cache (dense 27B, NVFP4, 22 GB) | 950k, stretched from 256k, 16-bit cache | 507k | 97k 3/3, 245k 3/3, 507k 3/3 | yes | 12 tok/s | 920 tok/s |
| 3 | DeepSeek V4 Flash (284B, 13B active, 3-bit GGUF, 104 GB) | 991k, llama.cpp, 16-bit cache | 255k | 61k 3/3, 255k 3/3, 502k 0/3, 970k 2/3 | yes | 16 tok/s | 300 tok/s |
| 4 | Qwen3.6-35B-A3B (35B, 3B active, NVFP4, 23 GB) | 1.01M, vLLM, stretched from 256k, 16-bit cache | 253k | 93k 3/3, 253k 3/3, 519k 2/3, 1M 1/3 | yes | 77 tok/s | 2,897 tok/s |
| 5 | Qwen3.8-Flash-Next (125B, 6B active, NVFP4, 133 GB) | 524k, stretched from 256k, FP8 cache; past 256k it returns the neighbouring record | 253k | 93k 3/3, 253k 3/3, 411k 2/3, 505k 2/3, 516k 2/3 | yes | 44 tok/s | 2,650 tok/s |
| 6 | Nemotron 3.5 Lightning (30B, 3B active, NVFP4, 22 GB) | 1.05M, native window, FP8 cache | 102k | 102k 3/3, 232k 1/3, 482k 1/3, 986k 1/3 | yes | 136 tok/s | 1,290 tok/s |
| 7 | Laguna S 2.1 (118B, 8B active, NVFP4, 100 GB) | 500k, trained to 1M, 500k fits here, FP8 cache; thinks on some requests only | none | 93k 2/3, 253k 2/3, 411k 1/3 | tool loop only | 12 tok/s | 1,520 tok/s |
| 8 | Nemotron 3 Super (120B, 12B active, NVFP4, 80 GB) | 1M, native window, FP8 cache | none | 93k 2/3, 235k 1/3, 489k 1/3 | yes | 16 tok/s | 1,045 tok/s |
<!-- results:end -->

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
  at 507k with a factor of 4 and a 16-bit cache, and at 519k with an FP8
  cache. At 986k with the FP8 cache it got two of three: asked for locker
  34548 it gave locker 34458's code, two digits swapped, after 46 minutes of
  reading.
- Qwen3.8-Flash-Next attends sparsely and loses its place past the trained
  length. A factor of 2 with a 524k window does better than a factor of 4 with
  558k (2/3 against 0/3 at about 517k). Neither is exact.
- Qwen3.6-35B-A3B, small and fast at 77 tok/s, stayed exact at 253k with a
  factor of 4, then got 2/3 at 519k and 1/3 at 1M.
- Qwen3-Coder-Next, an older model whose card only claims 256k, stopped
  calling tools altogether at a factor of 4. It is not in the table.

DeepSeek V4 Flash, GLM-5.3-Flash and Kimi Linear are trained to 1M, and most
of their layers keep a compressed or linear cache, so a full window costs a
few GiB. The first two only fit this box at 2 to 3 bits per weight, through
llama.cpp, which is a harsher cut than the 4-bit builds the rest run at.
DeepSeek at 3 bits was exact to 255k. At 502k it wrote down the wrong code
for the middle locker while thinking, started to doubt its lookups and never
called the tool. At 970k it got two of three, giving locker 26499's code for
locker 26995, after 72 minutes of reading.

Long prompts cost time. Flash-Next reads about 2,500 tok/s, so a cold 500k
prompt takes under four minutes. The dense 27B reads 900 tok/s for the first
100k and under 300 past 250k, and the same prompt takes 22 minutes. The
server caches what it has read, and a request that extends the previous
prompt only pays for the new part. For these hybrid models the cache resumes
from the end of an earlier prompt, so anything that changes the start of the
conversation, or shortens it, means reading it all again. The agent keeps its
system prompt fixed for that reason.

## Profiles

A profile in `config/vllm/` names the weights, the window, the parsers and the
memory share. `VLLM_PROFILE` and `VLLM_LONG` in `config/spark.env` pick one.

| profile | what it is | windows |
|---|---|---|
| `qwen3.8-27b` | dense 27B, exact past 500k, 16-bit cache | 256k native; `VLLM_LONG=1` or a token count stretches it to 900k |
| `qwen3.8-27b-fp8` | the same model with an FP8 cache, which makes room for a full 1.01M | 256k native, 1.01M stretched |
| `qwen3.8-flash-next` | fastest and strongest coder, served through a patched vLLM (below) | `VLLM_LONG=0` asks for 500k, `1` for the most that fits |
| `qwen3.6-35b-a3b` | 35B with 3B active, small and fast | 256k native, 1.01M stretched |
| `qwen3.5-122b`, `qwen3.5-122b-fp8` | 122B with 10B active, with a 16-bit or an FP8 cache | 256k native, 500k or 1.01M stretched |
| `deepseek-v4-flash-spark` | DeepSeek V4 Flash in 0xSero's single-Spark build: pruned experts, 3-bit EXL3, DSpark draft, through a patched vLLM (below) | 262k at most |
| `laguna-s-2.1` | Poolside's 118B coding model, trained to 1M | what its 93 GiB of weights leave room for |
| `nemotron-3-super`, `nemotron-3.5-lightning` | native 1M windows that fail the recall test | 1M |
| `gpt-oss-120b` | 128k | 128k |

llama.cpp profiles in `config/models/`:

| profile | what it is | windows |
|---|---|---|
| `deepseek-v4-flash` | DeepSeek V4 Flash, 284B with 13B active, 3-bit GGUF | trained to 1M; the largest that fits |
| `deepseek-v4-flash-q2`, `deepseek-v4-flash-dspark` | the same at 2 bits, alone and with DeepSeek's DSpark draft model for faster decoding | the largest that fits |
| `glm-5.3-flash` | GLM-5.3-Flash, 320B with 18B active, 2-bit GGUF | trained to 1M; the largest that fits |
| `kimi-linear-48b` | Kimi Linear 48B with 3B active, 8-bit, no thinking mode | trained to 1M; the largest that fits |
| `seed-oss-36b` | Seed-OSS-36B, dense, 8-bit weights and cache | trained to 512k; the largest that fits |
| `clef`, `clef-flash` | Cloudflare's decision models: typed answers at `/v1/systemone`, no text | passages up to 128k |
| `gpt-oss-120b`, `gpt-oss-20b` | OpenAI's open models | 128k |

`CTX=fit` in a llama.cpp profile leaves the window to llama.cpp: it starts
from the window the model was trained for and shrinks it until weights and
cache fit, keeping `FIT_MARGIN_MB` free for the rest of the machine. The
window it settled on is in the server log and on the control page.

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
- DeepSeek V4 Flash through `spark/deepseek-spark.sh`, which runs
  [0xSero's build](https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer)
  for one Spark (Apache-2.0, weights MIT) from its image, pinned by digest.
  The image would download and repack the weights at its first start; here
  `vllm.sh pull` fetches them and the bake-off repacks them right before the
  model's turn, so a start only loads. A profile can name such steps with
  `PULL` and `PREPARE`.
- llama.cpp built from source (`spark/serve.sh`, GGUF profiles in
  `config/models/`). This is how models whose release weights are larger
  than the machine run here at all, at 2 to 3 bits per weight. It reads each
  model's own chat template, so tool calls need no parser setting, and it
  can run a small sidekick model on a second port when the main profile
  leaves memory for one.

## Layout

    config/spark.env.example   hostname, backend, model, port, API key (copy to spark.env)
    config/vllm/*.env          vLLM profiles: HF repo, window, parsers, memory share
    config/models/*.env        llama.cpp profiles: GGUF repo, context, KV type
    spark/bootstrap.sh         one-shot setup on the Spark; idempotent; no sudo
    spark/build-llama.sh       llama.cpp with CUDA for sm_121, into llama.cpp/
    spark/pull-model.sh        GGUF download into models/<key>/ (checks the repo first)
    spark/vllm.sh              pull | serve | switch | install | stop | logs | image
    spark/flash-next.sh        Qwen3.8-Flash-Next through the pinned third-party recipe
    spark/deepseek-spark.sh    DeepSeek V4 Flash through 0xSero's single-Spark build
    spark/serve.sh             llama-server for a profile (main, sidekick, or any key)
    spark/service.sh           systemd user units for the llama.cpp instances
    spark/panel.py             control page: status, profile and context switch, connection strings
    spark/preflight.sh         memory, GPU processes, ports, does the profile fit
    spark/bake.sh              measure a list of profiles in turn: download, serve, scripts/bench.py
    spark/bench.sh             llama-bench at 0 and 8k depth
    spark/ollama.sh            fallback: Ollama from the tarball, in-tree
    scripts/deploy.sh          rsync this tree to the Spark (and run bootstrap); leaves its spark.env alone
    scripts/tunnel.sh          ssh -L for the model ports and the control page
    scripts/status.sh          what the Spark is doing
    scripts/smoke-test.py      tool-calling check, standard library only
    scripts/context-test.py    recall across the window through a tool call; the acceptance test
    scripts/bench.py           speed, thinking, tool loop and the recall test, one JSON line per model
    scripts/results.py         rank the results; --publish writes docs/results.json and the tables
    scripts/clef-test.py       the recall test for a decision model: multiple choice and claims to check
    bin/clone                  fable in the current directory, pointed at the server; tmux over SSH
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
    uvx --from "fable-local @ http://omega.local:8088/dist/<hash>/fable_local-0.2.8.tar.gz" fable

To run it on the Spark itself: `spark/bootstrap.sh --agent`, then
`agent/.venv/bin/fable`.

## The clone command

`bin/clone` starts fable in the current directory, already pointed at the
model server. `spark/bootstrap.sh` links it into `~/.local/bin` on the Spark,
and `make clone-command` does the same on a Mac, where it talks to the Spark
over the network:

    ssh omega.local
    cd ~/Development/steel
    clone                             # or clone -c to resume the last session here

Run in your home folder, it lists `~/Development`, most recently changed
first, and starts in the one you pick. Paste a git URL at that prompt, or run
`clone <git url>` anywhere, and it clones the repository into `~/Development`
first (or opens the folder if it is already there). Over SSH it runs inside tmux, one
session per directory, so a dropped connection or a closed laptop leaves the
agent working; `clone` in the same folder attaches to it again, and Ctrl-b d
leaves it running on purpose. Quitting the agent leaves a shell in the
project. Any fable option passes through (`clone --effort max`, `clone
--doctor`). `CLONE_TMUX=0` turns tmux off.

On the Spark the agent works on the Spark's copy of a project, so a project
has to be cloned or synced there first. To make plain `ssh fable` land in
the list, add this to `~/.ssh/config` on the Mac:

    Host fable
        HostName omega.local
        RequestTTY yes
        ForwardAgent yes
        RemoteCommand bash -lc clone

ForwardAgent lets git inside the session push with the Mac's keys; the
session keeps a stable link to whichever connection is current.

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
table. The Reliability table ranks every model measured so far by how far it
stays exact, with the codes it got right at each prompt size and its decode and
read-in speed, and says which model a running bake-off is on. The Connect box prints the full command for Claude Code and for fable
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

- A window of 1M that is also exact. Qwen3.8-27B with an FP8 cache is exact
  at 519k and got two of three at 986k. With a 16-bit cache that model passed
  at 507k; the cache that fits at that precision (950k tokens) has no headroom
  over a prompt that fills the window, so that profile stops at 900k, and its
  900k step is queued.
- Running the bake-off: Qwen3.5-122B (500k with a 16-bit cache, 1M with
  FP8), DeepSeek V4 Flash (2-bit, 2-bit with its DSpark draft, and 0xSero's
  single-Spark build), GLM-5.3-Flash, Kimi Linear and Seed-OSS-36B are queued
  on the Spark, then the models measured before, Flash-Next included, again
  with `scripts/bench.py`. Clef and Clef-Flash get `scripts/clef-test.py`, the
  same records as multiple choice and as claims to check.
  Larger models from the same labs (DeepSeek V4.1 Flash at 510 GB, MiniMax
  M3, Kimi K3, GLM-5.3) have no build that fits one Spark, and MiMo-V2.6-Flash
  only fits pruned or at under 2.5 bits.
- Flash-Next with a 16-bit cache. At that precision only about 290k fits, so
  the stretched results above are all FP8. Whether the cache or the sparse
  attention is what loses its place past 256k is not settled.
- A checker next to the main model. Cloudflare's Clef models answer typed
  questions (yes or no, pick one, a score) about a passage in one pass,
  without generating text, and llama.cpp serves them at `/v1/systemone`. The
  27B one is good at spotting claims a passage does not support, which is the
  failure that matters here. `scripts/clef-test.py` asks it to pick each code
  out of eight that include the neighbouring records' codes, and to judge
  true claims and off-by-one ones.
- The older llama.cpp profiles (`glm-4.5-air`, `gpt-oss-120b`, `qwen3-235b`)
  have not been run. `pull-model.sh` checks the repo before downloading and
  says which line to fix.
- Memory headroom. `MEM_UTIL` values are what worked on the night, with a
  desktop session and a download running. `spark/preflight.sh` and the
  control page show what is in use.
