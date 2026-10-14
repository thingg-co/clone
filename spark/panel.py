#!/usr/bin/env python3
"""Control page for the model server on the Spark. Standard library only.

    python3 -I spark/panel.py            # http://<spark>:8088

Shows what is running, switches profile and context length, starts and stops
the services, pulls weights, runs the checks, and prints the command that
points Claude Code or fable at this box.

Access: PANEL_HOST in config/spark.env decides where it listens. The shipped
config uses 0.0.0.0, so it is on the LAN; without the setting it stays on
loopback and is reached through the SSH tunnel. There is no login: anyone who
can reach the port can use the page, which shows the API key. Every action is
one of the fixed commands in build_action; arguments are checked against the
profile files before anything runs.
"""

import fnmatch
import gzip
import hashlib
import io
import ipaddress
import json
import os
import re
import shlex
import signal
import subprocess
import tarfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).with_name("panel.html")
KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
LOG_UNITS = ("vllm", "llama-server", "llama-sidekick", "clone-panel")


def read_env(path: Path) -> dict:
    """KEY=VALUE lines of a profile or spark.env, without quotes and trailing comments."""
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line.strip())
        if not m:
            continue
        v = m.group(2).strip()
        if v[:1] in ("'", '"'):
            end = v.rfind(v[0])
            v = v[1:end] if end > 0 else v[1:]
        else:
            v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
        out[m.group(1)] = v
    return out


def config() -> dict:
    env = read_env(ROOT / "config" / "spark.env")
    env.setdefault("BACKEND", "vllm")
    env.setdefault("PORT", "8080")
    env.setdefault("MODEL", "gpt-oss-120b")
    env.setdefault("VLLM_PROFILE", "qwen3.8-27b")
    env.setdefault("VLLM_LONG", "0")
    env.setdefault("MODELS_DIR", str(ROOT / "models"))
    return env


def to_int(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def hf_complete(repo_dir: Path) -> bool:
    """A snapshot with weights in it and no download still in flight."""
    if any((repo_dir / "blobs").glob("*.incomplete")):  # in flight, or cancelled part way
        return False
    return any(repo_dir.glob("snapshots/*/config.json")) and any(repo_dir.glob("snapshots/*/*.safetensors"))


# ---- fable, handed out as an installable archive ------------------------------
AGENT_DIR = ROOT / "agent"
_archive: dict = {"sig": None, "path": "", "data": b"", "version": ""}


def fable_archive() -> dict:
    """The agent as a source archive pip and uv can install from a URL. Rebuilt when a file changes.

    The path carries a hash of the content, so a client's cache is good until the code changes.
    """
    files = [AGENT_DIR / "pyproject.toml", AGENT_DIR / "README.md", *sorted((AGENT_DIR / "fable_local").rglob("*"))]
    files = [f for f in files if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc"]
    if not files or not files[0].is_file():
        return {"path": "", "data": b"", "version": ""}
    sig = tuple((str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in files)
    if sig != _archive["sig"]:
        m = re.search(r'^version\s*=\s*"([^"]+)"', files[0].read_text(), re.M)
        version = m.group(1) if m else "0"
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w") as tar:
            for f in files:
                data = f.read_bytes()
                info = tarfile.TarInfo(f"fable_local-{version}/{f.relative_to(AGENT_DIR)}")
                info.size, info.mode, info.mtime = len(data), 0o644, 0
                tar.addfile(info, io.BytesIO(data))
        data = gzip.compress(raw.getvalue(), mtime=0)
        _archive.update(sig=sig, data=data, version=version,
                        path=f"/dist/{hashlib.sha256(data).hexdigest()[:10]}/fable_local-{version}.tar.gz")
    return _archive


def hosts() -> list:
    """Names and addresses this machine answers to, for the commands on the page."""
    me = os.uname().nodename.split(".")[0]
    out = [{"host": f"{me}.local", "note": "LAN"}, {"host": me, "note": "Tailscale, or your own DNS"}]
    for ip in run(["hostname", "-I"]).split():
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if addr.version != 4 or ip.startswith("172.17."):   # skip IPv6 and the docker bridge
            continue
        out.append({"host": ip, "note": "Tailscale" if addr in ipaddress.ip_network("100.64.0.0/10") else "LAN"})
    return out


# ---- download progress --------------------------------------------------------
TOTALS: dict = {}    # (repo, include) -> bytes, or (None, time of the failed lookup)
SAMPLES: dict = {}   # profile id -> [(time, bytes)], for the rate


def repo_total(repo: str, include: str = "*"):
    """Size of what a pull fetches, from the Hugging Face file listing. None if it cannot be had."""
    k = (repo, include)
    hit = TOTALS.get(k)
    if isinstance(hit, int) or (hit and time.time() - hit[1] < 120):
        return hit if isinstance(hit, int) else None
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true", timeout=6) as r:
            items = json.load(r)
        pats = include.split() or ["*"]
        TOTALS[k] = sum(i.get("size", 0) for i in items
                        if i.get("type") == "file" and any(fnmatch.fnmatch(i["path"].split("/")[-1], p) for p in pats))
        return TOTALS[k]
    except Exception:
        TOTALS[k] = (None, time.time())
        return None


NET: list = []       # [(time, bytes received on all interfaces but loopback)]


def net_rate():
    """Bytes per second arriving over the network, averaged over the last minute. None off Linux."""
    try:
        rows = Path("/proc/net/dev").read_text().splitlines()[2:]
        total = sum(int(r.split(":")[1].split()[0]) for r in rows if not r.strip().startswith("lo:"))
    except (OSError, ValueError, IndexError):
        return None
    now = time.time()
    if not NET or now - NET[-1][0] >= 2:
        NET.append((now, total))
    del NET[:-40]
    old = next((n for n in NET if now - n[0] <= 60), NET[0])
    return (total - old[1]) / (now - old[0]) if now - old[0] >= 4 else None


def bytes_on_disk(files) -> int:
    n, seen = 0, set()
    for f in files:
        try:
            st = f.stat()
        except OSError:
            continue
        if (st.st_dev, st.st_ino) in seen:
            continue
        seen.add((st.st_dev, st.st_ino))
        n += min(st.st_size, st.st_blocks * 512) if f.name.endswith(".incomplete") else st.st_size
    return n


def downloads(profs: list, env: dict) -> list:
    """One entry per pull that is running or waiting to run."""
    ps = run(["ps", "-eo", "args"])
    models, out, now, net = Path(env["MODELS_DIR"]), [], time.time(), net_rate()
    for p in profs:
        script = "vllm.sh pull" if p["backend"] == "vllm" else "pull-model.sh"
        waiting = re.search(rf"{re.escape(script)}(?: [\w.-]+)* {re.escape(p['key'])}(?:\s|$)", ps, re.M) is not None
        if not (p["downloading"] or waiting):
            SAMPLES.pop(p["backend"] + "/" + p["key"], None)
            continue
        if p["backend"] == "vllm" and p.get("own_dir"):   # a runner that downloads into models/<key>
            d = models / p["key"]
            done = bytes_on_disk([*d.rglob("*.safetensors"), *d.rglob("*.incomplete")]) if d.is_dir() else 0
            total = repo_total(p["repo"])
        elif p["backend"] == "vllm":
            blobs = models / "hf-cache" / "hub" / ("models--" + p["repo"].replace("/", "--")) / "blobs"
            done = bytes_on_disk(blobs.iterdir()) if blobs.is_dir() else 0
            total = repo_total(p["repo"])
        else:
            d = models / p["dir"]
            done = bytes_on_disk([*d.rglob("*.gguf"), *d.rglob("*.incomplete")]) if d.is_dir() else 0
            total = repo_total(p["repo"], read_env(ROOT / "config" / "models" / (p["key"] + ".env")).get("HF_INCLUDE", "*"))
        hist = SAMPLES.setdefault(p["backend"] + "/" + p["key"], [])
        if not hist or now - hist[-1][0] >= 5:
            hist.append((now, done))
        del hist[:-150]
        # Bytes land in bursts, so average over up to ten minutes or the finish time jumps around.
        old = next((h for h in hist if now - h[0] <= 600), hist[0])
        rate = (done - old[1]) / (now - old[0]) if now - old[0] >= 30 else None
        queued = not p["downloading"] and done == 0
        # The downloader holds data in memory and writes files out in bursts, so bytes on disk
        # stand still for minutes at a time. The network counter shows what is arriving meanwhile.
        if not queued and net and (not rate or net > rate):
            rate = net / max(1, sum(1 for q in profs if q["downloading"]))  # the link is shared
        out.append({
            "key": p["key"], "backend": p["backend"], "done": done, "total": total, "rate": rate, "queued": queued,
            "eta": now + (total - done) / rate if rate and rate > 1e5 and total and total > done and not queued else None,
        })
    return out


def profiles(env: dict) -> list:
    out = []
    models = Path(env["MODELS_DIR"])
    ps = run(["ps", "-eo", "args"])  # leftover .incomplete files from a cancelled pull are not a download

    def fetching(repo: str) -> bool:
        return re.search(rf"hf download {re.escape(repo)}(\s|$)", ps, re.M) is not None
    for f in sorted((ROOT / "config" / "vllm").glob("*.env")):
        p = read_env(f)
        repo = p.get("HF_REPO", "")
        out.append({
            "backend": "vllm", "key": f.stem, "served": p.get("SERVED_NAME", f.stem), "repo": repo,
            "ctx": to_int(p.get("MAX_LEN"), 0), "long_ctx": to_int(p.get("LONG_MAX_LEN")),
            "native_ctx": to_int(p.get("ROPE_ORIG")), "runner": bool(p.get("RUNNER")), "own_dir": bool(p.get("PULL")),
            "thinking": "native" if p.get("REASONING_PARSER") else "prompted",
            "tools": "vLLM parser " + p["TOOL_PARSER"] if p.get("TOOL_PARSER") else "none", "min_free_gb": to_int(p.get("MIN_FREE_GB"), 0),
            "downloaded": (any((models / f.stem).rglob("*.safetensors")) if p.get("PULL")   # kept by its runner, not in the cache
                           else hf_complete(models / "hf-cache" / "hub" / ("models--" + repo.replace("/", "--")))),
            "downloading": fetching(repo),
        })
    for f in sorted((ROOT / "config" / "models").glob("*.env")):
        p = read_env(f)
        d = models / (p.get("MODEL_DIR") or f.stem)
        pattern = p.get("GGUF_MATCH", "").lower()
        out.append({
            "backend": "llama", "key": f.stem, "dir": d.name, "served": p.get("ALIAS", f.stem), "repo": p.get("HF_REPO", ""),
            "ctx": to_int(p.get("CTX"), 0), "long_ctx": None, "ctx_fit": p.get("CTX") == "fit",
            "thinking": "native" if re.search(r"gpt-oss|glm", f.stem) or "--reasoning on" in p.get("EXTRA_ARGS", "") else "prompted",
            "tools": "set by the model's template", "min_free_gb": to_int(p.get("MIN_FREE_GB"), 0),
            "downloaded": d.is_dir() and any(pattern in str(g).lower() for g in d.rglob("*.gguf")),
            "downloading": fetching(p.get("HF_REPO", "")),
        })
    return out


def run(cmd: list, timeout: float = 4) -> str:
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.TimeoutExpired):
        return ""


def unit_state(unit: str) -> str:
    return (run(["systemctl", "--user", "is-active", unit]).strip().splitlines() or ["unknown"])[-1]


def endpoint(port, key: str) -> dict:
    """What answers on a model port: the served ids and the context window."""
    info = {"port": to_int(port), "up": False, "models": [], "context": None}
    if not info["port"]:
        return info
    base = f"http://127.0.0.1:{info['port']}"
    headers = {"Authorization": f"Bearer {key}"}
    try:
        with urllib.request.urlopen(urllib.request.Request(base + "/v1/models", headers=headers), timeout=1.5) as r:
            data = json.load(r).get("data", [])
        info["up"] = True
        info["models"] = [m.get("id") for m in data if m.get("id")]
        info["context"] = next((int(m["max_model_len"]) for m in data if m.get("max_model_len")), None)
        if info["context"] is None:  # llama.cpp reports it elsewhere
            with urllib.request.urlopen(urllib.request.Request(base + "/props", headers=headers), timeout=1.5) as r:
                info["context"] = json.load(r).get("default_generation_settings", {}).get("n_ctx")
    except Exception:
        pass
    return info


def memory() -> dict:
    try:
        kb = dict(re.findall(r"^(\w+):\s+(\d+)", Path("/proc/meminfo").read_text(), flags=re.M))
        return {"total_gb": round(int(kb["MemTotal"]) / 1048576, 1), "available_gb": round(int(kb["MemAvailable"]) / 1048576, 1)}
    except (OSError, KeyError):
        return {}


def gpu() -> list:
    rows = run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader"]).strip().splitlines()
    return [r.strip() for r in rows if "," in r]


# ---- measured results -----------------------------------------------------------
def reach(recall: list) -> int:
    """The largest prompt that came back exact, with every smaller size exact as well."""
    best = 0
    for r in sorted(recall, key=lambda r: r.get("tokens", 0)):
        if not r.get("of") or r.get("right") != r.get("of"):
            break
        best = r["tokens"]
    return best


def results(published: Path | None = None, runs: Path | None = None) -> dict:
    """Every measured model, best first: how far it stays exact, then the share of codes it
    got right, then decode speed. docs/results.json holds the published set; each run of
    scripts/bench.py appends a line to run/results.jsonl, and the newest line for a label wins."""
    try:
        base = json.loads((published or ROOT / "docs" / "results.json").read_text())
    except (OSError, ValueError):
        base = {}
    by_label = {m["label"]: m for m in base.get("models", []) if m.get("label")}
    checkers = {m["label"]: m for m in base.get("checkers", []) if m.get("label")}
    try:
        lines = (runs or ROOT / "run" / "results.jsonl").read_text().splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if isinstance(m, dict) and m.get("kind") == "bench" and m.get("label"):
            by_label[m["label"]] = m
        elif isinstance(m, dict) and m.get("kind") == "checker" and m.get("label"):  # decision models, scripts/clef-test.py
            checkers[m["label"]] = m
    models = []
    for m in by_label.values():
        rec = [r for r in m.get("recall") or [] if isinstance(r, dict)]
        right, of = sum(r.get("right", 0) for r in rec), sum(r.get("of", 0) for r in rec)
        models.append({**m, "recall": sorted(rec, key=lambda r: r.get("tokens", 0)), "reach": reach(rec), "right": right, "of": of})
    models.sort(key=lambda m: (-m["reach"], -(m["right"] / m["of"] if m["of"] else 0), -(m.get("decode_tps") or 0)))
    return {"updated": base.get("updated"), "machine": base.get("machine"), "test": base.get("test"), "models": models,
            "checkers": list(checkers.values())}


def bake_state():
    """What spark/bake.sh is measuring, from run/bake.state: item, position, start time."""
    try:
        item, pos, since = (ROOT / "run" / "bake.state").read_text().split()[:3]
        return {"item": item, "position": pos, "since": to_int(since)}
    except (OSError, ValueError):
        return None


# ---- jobs: one fixed command at a time, output to logs/panel-job.log ----------
class Jobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.name = ""
        self.proc = None
        self.started = 0.0
        self.log = ROOT / "logs" / "panel-job.log"

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, name: str, script: str) -> None:
        with self.lock:
            if self.running():
                raise RuntimeError(f"busy: {self.name} is still running")
            self.log.parent.mkdir(parents=True, exist_ok=True)
            fh = open(self.log, "wb")
            fh.write(f"[{time.strftime('%H:%M:%S')}] {name}\n".encode())
            fh.flush()
            self.proc = subprocess.Popen(["bash", "-c", script], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            fh.close()
            self.name, self.started = name, time.time()

    def cancel(self) -> None:
        if self.running():
            os.killpg(self.proc.pid, signal.SIGTERM)

    def state(self) -> dict:
        tail = ""
        if self.log.is_file():
            with open(self.log, "rb") as fh:
                fh.seek(max(0, self.log.stat().st_size - 12000))
                tail = re.sub(r"\x1b\[[0-9;]*m", "", fh.read().decode("utf-8", "replace"))
        return {
            "name": self.name, "running": self.running(), "started": self.started,
            "exit_code": None if self.proc is None or self.running() else self.proc.returncode, "log": tail,
        }


JOBS = Jobs()
PRELUDE = ". spark/common.sh; load_env; "


def wait(port_var: str) -> str:
    return f'; log "waiting for :${port_var} (loading the weights takes a few minutes)"; wait_healthy "${port_var}" 1800 && log "healthy on :${port_var}" || warn "not healthy after 30 min"'


def build_action(body: dict, env: dict) -> tuple:
    """(job name, bash script) for a request, or ValueError. Every argument is checked here."""
    action = body.get("action")
    main_unit = "vllm.service" if env["BACKEND"] == "vllm" else "llama-server.service"
    known = {(p["backend"], p["key"]): p for p in profiles(env)}

    if action in ("start", "stop", "restart"):
        target = body.get("target", "main")
        if target not in ("main", "sidekick"):
            raise ValueError("target must be main or sidekick")
        if target == "sidekick" and not env.get("SIDEKICK_MODEL"):
            raise ValueError("no SIDEKICK_MODEL in config/spark.env")
        unit, port = (main_unit, "PORT") if target == "main" else ("llama-sidekick.service", "SIDEKICK_PORT")
        return f"{action} {target}", PRELUDE + f"systemctl --user {action} {unit}" + (wait(port) if action != "stop" else '; log "stopped"')

    if action in ("switch", "pull"):
        backend, key = body.get("backend"), str(body.get("profile", ""))
        if not KEY_RE.match(key) or (backend, key) not in known:
            raise ValueError(f"unknown profile {backend}/{key}")
        prof, q = known[(backend, key)], shlex.quote(key)
        if action == "pull":
            return f"pull {key}", f"spark/vllm.sh pull {q}" if backend == "vllm" else f"spark/pull-model.sh {q}"
        if not prof["downloaded"]:
            raise ValueError(f"{key} is not downloaded yet; pull it first")
        if backend == "llama":
            return f"switch to {key}", (
                "systemctl --user disable --now vllm.service 2>/dev/null || true; "
                "sed -i 's/^BACKEND=.*/BACKEND=llama/' config/spark.env; spark/service.sh install; "
                f"spark/service.sh switch {q}"
            )
        long = to_int(body.get("long", 0))
        if long is None or long < 0:
            raise ValueError("long must be 0, 1 or a token count")
        if long and not prof["long_ctx"]:
            raise ValueError(f"{key} has no extended context")
        if long > 1 and not prof["ctx"] < long <= prof["long_ctx"]:
            raise ValueError(f"context for {key} must be between {prof['ctx']} and {prof['long_ctx']}")
        return f"switch to {key}, long={long}", f"systemctl --user disable --now llama-server.service 2>/dev/null || true; spark/vllm.sh switch {q} {long}"

    if action in ("smoke", "messages"):
        target = body.get("target") or "main"
        if target not in ("main", "sidekick"):
            raise ValueError("target must be main or sidekick")
        port = env["PORT"] if target == "main" else env.get("SIDEKICK_PORT")
        live = endpoint(port, env.get("API_KEY", ""))
        if not live["up"]:
            why = "its weights may still be downloading, or it has not been started" if target == "main" else "start the sidekick first"
            raise ValueError(f"the {target} model is not serving on port {port} yet: {why}")
        port = int(port)

    if action == "smoke":
        return f"tool-calling check ({target})", PRELUDE + f'python3 -I scripts/smoke-test.py --base-url "http://127.0.0.1:{port}/v1" --api-key "$API_KEY"'

    if action == "messages":
        model = live["models"][0] if live["models"] else ""
        if not KEY_RE.match(model):
            raise ValueError("the server lists no usable model name")
        payload = json.dumps({"model": model, "max_tokens": 2048, "messages": [{"role": "user", "content": "Reply with the single word: ready"}]})
        return f"Claude endpoint check ({target})", PRELUDE + (
            f'curl -sS -m 600 -w "\\nHTTP %{{http_code}}\\n" -X POST "http://127.0.0.1:{port}/v1/messages" '
            '-H "Authorization: Bearer $API_KEY" -H "x-api-key: $API_KEY" -H "anthropic-version: 2023-06-01" -H "content-type: application/json" '
            f"-d {shlex.quote(payload)}"
        )
    raise ValueError(f"unknown action {action!r}")


def state() -> dict:
    env = config()
    key = env.get("API_KEY", "")
    main_unit = "vllm.service" if env["BACKEND"] == "vllm" else "llama-server.service"
    profs = profiles(env)
    return {
        "host": os.uname().nodename,
        "downloads": downloads(profs, env),
        "hosts": hosts(),
        "fable": {"archive": fable_archive()["path"], "version": fable_archive()["version"]},
        "config": {
            "backend": env["BACKEND"], "vllm_profile": env["VLLM_PROFILE"], "vllm_long": to_int(env["VLLM_LONG"], 0),
            "model": env["MODEL"], "port": to_int(env["PORT"]), "sidekick_model": env.get("SIDEKICK_MODEL", ""),
            "sidekick_port": to_int(env.get("SIDEKICK_PORT")), "api_key": key,
        },
        "profiles": profs,
        "main": {"unit": main_unit, "service": unit_state(main_unit), **endpoint(env["PORT"], key)},
        "sidekick": {"unit": "llama-sidekick.service", "service": unit_state("llama-sidekick.service"), **endpoint(env.get("SIDEKICK_PORT"), key)} if env.get("SIDEKICK_MODEL") else None,
        "memory": memory(),
        "gpu": gpu(),
        "job": JOBS.state(),
        "bake": bake_state(),
    }


_snap = {"t": 0.0, "state": None}
_snap_lock = threading.Lock()


def snapshot() -> dict:
    """state(), computed at most once a second however many pages are open."""
    with _snap_lock:
        if time.time() - _snap["t"] >= 0.9:
            _snap["state"], _snap["t"] = state(), time.time()
        return _snap["state"]


class Handler(BaseHTTPRequestHandler):
    server_version = "clone-panel"

    def log_message(self, fmt, *args):  # actions are logged by the job runner; requests are noise
        pass

    def send(self, code: int, body, ctype: str = "application/json", headers: dict | None = None) -> None:
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        for k, v in {"Cache-Control": "no-store", "X-Frame-Options": "DENY", **(headers or {})}.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def allowed(self) -> bool:
        """No login. The Host check only stops a hostile web page from rebinding its own name to this address."""
        host = (self.headers.get("Host") or "").strip().lower()
        host = (host.rsplit(":", 1)[0] if re.search(r":\d+$", host) else host).strip("[]")
        me = os.uname().nodename.lower().split(".")[0]
        if host in ("localhost", me) or host.startswith(me + ".") or re.fullmatch(r"[0-9a-f:.]+", host):
            return True
        self.send(403, {"error": f"open the panel by the machine's own name or address, not {host!r}"})
        return False

    def do_GET(self):
        if not self.allowed():
            return
        url = urlparse(self.path)
        if url.path == "/":
            self.send(200, PAGE.read_bytes(), "text/html")
        elif url.path == "/api/state":
            self.send(200, state())
        elif url.path == "/api/results":
            self.send(200, results())
        elif url.path.startswith("/dist/") and url.path.endswith(".tar.gz"):
            # whatever hash was asked for, the current build is what there is to give
            archive = fable_archive()
            if not archive["data"]:
                return self.send(404, {"error": "no agent/ directory in this tree"})
            self.send(200, archive["data"], "application/gzip", {"Cache-Control": "public, max-age=31536000, immutable"})
        elif url.path == "/api/events":  # server-sent events: the page stays current without polling
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b"data: " + json.dumps(snapshot()).encode() + b"\n\n")
                    self.wfile.flush()
                    time.sleep(1)
            except OSError:  # the browser went away
                pass
        elif url.path == "/api/logs":
            unit = (parse_qs(url.query).get("unit") or ["vllm"])[0]
            if unit not in LOG_UNITS:
                return self.send(400, {"error": "unknown unit"})
            text = run(["journalctl", "--user", "-u", f"{unit}.service", "-n", "300", "--no-pager", "-o", "cat"], timeout=8)
            self.send(200, {"unit": unit, "log": re.sub(r"\x1b\[[0-9;]*m", "", text)[-40000:] or "(no log lines)"})
        else:
            self.send(404, {"error": "not found"})

    def do_POST(self):
        if not self.allowed():
            return
        # A page on another site cannot set this header, so it cannot press the buttons.
        if self.headers.get("X-Clone-Panel") != "1":
            return self.send(403, {"error": "missing X-Clone-Panel header"})
        try:
            body = json.loads(self.rfile.read(min(to_int(self.headers.get("Content-Length"), 0), 65536)) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            if self.path == "/api/cancel":
                JOBS.cancel()
                return self.send(200, {"ok": True})
            if self.path != "/api/action":
                return self.send(404, {"error": "not found"})
            name, script = build_action(body, config())
            JOBS.start(name, script)
            self.send(200, {"ok": True, "job": name})
        except (ValueError, RuntimeError) as e:
            self.send(409 if isinstance(e, RuntimeError) else 400, {"error": str(e)})


def main() -> None:
    env = config()
    host, port = env.get("PANEL_HOST", "127.0.0.1"), to_int(env.get("PANEL_PORT"), 8088)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    print(f"clone panel on http://{host}:{port} (root {ROOT})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
