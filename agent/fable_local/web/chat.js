// The chat page for fable --web (server.py). The session files are the record: the page draws
// them, and while a turn runs it draws the worker's events on top (streamed text and thinking,
// tool calls starting, permission questions), each record replacing its live version.
"use strict";

const $ = (id) => document.getElementById(id);
const COMPACT_NOTE = "[context compacted] Summary of the session so far:\n\n";
const COMPACT_ACK = "Understood. Continuing from the summary.";
const COMPACT_GO = "Continue the task described in the summary from its next step.";
const SHOWN = 20000;  // characters of one tool result drawn on the page
const MODES = { ask: "Ask before changes", auto: "Auto-approve", readonly: "Read-only" };

const S = {
  state: null, chats: [], id: null, view: null, es: null, running: false, boot: null,
  prompts: [], recall: -1, draft: "", holder: null, size: null, adopt: false, started: 0, lastSent: "",
};
const prefs = (() => { try { return JSON.parse(localStorage.getItem("fable.prefs") || "{}") || {}; } catch (e) { return {}; } })();
function savePrefs() { try { localStorage.setItem("fable.prefs", JSON.stringify(prefs)); } catch (e) { /* private window */ } }

// ---- small helpers --------------------------------------------------------------
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;  // only ever Markdown.render output
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids) if (kid != null && kid !== false) e.append(kid);
  return e;
}

async function api(path, body) {
  const init = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json", "X-Fable": "1" }, body: JSON.stringify(body),
  };
  const r = await fetch(path, init);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.status + " " + r.statusText);
  return data;
}

function textOf(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((p) => (p && p.text) || "").join("\n");
  return "";
}
function splitThink(text) {
  const m = text.match(/^\s*<think>([\s\S]*?)(<\/think>\s*|$)/);
  return m ? [m[1], text.slice(m[0].length)] : ["", text];
}
function base(path) { const p = String(path || "").replace(/\/+$/, ""); return p.slice(p.lastIndexOf("/") + 1) || p; }
function tilde(path) {
  const h = S.state && S.state.home;
  return h && path && (path === h || path.startsWith(h + "/")) ? "~" + path.slice(h.length) : path || "";
}
function tokens(n) { return n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 0 : 2).replace(/\.?0+$/, "") + "M" : n >= 1000 ? (n / 1000).toFixed(n >= 1e5 ? 0 : 1) + "k" : String(n); }
function took(ms) { const s = Math.round(ms / 1000); return s < 60 ? s + "s" : Math.floor(s / 60) + "m " + (s % 60) + "s"; }
function atBottom() { const sc = $("scroll"); return sc.scrollHeight - sc.scrollTop - sc.clientHeight < 80; }
function toBottom() { const sc = $("scroll"); sc.scrollTop = sc.scrollHeight; }

function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
  // plain http on the LAN is not a secure context, so no clipboard API
  const ta = el("textarea", { style: "position:fixed;opacity:0" });
  ta.value = text;
  document.body.append(ta);
  ta.select();
  document.execCommand("copy");
  ta.remove();
  return Promise.resolve();
}

function outPre(text, diff) {
  const shown = text.length > SHOWN ? text.slice(0, SHOWN) + "\n… " + (text.length - SHOWN) + " more characters" : text;
  const pre = el("pre", { class: "out" });
  if (!diff) { pre.textContent = shown; return pre; }
  for (const line of shown.split("\n")) {
    const cls = /^(\+\+\+|---)/.test(line) ? "dh" : line[0] === "+" ? "da" : line[0] === "-" ? "dd" : line.startsWith("@@") ? "dh" : null;
    pre.append(cls ? el("span", { class: cls }, line) : line, "\n");
  }
  return pre;
}

// ---- one assistant message ----------------------------------------------------------
class Step {
  constructor(parent) {
    this.el = el("div", { class: "step" });
    parent.append(this.el);
    this.reasoning = ""; this.text = "";
    this.thinkEl = null; this.textEl = null; this.toolsEl = null;
    this.t0 = 0; this.t1 = 0; this.done = false; this.queued = false;
  }
  addReasoning(s) { if (!this.t0) this.t0 = Date.now(); this.reasoning += s; this.schedule(); }
  addText(s) { if (this.t0 && !this.t1) this.t1 = Date.now(); this.text += s; this.schedule(); }
  finish(reasoning, text) {
    if (this.t0 && !this.t1) this.t1 = Date.now();
    if (reasoning) this.reasoning = reasoning;
    this.text = text;
    this.done = true;
    this.draw();
  }
  tools() {
    if (!this.toolsEl) { this.toolsEl = el("div", { class: "tools" }); this.el.append(this.toolsEl); }
    return this.toolsEl;
  }
  schedule() {
    if (this.queued) return;
    this.queued = true;
    requestAnimationFrame(() => { this.queued = false; this.draw(); });
  }
  draw() {
    const pinned = atBottom();
    if (this.reasoning.trim()) {
      if (!this.thinkEl) {
        this.thinkEl = el("details", { class: "think" }, el("summary"), el("div", { class: "md" }));
        // drawn only while open: long reasoning is costly to lay out on every token
        this.thinkEl.addEventListener("toggle", () => { if (this.thinkEl.open) this.thinkEl.lastChild.innerHTML = Markdown.render(this.reasoning); });
        this.el.prepend(this.thinkEl);
      }
      const live = !this.done && !this.t1;
      this.thinkEl.classList.toggle("live", live);
      this.thinkEl.firstChild.textContent = live ? "Thinking…" : this.t0 && this.t1 ? "Thought for " + took(this.t1 - this.t0) : "Thinking";
      if (this.thinkEl.open) this.thinkEl.lastChild.innerHTML = Markdown.render(this.reasoning);
    }
    if (this.text.trim()) {
      if (!this.textEl) {
        this.textEl = el("div", { class: "md answer" });
        if (this.toolsEl) this.el.insertBefore(this.textEl, this.toolsEl); else this.el.append(this.textEl);
      }
      this.textEl.innerHTML = Markdown.render(this.text);
    }
    if (pinned) toBottom();
  }
}

// ---- one tool call -------------------------------------------------------------------
class Call {
  constructor(parent, tc) {
    const fn = tc.function || {};
    this.id = tc.id;
    this.name = fn.name || "?";
    try { this.args = JSON.parse(fn.arguments || "{}") || {}; } catch (e) { this.args = {}; }
    this.started = false;
    if (this.name === "todo_write") {  // the plan stays in view
      this.el = el("div", { class: "todos" }, todoList(this.args.todos));
      this.body = this.el;
      parent.append(this.el);
      return;
    }
    this.latest = el("span", { class: "latest" });
    this.el = el("details", { class: "tool queued" },
      el("summary", {}, el("span", { class: "ico" }), el("span", { class: "name" }, this.name), " ",
        el("span", { class: "sum" }, tc.summary || ""), this.latest),
      this.body = el("div", { class: "tbody" }));
    const a = this.args;
    if (this.name === "bash" && a.command) this.body.append(el("pre", { class: "args" }, "$ " + a.command));
    else if (this.name === "write_file" && typeof a.content === "string") this.body.append(outPre(a.content, false));
    else if (this.name === "spawn_subagent" && a.task) this.body.append(el("div", { class: "md args", html: Markdown.render(a.task) }));
    else if (!["read_file", "edit_file", "list_dir", "glob_files", "grep_files", "web_fetch"].includes(this.name) && Object.keys(a).length)
      this.body.append(el("pre", { class: "args" }, JSON.stringify(a, null, 2)));
    parent.append(this.el);
  }
  start() { this.started = true; this.el.classList.remove("queued"); this.el.classList.add("running"); }
  sub(name, summary) {
    this.body.append(el("div", { class: "subline" }, el("span", { class: "name" }, name), " ", summary || ""));
    this.latest.textContent = " › " + name + " " + (summary || "");
  }
  subResult(error, text) {
    if (error && text) this.body.append(el("div", { class: "subline err" }, text.split("\n")[0].slice(0, 300)));
  }
  note(level, text) { this.body.append(el("div", { class: "subline " + (level === "error" ? "err" : "") }, text)); }
  display(text) {
    if (this.name === "todo_write") return;
    this.body.append(outPre(text, false));
  }
  done(content) {
    this.started = true;
    if (this.name === "todo_write") return;
    const error = content.startsWith("error");
    this.el.classList.remove("queued", "running");
    this.el.classList.add(error ? "error" : "ok");
    if (this.latest) this.latest.textContent = "";
    if (this.name === "spawn_subagent" && !error) this.body.append(el("div", { class: "md report", html: Markdown.render(content) }));
    else this.body.append(outPre(content, this.name === "edit_file" && !error));
    if (error) this.el.title = content.split("\n")[0].slice(0, 200);
  }
}

function todoList(todos) {
  const mark = { done: "☑", completed: "☑", in_progress: "◐" };
  const ul = el("ul");
  for (const t of Array.isArray(todos) ? todos : []) {
    const st = String(t.status || "pending");
    ul.append(el("li", { class: st }, el("span", { class: "box" }, mark[st] || "☐"), " ", String(t.content || "")));
  }
  return ul;
}

// ---- the conversation ------------------------------------------------------------------
class View {
  constructor(root) { this.root = root; this.reset(); }
  reset() {
    this.root.textContent = "";
    this.calls = new Map();
    this.queue = [];
    this.running = null;
    this.reply = null;
    this.live = null;
    this.divider = null;
    this.pending = null;
    this.cards = new Map();
  }
  ensureReply() {
    if (!this.reply) { this.reply = el("div", { class: "turn assistant" }); this.root.append(this.reply); }
    return this.reply;
  }
  ensureLive() { if (!this.live) this.live = new Step(this.ensureReply()); return this.live; }
  user(text, pending) {
    this.reply = null; this.live = null;
    const turn = el("div", { class: "turn user" + (pending ? " pending" : "") }, el("div", { class: "bubble" }, text));
    this.root.append(turn);
    if (pending) this.pending = turn;
    return turn;
  }
  note(level, text, depth) {
    if (depth && this.running) return this.running.note(level, text);
    (this.reply || this.root).append(el("div", { class: "note " + (level || "info") }, text));
    if (atBottom()) toBottom();
  }
  record(r) {
    if (r.type === "compacted") {
      this.divider = el("details", { class: "compact" }, el("summary", {}, "Context compacted"), el("div", { class: "md" }));
      this.root.append(this.divider);
      this.reply = null; this.live = null;
      return;
    }
    if (r.type !== "message") return;
    const m = r.message || {};
    const text = textOf(m.content);
    if (m.role === "user") {
      if (text.startsWith(COMPACT_NOTE)) {
        if (this.divider) this.divider.lastChild.innerHTML = Markdown.render(text.slice(COMPACT_NOTE.length));
        return;
      }
      if (text === COMPACT_GO) return;
      if (this.pending) { this.pending.remove(); this.pending = null; }
      this.user(text);
      return;
    }
    if (m.role === "assistant") {
      if (text === COMPACT_ACK) return;
      if (text === "[interrupted]") { this.live = null; this.note("warn", "Stopped"); return; }
      const [inner, answer] = splitThink(text);
      const step = this.live || new Step(this.ensureReply());
      this.live = null;
      step.finish(m.reasoning_content || r.reasoning || inner, answer);
      if (answer.trim() && !(m.tool_calls || []).length) {
        const b = el("button", { type: "button", class: "copy-md", title: "Copy the answer as Markdown" }, "Copy");
        b.onclick = () => copyText(answer.trim()).then(() => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1500); });
        step.el.append(el("div", { class: "msg-acts" }, b));
      }
      for (const tc of m.tool_calls || []) {
        const call = new Call(step.tools(), tc);
        this.calls.set(tc.id, call);
        this.queue.push(call);
      }
      return;
    }
    if (m.role === "tool") {
      const call = this.calls.get(m.tool_call_id);
      if (!call) return;
      call.done(text);
      this.queue = this.queue.filter((c) => c !== call);
      if (this.running === call) this.running = null;
    }
  }
  event(ev) {
    switch (ev.type) {
      case "record": this.record(ev.record); break;
      case "text": this.ensureLive().addText(ev.s); break;
      case "reasoning": this.ensureLive().addReasoning(ev.s); break;
      case "tool":
        if (ev.depth) { if (this.running) this.running.sub(ev.name, ev.summary); }
        else {
          const call = this.queue.find((c) => c.name === ev.name && !c.started);
          if (call) { call.start(); this.running = call; }
        }
        break;
      case "tool_result":
        if (!this.running || ev.display == null) break;
        if (ev.depth) this.running.subResult(ev.error, ev.display);
        else this.running.display(ev.display);
        break;
      case "permission": this.permission(ev); break;
      case "answered": this.answered(ev.id); break;
      case "note": this.note(ev.level, ev.text, ev.depth); break;
    }
  }
  permission(ev) {
    const answer = async (a, card) => {
      for (const b of card.querySelectorAll("button")) b.disabled = true;
      try { await api("/api/chats/" + encodeURIComponent(S.id) + "/answer", { id: ev.id, answer: a }); }
      catch (e) { card.remove(); this.note("error", e.message); }
    };
    const diff = ev.name === "edit_file";
    const card = el("div", { class: "perm" },
      el("div", { class: "perm-head" }, el("b", {}, ev.name), ev.depth ? " (in the subagent)" : "", " wants to run"),
      outPre(ev.detail || ev.summary || "", diff),
      el("div", { class: "perm-acts" },
        el("button", { type: "button", class: "primary", onclick: () => answer("y", card) }, "Allow"),
        el("button", { type: "button", onclick: () => answer("a", card) }, "Always allow " + ev.name),
        el("button", { type: "button", onclick: () => answer("n", card) }, "Deny")));
    this.cards.set(ev.id, card);
    this.root.append(card);
    toBottom();  // no focus on Allow: an Enter meant for the message box must not approve a command
  }
  answered(id) {
    const card = this.cards.get(id);
    if (card) { card.remove(); this.cards.delete(id); }
  }
}

// ---- the list of chats ----------------------------------------------------------------
async function refreshList() {
  try { S.chats = (await api("/api/chats")).chats; } catch (e) { return; }
  renderList();
  const mine = S.chats.find((c) => c.id === S.id);
  if (!mine) return;
  header();
  if (S.running || S.adopt) {
    S.size = mine.size;  // our own turn wrote it
    S.adopt = false;
  } else if (S.size !== null && mine.size !== S.size) {
    openChat(S.id, true);  // the terminal added to it; its messages show up here as they are saved
    return;
  }
  S.holder = mine.holder;
  showHolder();
}

function renderList() {
  const nav = $("list");
  nav.textContent = "";
  const q = $("filter").value.trim().toLowerCase();
  const day = new Date(); day.setHours(0, 0, 0, 0);
  const today = day.getTime() / 1000;
  const group = (t) => t >= today ? "Today" : t >= today - 86400 ? "Yesterday" : t >= today - 7 * 86400 ? "Previous 7 days" : t >= today - 30 * 86400 ? "Previous 30 days" : "Older";
  let current = null;
  for (const c of S.chats) {
    if (q && !(c.title + " " + c.cwd).toLowerCase().includes(q)) continue;
    const g = group(c.updated);
    if (g !== current) { current = g; nav.append(el("h3", {}, g)); }
    const busy = c.running ? "working" : c.holder ? "running in " + c.holder : null;
    nav.append(el("a", { href: "#" + c.id, class: "item" + (c.id === S.id ? " active" : ""), title: c.title + "\n" + tilde(c.cwd) },
      el("span", { class: "t" }, busy ? el("i", { class: "dot", title: busy }) : null, c.title),
      el("span", { class: "p" }, base(c.cwd)),
      el("span", { class: "acts" },
        el("button", { type: "button", title: "Rename", onclick: (e) => { e.preventDefault(); rename(c); } }, "✎"),
        el("button", { type: "button", title: "Delete", onclick: (e) => { e.preventDefault(); remove(c); } }, "✕"))));
  }
  if (!nav.childElementCount) nav.append(el("p", { class: "empty" }, q ? "Nothing matches." : "No chats yet."));
}

async function rename(c) {
  const title = prompt("Name for this chat", c.title);
  if (title === null) return;
  try { await api("/api/chats/" + encodeURIComponent(c.id) + "/rename", { title }); } catch (e) { alert(e.message); }
  await refreshList();
  header();
}

async function remove(c) {
  if (!confirm("Delete “" + c.title + "”? It moves to sessions/trash on the server.")) return;
  try { await api("/api/chats/" + encodeURIComponent(c.id) + "/delete", {}); } catch (e) { alert(e.message); return; }
  if (c.id === S.id) location.hash = "";
  await refreshList();
}

// ---- the open chat ---------------------------------------------------------------------
function closeStream() { if (S.es) { S.es.close(); S.es = null; } }

async function openChat(id, quiet) {
  closeStream();
  S.id = id || null;
  S.holder = null; S.size = null;
  if (!quiet) S.view = new View($("log"));
  renderList();
  header();
  $("newchat").hidden = !!id;
  $("empty").hidden = !!id;
  if (!id) { setRunning(false); $("usage").textContent = ""; menus(null); projectPicker(); $("input").focus(); return; }
  let data;
  try { data = await api("/api/chats/" + encodeURIComponent(id)); }
  catch (e) { S.view.reset(); S.view.note("error", e.message); return; }
  if (S.id !== id) return;
  const pinned = !quiet || atBottom();
  const view = new View($("log"));
  S.view = view;
  S.boot = data.boot;
  for (const r of data.records) view.record(r);
  for (const ev of data.events) handle(ev, true);
  S.started = data.started ? data.started * 1000 : 0;
  setRunning(data.running);
  const st = data.state || {};
  if (st.window) usage(st.used, st.window);
  else $("usage").textContent = "";
  menus(st);
  const mine = S.chats.find((c) => c.id === id);
  S.size = mine ? mine.size : -1;
  S.holder = data.holder;
  showHolder();
  header();
  stream(id, data.seq);
  if (pinned) toBottom();
}

function stream(id, after) {
  const es = new EventSource("/api/chats/" + encodeURIComponent(id) + "/events?after=" + after);
  S.es = es;
  es.addEventListener("hello", (e) => {
    const h = JSON.parse(e.data);
    if (S.es === es && S.boot && h.boot !== S.boot) openChat(id, true);  // the server restarted
  });
  es.onmessage = (e) => { if (S.es === es) for (const ev of JSON.parse(e.data)) handle(ev, false); };
}

function handle(ev, replay) {
  switch (ev.type) {
    case "done": {
      setRunning(false);
      const st = ev.state || {};
      if (st.window) usage(st.used, st.window);
      menus(st);  // /effort and /yolo change them in the chat
      if (!replay) { S.adopt = true; refreshList(); loadPrompts(); }
      break;
    }
    case "busy":
      S.view.note("error", "A turn is running in this session in " + ev.holder + ". Send again when it is done.");
      if (S.view.pending) { S.view.pending.remove(); S.view.pending = null; }
      if (!$("input").value) { $("input").value = S.lastSent || ""; autosize(); }
      break;
    case "status": usage(ev.used, ev.window); break;
    default:
      S.view.event(ev);
      if (!replay && ev.type === "record" && !S.chats.some((c) => c.id === S.id)) refreshList();  // a new chat's first message
  }
}

function usage(used, window) {
  const pct = window ? Math.round((100 * used) / window) : 0;
  const u = $("usage");
  u.textContent = tokens(used) + " / " + tokens(window) + " tokens";
  u.className = pct >= 85 ? "bad" : pct >= 60 ? "warn" : "";
  u.title = pct + "% of the context window";
}

// The menus show what the open chat runs with; a new chat starts with the last ones picked.
function menus(st) {
  $("effort").value = (st && st.effort) || prefs.effort || (S.state && S.state.effort) || "max";
  $("mode").value = (st && st.mode) || prefs.mode || (S.state && S.state.mode) || "ask";
}

function setRunning(on) {
  S.running = on;
  if (on && !S.started) S.started = Date.now();
  if (!on) S.started = 0;
  document.body.classList.toggle("running", on);
  $("send").title = on ? "Stop (Esc)" : "Send (Enter)";
  $("send").setAttribute("aria-label", on ? "Stop" : "Send");
  tick();
}

function tick() {
  $("working").textContent = S.running ? "Working · " + took(Date.now() - (S.started || Date.now())) : "";
}

function showHolder() {
  const b = $("holder");
  b.hidden = !S.holder || S.running;
  b.textContent = S.holder ? "A turn is running in this session in " + S.holder + ". Its messages show up here as they are saved." : "";
}

function header() {
  const c = S.chats.find((x) => x.id === S.id);
  $("title").textContent = S.id ? (c ? c.title : "…") : "New chat";
  $("title").disabled = !c;
  $("where").textContent = c ? tilde(c.cwd) : "";
  document.title = (c ? c.title + " · " : "") + "fable";
}

// ---- new chats and sending -------------------------------------------------------------
function projectPicker() {
  const sel = $("project");
  const projects = (S.state && S.state.projects) || [];
  sel.textContent = "";
  for (const p of projects) sel.append(el("option", { value: p }, tilde(p)));
  sel.append(el("option", { value: "" }, "Another folder…"));
  sel.value = projects.includes(prefs.project) ? prefs.project : projects[0] || "";
  $("folder").hidden = sel.value !== "";
}

function folder() {
  const v = $("project").value || $("folder").value.trim();
  const h = S.state && S.state.home;
  return v.startsWith("~/") && h ? h + v.slice(1) : v;
}

async function submit() {
  if (S.running) return stop();
  const input = $("input");
  const text = input.value.trim();
  if (!text) return;
  const settings = { effort: $("effort").value, mode: $("mode").value };
  if (text === "/clear") {
    input.value = "";
    location.hash = "";
    return;
  }
  input.value = ""; autosize(); S.recall = -1;
  S.lastSent = text;
  if (!S.id) {
    const cwd = folder();
    if (!cwd) { input.value = text; $("folder").hidden = false; $("folder").focus(); return; }
    setRunning(true);
    try {
      const { id } = await api("/api/chats", { cwd, text, ...settings });
      prefs.project = cwd; savePrefs();
      await refreshList();
      location.hash = id;
    } catch (e) {
      setRunning(false);
      input.value = text; autosize();
      S.view.note("error", e.message);
    }
    return;
  }
  const command = text.startsWith("/");
  if (!command) S.view.user(text, true);
  toBottom();
  setRunning(true);
  try {
    await api("/api/chats/" + encodeURIComponent(S.id) + "/send", { text, ...settings });
  } catch (e) {
    setRunning(false);
    if (S.view.pending) { S.view.pending.remove(); S.view.pending = null; }
    input.value = text; autosize();
    S.view.note("error", e.message);
  }
}

async function stop() {
  if (!S.id) return;
  try { await api("/api/chats/" + encodeURIComponent(S.id) + "/stop", {}); } catch (e) { S.view.note("error", e.message); }
}

function autosize() {
  const t = $("input");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, window.innerHeight * 0.4) + "px";
}

async function loadPrompts() {
  try { S.prompts = (await api("/api/prompts")).prompts; } catch (e) { /* keep the old list */ }
}

function recall(step) {  // Up and Down through what was sent before, here or in the terminal
  const t = $("input");
  if (!S.prompts.length) return false;
  if (S.recall === -1) S.draft = t.value;
  const next = S.recall + step;
  if (next < -1 || next >= S.prompts.length) return false;
  S.recall = next;
  t.value = next === -1 ? S.draft : S.prompts[next];
  autosize();
  t.setSelectionRange(t.value.length, t.value.length);
  return true;
}

// ---- start -----------------------------------------------------------------------------
async function loadState() {
  try { S.state = await api("/api/state"); } catch (e) { S.state = null; }
  const st = S.state;
  const m = $("model");
  if (!st) { m.textContent = "the chat server is not answering"; m.className = "bad"; return; }
  if (st.error) {
    m.textContent = "no model server at " + st.base_url;
    m.className = "bad";
    m.title = st.error;
  } else {
    m.textContent = st.model + (st.context ? " · " + tokens(st.context) + " context" : "");
    m.className = "";
    m.title = (st.server || "server") + " at " + st.base_url;
  }
  $("version").textContent = "fable " + st.version;
}

function setup() {
  const effort = $("effort"), mode = $("mode");
  for (const l of (S.state && S.state.levels) || ["off", "low", "medium", "high", "max"]) effort.append(el("option", { value: l }, l));
  for (const [k, v] of Object.entries(MODES)) mode.append(el("option", { value: k }, v));
  menus(null);
  effort.onchange = () => { prefs.effort = effort.value; savePrefs(); };
  mode.onchange = () => { prefs.mode = mode.value; savePrefs(); };

  $("composer").addEventListener("submit", (e) => { e.preventDefault(); submit(); });
  const input = $("input");
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!S.running) submit(); return; }
    if (e.key === "Escape" && S.running) { e.preventDefault(); stop(); return; }
    const caret = input.selectionStart === input.selectionEnd;
    if (e.key === "ArrowUp" && caret && !input.value.slice(0, input.selectionStart).includes("\n")) { if (recall(1)) e.preventDefault(); }
    if (e.key === "ArrowDown" && caret && !input.value.slice(input.selectionEnd).includes("\n")) { if (recall(-1)) e.preventDefault(); }
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && S.running && document.activeElement !== input) stop(); });
  $("project").onchange = () => { $("folder").hidden = $("project").value !== ""; if (!$("folder").hidden) $("folder").focus(); };
  $("new").onclick = () => { location.hash = ""; if (!S.id) openChat(""); closeSide(); };
  $("filter").oninput = renderList;
  $("title").onclick = () => { const c = S.chats.find((x) => x.id === S.id); if (c) rename(c); };
  $("menu").onclick = () => document.body.classList.toggle("side-open");
  $("shade").onclick = closeSide;
  $("list").addEventListener("click", (e) => { if (e.target.closest("a.item")) closeSide(); });
  document.addEventListener("click", (e) => {
    const b = e.target.closest("button.copy");
    if (!b) return;
    copyText(b.closest(".code").querySelector("code").textContent).then(() => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1500); });
  });
  window.addEventListener("hashchange", () => openChat(location.hash.slice(1)));
}

function closeSide() { document.body.classList.remove("side-open"); }

(async function main() {
  await loadState();
  setup();
  await Promise.all([refreshList(), loadPrompts()]);
  await openChat(location.hash.slice(1));
  setInterval(tick, 1000);
  setInterval(() => refreshList(), 5000);
  setInterval(loadState, 30000);
})();
