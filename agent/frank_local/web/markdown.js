// Markdown to HTML for the chat page: what models write (fenced code, lists, tables, quotes,
// headings, emphasis, links). Everything is escaped first; the only tags in the output are the
// ones made here, and links are http(s) or mailto. No images, so a reply cannot make the
// browser fetch an address it chose.
(function (root) {
  "use strict";

  const ESC = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ESC[c]);
  }

  function safeUrl(url) {
    const u = url.trim().replace(/^<|>$/g, "");
    return /^(https?:\/\/|mailto:)/i.test(u) ? u : null;
  }

  function link(href, html) {
    return '<a href="' + esc(href) + '" target="_blank" rel="noopener noreferrer">' + html + "</a>";
  }

  // Inline markup. Code spans, links and bare URLs are cut out into slots first, so that
  // emphasis never reaches inside them; the rest is escaped, formatted, and the slots put back.
  function inline(src) {
    const slots = [];
    const keep = (html) => "\u0000" + (slots.push(html) - 1) + "\u0000";
    let s = src;
    s = s.replace(/(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g, (_, ticks, code) => {
      const c = code.length > 2 && code[0] === " " && code[code.length - 1] === " " ? code.slice(1, -1) : code;
      return keep("<code>" + esc(c) + "</code>");
    });
    s = s.replace(/\[([^\]\n]+)\]\(\s*(<[^>\n]*>|[^\s)]+)(?:\s+"[^"\n]*")?\s*\)/g, (all, text, url) => {
      const href = safeUrl(url);
      return href ? keep(link(href, emphasis(esc(text)))) : all;
    });
    s = s.replace(/\bhttps?:\/\/[^\s<>"'`\u0000]+/g, (url) => {
      let cut = url;
      for (;;) {  // trailing punctuation belongs to the sentence, and a ")" only if it opens nowhere
        const last = cut[cut.length - 1];
        const unbalanced = last === ")" && (cut.match(/\)/g) || []).length > (cut.match(/\(/g) || []).length;
        if (/[.,:;!?*_\]]/.test(last) || unbalanced) cut = cut.slice(0, -1);
        else break;
      }
      return keep(link(cut, esc(cut))) + url.slice(cut.length);
    });
    s = emphasis(esc(s));
    for (let i = 0; i < 4 && s.indexOf("\u0000") >= 0; i++) {
      s = s.replace(/\u0000(\d+)\u0000/g, (_, n) => slots[+n]);
    }
    return s;
  }

  function emphasis(s) {
    return s
      .replace(/\*\*(?=\S)([\s\S]*?\S)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^\w])__(?=\S)([\s\S]*?\S)__(?!\w)/g, "$1<strong>$2</strong>")
      .replace(/(^|[^*\w])\*(?=[^\s*])([^*\n]*?[^\s*])\*(?!\*)/g, "$1<em>$2</em>")
      .replace(/(^|[^\w])_(?=[^\s_])([^_\n]*?[^\s_])_(?!\w)/g, "$1<em>$2</em>")
      .replace(/~~(?=\S)([\s\S]*?\S)~~/g, "<del>$1</del>");
  }

  const FENCE = /^ {0,3}(`{3,}|~{3,})\s*([^\s`]*)[^`]*$/;
  const HEADING = /^ {0,3}(#{1,6})(?:\s+(.*?))?\s*#*\s*$/;
  const RULE = /^ {0,3}([-*_])(?:\s*\1){2,}\s*$/;
  const QUOTE = /^ {0,3}> ?/;
  const ITEM = /^(\s*)([-*+]|(\d{1,9})[.)])(\s+|$)(.*)$/;
  const DELIM = /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/;

  function blank(line) {
    return /^\s*$/.test(line);
  }

  function indentOf(line) {
    return line.match(/^\s*/)[0].replace(/\t/g, "    ").length;
  }

  function isTable(lines, i) {
    return lines[i].indexOf("|") >= 0 && i + 1 < lines.length && DELIM.test(lines[i + 1]) && lines[i + 1].indexOf("-") >= 0;
  }

  function startsBlock(lines, i) {
    const l = lines[i];
    return FENCE.test(l) || HEADING.test(l) || RULE.test(l) || QUOTE.test(l) || ITEM.test(l) || isTable(lines, i);
  }

  function cells(row) {
    let s = row.trim();
    if (s.startsWith("|")) s = s.slice(1);
    if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
    const out = [];
    let cur = "";
    for (let i = 0; i < s.length; i++) {
      if (s[i] === "\\" && s[i + 1] === "|") { cur += "|"; i++; } else if (s[i] === "|") { out.push(cur.trim()); cur = ""; } else cur += s[i];
    }
    out.push(cur.trim());
    return out;
  }

  function codeBlock(code, lang) {
    return '<div class="code"><div class="code-head"><span>' + esc(lang) + '</span><button type="button" class="copy">Copy</button></div>' +
      "<pre><code>" + esc(code) + "</code></pre></div>";
  }

  function render(src) {
    const lines = String(src == null ? "" : src).replace(/\u0000/g, "").replace(/\r\n?/g, "\n").split("\n");
    return blocks(lines);
  }

  function blocks(lines) {
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      let m;
      if (blank(line)) { i++; continue; }

      if ((m = line.match(FENCE))) {
        const mark = m[1];
        const indent = indentOf(line);
        const close = new RegExp("^ {0,3}" + (mark[0] === "`" ? "`" : "~") + "{" + mark.length + ",}\\s*$");
        const body = [];
        i++;
        while (i < lines.length && !close.test(lines[i])) {
          body.push(indent ? lines[i].replace(new RegExp("^ {0," + indent + "}"), "") : lines[i]);
          i++;
        }
        i++;  // the closing fence; a reply still streaming may not have it yet
        out.push(codeBlock(body.join("\n"), m[2]));
        continue;
      }

      if ((m = line.match(HEADING))) {
        const n = m[1].length;
        out.push("<h" + n + ">" + inline(m[2] || "") + "</h" + n + ">");
        i++;
        continue;
      }

      if (RULE.test(line)) { out.push("<hr>"); i++; continue; }

      if (isTable(lines, i)) {
        const head = cells(lines[i]);
        const aligns = cells(lines[i + 1]).map((c) => (/^:-+:$/.test(c) ? "center" : /-:$/.test(c) ? "right" : /^:-/.test(c) ? "left" : ""));
        const cell = (tag, c, k) => "<" + tag + (aligns[k] ? ' style="text-align:' + aligns[k] + '"' : "") + ">" + inline(c) + "</" + tag + ">";
        let html = '<div class="table"><table><thead><tr>' + head.map((c, k) => cell("th", c, k)).join("") + "</tr></thead><tbody>";
        i += 2;
        while (i < lines.length && !blank(lines[i]) && lines[i].indexOf("|") >= 0) {
          const row = cells(lines[i]);
          html += "<tr>" + head.map((_, k) => cell("td", row[k] || "", k)).join("") + "</tr>";
          i++;
        }
        out.push(html + "</tbody></table></div>");
        continue;
      }

      if (QUOTE.test(line)) {
        const inner = [];
        while (i < lines.length && !blank(lines[i]) && (QUOTE.test(lines[i]) || !startsBlock(lines, i))) {
          inner.push(lines[i].replace(QUOTE, ""));
          i++;
        }
        out.push("<blockquote>" + blocks(inner) + "</blockquote>");
        continue;
      }

      if ((m = line.match(ITEM))) {
        i = list(lines, i, out);
        continue;
      }

      const para = [];
      while (i < lines.length && !blank(lines[i]) && (para.length === 0 || !startsBlock(lines, i))) {
        para.push(lines[i].trim());
        i++;
      }
      out.push("<p>" + inline(para.join("\n")).replace(/\n/g, "<br>") + "</p>");
    }
    return out.join("");
  }

  function list(lines, i, out) {
    const first = lines[i].match(ITEM);
    const base = indentOf(lines[i]);
    const ordered = first[3] !== undefined;
    const items = [];
    let cur = null;
    let loose = false;
    while (i < lines.length) {
      const line = lines[i];
      const m = line.match(ITEM);
      if (m && indentOf(line) <= base + 1 && (m[3] !== undefined) === ordered) {
        cur = { lines: [m[5]], indent: indentOf(line) + m[2].length + Math.max(1, m[4].length) };
        items.push(cur);
        i++;
        continue;
      }
      if (blank(line)) {
        let j = i + 1;
        while (j < lines.length && blank(lines[j])) j++;
        if (j < lines.length && cur && indentOf(lines[j]) > base) {
          const next = lines[j].match(ITEM);
          if (!next || indentOf(lines[j]) > base + 1 || (next[3] !== undefined) === ordered) {
            loose = true;
            for (; i < j; i++) cur.lines.push("");
            continue;
          }
        }
        if (j < lines.length && cur) {
          const next = lines[j].match(ITEM);
          if (next && indentOf(lines[j]) <= base + 1 && (next[3] !== undefined) === ordered) {
            loose = true;
            i = j;
            continue;
          }
        }
        break;
      }
      if (cur && indentOf(line) > base) {
        cur.lines.push(line.slice(Math.min(indentOf(line), cur.indent)));
        i++;
        continue;
      }
      if (cur && !startsBlock(lines, i)) {  // a wrapped line of the item above
        cur.lines.push(line.trim());
        i++;
        continue;
      }
      break;
    }
    const tag = ordered ? "ol" : "ul";
    const start = ordered && +first[3] !== 1 ? ' start="' + +first[3] + '"' : "";
    out.push("<" + tag + start + ">" + items.map((it) => "<li>" + item(it.lines, loose) + "</li>").join("") + "</" + tag + ">");
    return i;
  }

  function item(lines, loose) {
    let box = "";
    const m = lines[0].match(/^\[([ xX])\]\s+/);
    if (m) {
      box = '<span class="box">' + (m[1] === " " ? "☐" : "☑") + "</span> ";
      lines = [lines[0].slice(m[0].length), ...lines.slice(1)];
    }
    let html = blocks(lines);
    if (!loose) html = html.replace(/^<p>([\s\S]*?)<\/p>/, "$1");
    return box + html;
  }

  const api = { render, inline, esc };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Markdown = api;
})(typeof window !== "undefined" ? window : globalThis);
