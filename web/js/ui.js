// Shared page chrome and small renderers.

import { info } from "./api.js";

const NAV = [
  ["dashboard.html", "Workshop", "工房"],
  ["live.html", "Record & train", "記録・稽古"],
  ["radar.html", "Succession", "継承"],
];

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export const $ = (sel, root = document) => root.querySelector(sel);

export function header() {
  const here = location.pathname.split("/").pop() || "dashboard.html";
  const bar = document.createElement("header");
  bar.className = "topbar";
  bar.innerHTML = `
    <a class="brand" href="index.html"><i class="mark"></i><b>TSUKUMO</b><span>付喪</span></a>
    <nav>${NAV.map(([href, en, ja]) =>
      `<a href="${href}" class="${here === href ? "active" : ""}">${en}<span class="ja">${ja}</span></a>`).join("")}</nav>
    <div class="spacer"></div>
    <span class="pill" id="ai-pill">AI …</span>
    ${themeSwitch()}`;
  document.body.prepend(bar);
  wireThemeSwitches();
  info().then((i) => {
    const pill = $("#ai-pill");
    pill.textContent = i.ai === "claude" ? "Claude connected" : "AI offline: rule-based cards";
    pill.className = `pill ${i.ai === "claude" ? "ok" : "warn"}`;
  });
}

// Theme: "dark" (sumi) is the default; the choice is remembered per browser.
export const theme = () => document.documentElement.dataset.theme === "light" ? "light" : "dark";

export function setTheme(t) {
  if (t === "light") document.documentElement.dataset.theme = "light";
  else delete document.documentElement.dataset.theme;
  try { localStorage.setItem("tsukumo-theme", t); } catch {}
  document.querySelectorAll(".switch[data-theme-switch]").forEach((b) => {
    b.setAttribute("aria-checked", String(t === "light"));
  });
  window.dispatchEvent(new Event("resize"));  // canvases read colours from CSS at draw time
}

export const themeSwitch = () =>
  `<button class="switch" role="switch" data-theme-switch aria-checked="${theme() === "light"}" aria-label="Light theme"><i></i></button>`;

export function wireThemeSwitches(root = document) {
  root.querySelectorAll(".switch[data-theme-switch]:not([data-wired])").forEach((b) => {
    b.dataset.wired = "1";
    b.addEventListener("click", () => setTheme(theme() === "light" ? "dark" : "light"));
  });
}

export function initials(name) {
  return name.split(/\s+/).map((p) => p[0]).slice(0, 2).join("").toUpperCase();
}

// Mirrors services.by_camera: filmed scores carry the tracker's noise.
export const filmed = (source) => source === "camera" || source.startsWith("video");

export function scoreClass(score) {
  return score == null ? "" : score >= 75 ? "ok" : score >= 55 ? "warn" : "bad";
}

export function ago(ts) {
  const d = (Date.now() / 1000 - ts) / 86400;
  if (d < 1 / 24) return "just now";
  if (d < 1) return `${Math.round(d * 24)}h ago`;
  return `${Math.round(d)}d ago`;
}

export function cardHTML(card, lang = "en", { showLangs = true, langs = ["ja", "en", "vi", "id"], sub = null } = {}) {
  const c = card.content[lang] || card.content.en;
  const second = sub && sub !== lang ? card.content[sub] : null;
  return `
    <article class="kcard" data-card="${esc(card.id)}">
      <div class="row" style="justify-content:space-between">
        <div>
          <div class="k-title">${esc(c.title)}</div>
          ${second ? `<div class="k-sub">${esc(second.title)}</div>` : ""}
        </div>
        ${showLangs ? `<div class="langs">${langs.map((l) =>
          `<button data-lang="${l}" class="${l === lang ? "on" : ""}">${l.toUpperCase()}</button>`).join("")}</div>` : ""}
      </div>
      <dl>
        <dt>When</dt><dd>${esc(c.trigger)}</dd>
        <dt>Do</dt><dd>${esc(c.action)}</dd>
        <dt>Why</dt><dd>${esc(c.reason)}</dd>
      </dl>
      <div class="k-foot">
        <span>stroke ${card.stroke_index}${card.evidence?.summary ? ` · ${esc(card.evidence.summary)}` : ""}</span>
        <span>${card.source === "claude" ? "structured by Claude" : card.source === "seed" ? "recorded" : "offline draft"}</span>
      </div>
    </article>`;
}

// Re-render a card in another language when its language buttons are clicked.
export function wireCardLangs(root, cardsById) {
  root.addEventListener("click", (e) => {
    const btn = e.target.closest(".langs button");
    if (!btn) return;
    const el = btn.closest(".kcard");
    const card = cardsById[el.dataset.card];
    el.outerHTML = cardHTML(card, btn.dataset.lang);
  });
}

export function toast(html, kind = "", ms = 4500) {
  let stack = $(".toast-stack");
  if (!stack) {
    stack = document.createElement("div");
    stack.className = "toast-stack";
    document.body.append(stack);
  }
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.innerHTML = html;
  stack.append(t);
  setTimeout(() => t.remove(), ms);
  return t;
}
