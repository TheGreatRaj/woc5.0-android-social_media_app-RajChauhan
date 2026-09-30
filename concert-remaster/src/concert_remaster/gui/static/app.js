// Concert Remaster front end: plain ES modules, no build step, no internet.

const $ = (sel, root = document) => root.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "style") node.style.cssText = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "html") node.innerHTML = value;
    else if (value === true) node.setAttribute(key, "");
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, { method = "GET", body } = {}) {
  const res = await fetch(path, {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not JSON */ }
    throw new Error(detail);
  }
  const type = res.headers.get("content-type") || "";
  return type.includes("application/json") ? res.json() : res.blob();
}

function toast(message, error = false) {
  const t = $("#toast");
  t.textContent = message;
  t.className = "toast show" + (error ? " error" : "");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.className = "toast"), error ? 6000 : 2800);
}

async function guarded(fn) {
  try { return await fn(); } catch (e) { toast(e.message || String(e), true); }
}

const fmtTime = (s) => {
  if (s == null || !isFinite(s)) return "–";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
};
const parseTime = (text) => {
  const parts = String(text).trim().split(":").map(Number);
  if (parts.some(isNaN)) return null;
  return parts.reduce((acc, p) => acc * 60 + p, 0);
};
const fmtSize = (b) => (b > 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.round(b / 1e3)} kB`);

const KIND_LABEL = { song: "Song", interlude: "Interlude", talk: "Artist talking", crowd: "Applause", silence: "Silence" };
const KIND_COLOR = { song: "#8b5cf6", interlude: "#a78bfa", talk: "#f59e0b", crowd: "#10b981", silence: "#475569" };
const ACTIONS = {
  talk: [["", "Default"], ["enhance", "Clear voice"], ["keep", "Keep as is"], ["remove", "Cut out"], ["music_only", "Music only"]],
  crowd: [["", "Default"], ["keep", "Keep"], ["shorten", "Shorten"], ["remove", "Cut out"]],
  song: [["", "Default"], ["keep", "Keep speech"], ["music_only", "Mute speech"]],
};

const state = {
  info: null, schema: null, projects: [], current: null, peaks: null,
  selected: null, poll: null, saveTimer: null, lastStatus: null, playing: null,
};

// --- boot -------------------------------------------------------------------------------

async function init() {
  const [info, schema] = await Promise.all([api("/api/info"), api("/api/schema")]);
  state.info = info;
  state.schema = schema;
  const d = info.devices;
  const device = $("#device");
  if (d.cuda) { device.textContent = `NVIDIA · ${d.gpu}`; device.classList.add("good"); }
  else if (d.directml) { device.textContent = `AMD/DirectML · ${d.directml_device || "GPU"}`; device.classList.add("accent"); }
  else { device.textContent = `CPU · ${d.cpu_threads} threads`; device.classList.add("warn"); }
  if (!info.ffmpeg) toast("ffmpeg was not found. Run setup.bat again.", true);

  $("#btn-new").onclick = openNewDialog;
  $("#btn-defaults").onclick = () => openSettings(null);
  await refreshProjects();
  if (state.projects.length) await selectProject(state.projects[0].id);
  else renderEmpty();
  setInterval(refreshProjects, 5000);
}

async function refreshProjects() {
  try { state.projects = await api("/api/projects"); } catch { return; }
  renderSidebar();
  const running = state.projects.find((p) => p.running);
  if (running && state.current?.id === running.id && !state.poll) startPolling();
}

function anyRunning() { return state.projects.some((p) => p.running); }

function statusChip(p) {
  const status = p.progress?.status;
  if (p.running) return el("span", { class: "chip accent" }, `${Math.round((p.progress.overall || 0) * 100)}%`);
  if (status === "error") return el("span", { class: "chip bad" }, "error");
  if (status === "cancelled" || status === "interrupted") return el("span", { class: "chip warn" }, "paused");
  if (p.exported) return el("span", { class: "chip good" }, "exported");
  if (p.analyzed) return el("span", { class: "chip" }, `${p.songs} songs`);
  return el("span", { class: "chip" }, "new");
}

function renderSidebar() {
  const list = $("#projects");
  list.replaceChildren(...state.projects.map((p) =>
    el("div", { class: "project-item" + (state.current?.id === p.id ? " active" : ""), onclick: () => selectProject(p.id) },
      el("div", { class: "name", title: p.name }, p.name),
      el("div", { class: "meta" }, fmtTime(p.source_info?.duration), statusChip(p)))));
  if (!state.projects.length) list.append(el("div", { class: "muted small", style: "padding:6px" }, "No projects yet."));
}

function renderEmpty() {
  state.current = null;
  $("#main").replaceChildren(el("div", { class: "empty" },
    el("h1", {}, "Make your concert recordings sound like the record"),
    el("p", { class: "muted" }, "Everything runs on this PC with local AI models. Songs are found automatically, crowd noise and venue echo removed, every instrument separated, and the result mixed and mastered."),
    el("div", { class: "features" },
      el("div", { class: "feature" }, el("b", {}, "Separate"), el("span", { class: "muted small" }, "Vocals, drums, bass, guitar, piano, flute and more as clean stems.")),
      el("div", { class: "feature" }, el("b", {}, "Recognise"), el("span", { class: "muted small" }, "Songs, the artist's talking and applause, across 3-hour shows.")),
      el("div", { class: "feature" }, el("b", {}, "Master"), el("span", { class: "muted small" }, "Tone matched to the studio original, streaming-ready loudness."))),
    el("button", { class: "btn primary", onclick: openNewDialog }, "Start with a recording")));
}

// --- project view ----------------------------------------------------------------------

async function selectProject(id) {
  stopPolling();
  state.selected = null;
  state.peaks = null;
  await loadProject(id);
  renderSidebar();
}

async function loadProject(id) {
  const detail = await api(`/api/projects/${encodeURIComponent(id)}`);
  const changed = state.current?.id !== id;
  state.current = detail;
  state.lastStatus = detail.progress?.status;
  if ((changed || !state.peaks) && detail.stages.includes("source")) {
    guarded(async () => { state.peaks = await api(`/api/projects/${encodeURIComponent(id)}/peaks`); timeline?.setPeaks(state.peaks); });
  }
  if (!state.selected && detail.segments?.length) state.selected = (detail.segments.find((s) => s.kind === "song") || detail.segments[0]).id;
  renderProject();
  if (detail.running) startPolling();
}

function projectUrl(suffix = "") { return `/api/projects/${encodeURIComponent(state.current.id)}${suffix}`; }

let timeline = null;

function renderProject() {
  const p = state.current;
  const main = $("#main");
  const info = p.source_info || {};
  const analysis = p.analysis || {};
  const chips = [
    info.duration && el("span", { class: "chip" }, fmtTime(info.duration)),
    info.codec && el("span", { class: "chip" }, `${info.codec.toUpperCase()} ${info.sample_rate ? (info.sample_rate / 1000) + " kHz" : ""}`),
    info.bitrate_kbps && el("span", { class: "chip" }, `${info.bitrate_kbps} kbps`),
    analysis.mono !== undefined && el("span", { class: "chip" + (analysis.mono ? " warn" : "") }, analysis.mono ? "mono recording" : "stereo"),
    analysis.bandwidth_hz && el("span", { class: "chip" + (analysis.bandwidth_hz < 15000 ? " warn" : "") }, `content up to ${(analysis.bandwidth_hz / 1000).toFixed(1)} kHz`),
    analysis.declipped_samples > 0 && el("span", { class: "chip warn" }, `${analysis.declipped_samples.toLocaleString()} clipped samples repaired`),
  ];
  const busy = anyRunning();
  const header = el("div", { class: "card" },
    el("div", { class: "proj-head" },
      el("div", {}, el("h1", {}, p.name), el("div", { class: "path" }, p.source), el("div", { class: "row wrap" }, chips)),
      el("div", { class: "row" },
        el("button", { class: "btn", onclick: () => openSettings(p) }, "Settings"),
        el("button", { class: "btn ghost", onclick: () => guarded(() => api("/api/open", { method: "POST", body: { path: p.source.replace(/[^\\/]+$/, "") } })) }, "Open source folder"),
        el("button", { class: "btn ghost danger", onclick: deleteProject, disabled: p.running }, "Delete"))),
    renderSteps(p));
  const parts = [header, renderProgress(p)];
  if (p.segments?.length) {
    parts.push(renderTimelineCard(p), renderSegmentsCard(p));
    const seg = p.segments.find((s) => s.id === state.selected);
    if (seg) parts.push(seg.kind === "song" || seg.kind === "interlude" ? renderSongPanel(seg) : renderOtherPanel(seg));
    parts.push(renderExportCard(p));
  } else if (!p.running) {
    parts.push(el("div", { class: "card" }, el("h3", {}, "Analyze"),
      el("p", { class: "muted" }, "Separates every instrument with the AI models, finds the songs, the artist's talking and the applause, and identifies each song. Long shows take a while; you can stop at any time and continue later."),
      el("button", { class: "btn primary", disabled: busy, onclick: () => runTask("analyze") }, p.stages.length ? "Continue analysis" : "Start analysis")));
  }
  main.replaceChildren(...parts.filter(Boolean));
  if (p.segments?.length) setupTimeline();
}

function renderSteps(p) {
  const analyzing = p.running && ["source", "separation", "segments", "identify", "starting"].includes(p.progress?.stage);
  const exporting = p.running && p.progress?.stage === "render";
  const step = (num, label, sub, cls) => el("div", { class: `step ${cls}` }, el("div", { class: "num" }, num), el("div", {}, el("div", { class: "label" }, label), el("div", { class: "sub" }, sub)));
  return el("div", { class: "steps" },
    step(1, "Analyze", analyzing ? "running…" : p.analyzed ? `${p.songs} songs found` : "separate, detect, identify", p.analyzed ? "done" : analyzing || !p.analyzed ? "active" : ""),
    step(2, "Review", "titles, talking, mix", p.analyzed && !p.exported && !exporting ? "active" : p.exported ? "done" : ""),
    step(3, "Export", exporting ? "running…" : p.exported ? "files ready" : "songs, show, stems", exporting ? "active" : p.exported ? "done" : ""));
}

function renderProgress(p) {
  const pr = p.progress || {};
  if (!p.running && !["error", "cancelled", "interrupted"].includes(pr.status)) return null;
  const card = el("div", { class: "card progress-card", id: "progress-card" });
  fillProgress(card, pr, p.running);
  return card;
}

function fillProgress(card, pr, running) {
  const pct = Math.round((pr.overall ?? pr.fraction ?? 0) * 1000) / 10;
  const eta = pr.eta_seconds ? `about ${fmtTime(pr.eta_seconds)} left in this step` : "";
  const parts = [
    el("div", { class: "card-head" },
      el("div", {}, el("div", { class: "stage" }, running ? (pr.stage_label || "Working") : pr.status === "error" ? "Something went wrong" : "Paused"),
        el("div", { class: "muted small" }, pr.message || "")),
      running ? el("button", { class: "btn danger", onclick: stopJob }, "Stop")
        : el("button", { class: "btn primary", disabled: anyRunning(), onclick: () => runTask(state.current.segments?.length ? "export" : "analyze") }, "Continue")),
  ];
  if (running || pr.status !== "error") {
    parts.push(el("div", { class: "bar" }, el("div", { class: "fill", style: `width:${pct}%` })),
      el("div", { class: "meta" }, el("span", {}, `${pct}%`), el("span", {}, eta), el("span", {}, pr.elapsed_seconds ? `running ${fmtTime(pr.elapsed_seconds)}` : "")));
  } else {
    parts.push(el("div", { class: "error-box" }, pr.message || "Unknown error"),
      el("div", { class: "muted small", style: "margin-top:8px" }, "The full log is in the project's 'logs' folder. Finished work is kept: fix the cause (e.g. settings) and press Continue."));
  }
  card.replaceChildren(...parts);
}

function startPolling() {
  stopPolling();
  state.poll = setInterval(async () => {
    if (!state.current) return;
    let pr;
    try { pr = await api(projectUrl("/progress")); } catch { return; }
    const card = $("#progress-card");
    if (pr.running) {
      if (card) fillProgress(card, pr, true); else renderProject();
    } else {
      stopPolling();
      await refreshProjects();
      await loadProject(state.current.id);
      if (pr.status === "done") toast(pr.message || "Finished");
    }
  }, 1000);
}

function stopPolling() { clearInterval(state.poll); state.poll = null; }

async function runTask(task, only) {
  await guarded(async () => {
    await api(projectUrl("/run"), { method: "POST", body: { task, only } });
    await refreshProjects();
    await loadProject(state.current.id);
    startPolling();
  });
}

async function stopJob() {
  await guarded(async () => { await api(projectUrl("/stop"), { method: "POST" }); toast("Stopped. Finished parts are kept."); });
  await refreshProjects();
  await loadProject(state.current.id);
}

async function deleteProject() {
  const keep = confirm("Delete this project's working files?\n\nOK = delete working files but keep exported songs.\nCancel = do nothing.");
  if (!keep) return;
  await guarded(async () => {
    await api(projectUrl("?keep_outputs=true"), { method: "DELETE" });
    toast("Working files deleted; exported files kept.");
    await refreshProjects();
    if (state.projects.length) selectProject(state.projects[0].id); else renderEmpty();
  });
}

// --- timeline ----------------------------------------------------------------------------

function renderTimelineCard(p) {
  return el("div", { class: "card" },
    el("div", { class: "card-head" },
      el("h3", {}, "Timeline"),
      el("div", { class: "timeline-tools" },
        el("span", { class: "muted small" }, "Scroll to zoom · drag to move · drag an edge to adjust · double-click to listen"),
        el("button", { class: "btn small", onclick: () => timeline?.zoom(0.5) }, "+"),
        el("button", { class: "btn small", onclick: () => timeline?.zoom(2) }, "−"),
        el("button", { class: "btn small", onclick: () => timeline?.fit() }, "Fit"),
        el("button", { class: "btn small", onclick: splitAtCursor, title: "Split the selected part where you last clicked" }, "Split here"),
        el("button", { class: "btn small", onclick: redetect, disabled: anyRunning(), title: "Detect songs and speech again with the current settings" }, "Re-detect"))),
    el("div", { class: "timeline-wrap" }, el("canvas", { id: "timeline" })),
    el("div", { class: "legend" }, ...Object.entries(KIND_LABEL).map(([k, v]) => el("span", {}, el("i", { class: "dot", style: `background:${KIND_COLOR[k]}` }), v))));
}

class Timeline {
  constructor(canvas, duration) {
    this.canvas = canvas;
    this.duration = duration;
    this.t0 = 0; this.t1 = duration;
    this.segments = []; this.peaks = null; this.selected = null; this.cursor = null; this.playhead = null;
    this.drag = null;
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
    canvas.addEventListener("wheel", (e) => { e.preventDefault(); this.zoom(e.deltaY > 0 ? 1.25 : 0.8, this.timeAt(e.offsetX)); }, { passive: false });
    canvas.addEventListener("mousedown", (e) => this.down(e));
    window.addEventListener("mousemove", (e) => this.move(e));
    window.addEventListener("mouseup", (e) => this.up(e));
    canvas.addEventListener("dblclick", (e) => playRange("source", this.timeAt(e.offsetX), 30));
  }
  resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, r.width * dpr); this.canvas.height = Math.max(1, r.height * dpr);
    this.w = r.width; this.h = r.height; this.dpr = dpr;
    this.draw();
  }
  setPeaks(peaks) { this.peaks = peaks; this.draw(); }
  setSegments(segments, selected) { this.segments = segments; this.selected = selected; this.draw(); }
  timeAt(x) { return this.t0 + (x / this.w) * (this.t1 - this.t0); }
  xAt(t) { return ((t - this.t0) / (this.t1 - this.t0)) * this.w; }
  zoom(factor, around) {
    const c = around ?? (this.t0 + this.t1) / 2;
    const span = Math.min(this.duration, Math.max(5, (this.t1 - this.t0) * factor));
    this.t0 = Math.max(0, Math.min(this.duration - span, c - (c - this.t0) * (span / (this.t1 - this.t0))));
    this.t1 = this.t0 + span;
    this.draw();
  }
  fit() { this.t0 = 0; this.t1 = this.duration; this.draw(); }
  edgeAt(x) {
    for (let i = 0; i < this.segments.length - 1; i++) {
      const t = this.segments[i].end;
      if (Math.abs(this.xAt(t) - x) < 6) return i;
    }
    return -1;
  }
  down(e) {
    const x = e.offsetX, edge = this.edgeAt(x);
    if (edge >= 0) this.drag = { kind: "edge", index: edge };
    else this.drag = { kind: "pan", x, t0: this.t0, t1: this.t1, moved: false };
  }
  move(e) {
    if (!this.drag) {
      if (e.target === this.canvas) this.canvas.style.cursor = this.edgeAt(e.offsetX) >= 0 ? "ew-resize" : "crosshair";
      return;
    }
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    if (this.drag.kind === "edge") {
      const i = this.drag.index, a = this.segments[i], b = this.segments[i + 1];
      const t = Math.max(a.start + 1, Math.min(b.end - 1, this.timeAt(x)));
      a.end = b.start = Math.round(t * 10) / 10;
      this.draw();
    } else {
      const dt = ((x - this.drag.x) / this.w) * (this.drag.t1 - this.drag.t0);
      if (Math.abs(x - this.drag.x) > 3) this.drag.moved = true;
      const span = this.drag.t1 - this.drag.t0;
      this.t0 = Math.max(0, Math.min(this.duration - span, this.drag.t0 - dt));
      this.t1 = this.t0 + span;
      this.draw();
    }
  }
  up(e) {
    if (!this.drag) return;
    const drag = this.drag;
    this.drag = null;
    if (drag.kind === "edge") { segmentsChanged(); return; }
    if (!drag.moved && e.target === this.canvas) {
      const t = this.timeAt(e.offsetX);
      this.cursor = t;
      const seg = this.segments.find((s) => t >= s.start && t < s.end);
      if (seg) selectSegment(seg.id);
      this.draw();
    }
  }
  draw() {
    const ctx = this.canvas.getContext("2d");
    if (!ctx) return;
    const { w, h, dpr } = this;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const band = 26, wave = h - band - 18;
    for (const s of this.segments) {
      const x0 = this.xAt(s.start), x1 = this.xAt(s.end);
      if (x1 < 0 || x0 > w) continue;
      ctx.globalAlpha = s.include === false ? 0.08 : 0.13;
      ctx.fillStyle = KIND_COLOR[s.kind];
      ctx.fillRect(x0, 0, x1 - x0, wave);
      ctx.globalAlpha = s.include === false ? 0.35 : 0.9;
      ctx.fillRect(x0 + 1, wave + 4, Math.max(1, x1 - x0 - 2), band - 6);
      ctx.globalAlpha = 1;
      if (x1 - x0 > 50) {
        ctx.fillStyle = "#0d0f14";
        ctx.font = "600 11px Segoe UI, system-ui, sans-serif";
        const label = s.kind === "song" ? `${s.track}. ${s.title}` : KIND_LABEL[s.kind];
        ctx.save(); ctx.beginPath(); ctx.rect(x0 + 4, wave + 4, x1 - x0 - 8, band - 6); ctx.clip();
        ctx.fillText(label, x0 + 7, wave + 20); ctx.restore();
      }
      if (s.id === this.selected) {
        ctx.strokeStyle = "#e7e9ee"; ctx.lineWidth = 1.5;
        ctx.strokeRect(x0 + 0.75, 0.75, x1 - x0 - 1.5, wave + band - 2);
      }
    }
    if (this.peaks?.peaks?.length) {
      const rate = this.peaks.rate, data = this.peaks.peaks, mid = wave / 2;
      ctx.fillStyle = "#a5b4fc";
      for (let x = 0; x < w; x++) {
        const a = Math.floor(this.timeAt(x) * rate), b = Math.max(a + 1, Math.floor(this.timeAt(x + 1) * rate));
        let m = 0;
        for (let i = a; i < b && i < data.length; i++) if (data[i] > m) m = data[i];
        const hgt = Math.max(1, m * (wave / 2 - 4));
        ctx.fillRect(x, mid - hgt, 1, hgt * 2);
      }
    }
    ctx.fillStyle = "#8b93a7"; ctx.font = "11px Segoe UI, system-ui, sans-serif";
    const span = this.t1 - this.t0;
    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600].find((s) => span / s < 12) || 3600;
    for (let t = Math.ceil(this.t0 / step) * step; t < this.t1; t += step) {
      const x = this.xAt(t);
      ctx.fillRect(x, h - 16, 1, 4);
      ctx.fillText(fmtTime(t), x + 3, h - 4);
    }
    for (const [t, color] of [[this.cursor, "#22d3ee"], [this.playhead, "#f472b6"]]) {
      if (t == null) continue;
      const x = this.xAt(t);
      ctx.fillStyle = color; ctx.fillRect(x, 0, 1.5, wave + band);
    }
  }
}

function setupTimeline() {
  const canvas = $("#timeline");
  const p = state.current;
  const duration = p.analysis?.duration || p.source_info?.duration || 1;
  const keepView = timeline && timeline.duration === duration ? [timeline.t0, timeline.t1, timeline.cursor] : null;
  timeline = new Timeline(canvas, duration);
  if (keepView) [timeline.t0, timeline.t1, timeline.cursor] = keepView;
  timeline.setSegments(p.segments, state.selected);
  if (state.peaks) timeline.setPeaks(state.peaks);
}

function selectSegment(id) {
  state.selected = id;
  renderProject();
  document.querySelector(`tr[data-id="${id}"]`)?.scrollIntoView({ block: "nearest" });
}

function segmentsChanged() {
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(saveSegments, 400);
}

async function saveSegments() {
  await guarded(async () => {
    const segs = await api(projectUrl("/segments"), { method: "PUT", body: state.current.segments });
    const selectedStart = state.current.segments.find((s) => s.id === state.selected)?.start;
    state.current.segments = segs;
    const again = segs.find((s) => s.start === selectedStart);
    if (again) state.selected = again.id;
    renderProject();
  });
}

function splitAtCursor() {
  const t = timeline?.cursor;
  const segs = state.current.segments;
  const i = segs.findIndex((s) => t > s.start + 1 && t < s.end - 1);
  if (t == null || i < 0) { toast("Click on the timeline where you want to split first."); return; }
  const a = segs[i], b = { ...a, id: a.id + "b", start: Math.round(t * 10) / 10, title: "", reference: null, identification: null, locked: false };
  a.end = b.start;
  segs.splice(i + 1, 0, b);
  saveSegments();
}

function mergeWithNext(seg) {
  const segs = state.current.segments, i = segs.findIndex((s) => s.id === seg.id);
  if (i < 0 || i === segs.length - 1) return;
  segs[i].end = segs[i + 1].end;
  segs.splice(i + 1, 1);
  saveSegments();
}

async function redetect() {
  if (!confirm("Detect songs, talking and applause again? Your edits to the timeline (titles, splits) will be replaced.")) return;
  runTask("redetect");
}

// --- segments table -------------------------------------------------------------------------

function renderSegmentsCard(p) {
  const rows = p.segments.map((s) => {
    const ident = s.identification;
    const actionOptions = ACTIONS[s.kind === "interlude" ? "song" : s.kind];
    return el("tr", { "data-id": s.id, class: (s.id === state.selected ? "selected " : "") + (s.include === false ? "excluded" : ""), onclick: () => selectSegment(s.id) },
      el("td", {}, el("input", { type: "checkbox", checked: s.include !== false, title: "Include in the export", onclick: (e) => e.stopPropagation(), onchange: (e) => { s.include = e.target.checked; segmentsChanged(); } })),
      el("td", { class: "kind-cell" }, el("i", { class: "dot", style: `background:${KIND_COLOR[s.kind]}` }),
        el("select", { onclick: (e) => e.stopPropagation(), onchange: (e) => { s.kind = e.target.value; segmentsChanged(); } },
          ...Object.entries(KIND_LABEL).map(([k, v]) => el("option", { value: k, selected: k === s.kind }, v)))),
      el("td", {}, s.kind === "song" ? el("b", {}, s.track) : ""),
      el("td", { class: "time" }, el("span", { title: "Start" }, fmtTime(s.start)), " – ", el("span", {}, fmtTime(s.end)), el("div", {}, fmtTime(s.end - s.start))),
      el("td", {}, s.kind === "song" || s.kind === "interlude"
        ? el("input", { type: "text", value: s.title || "", placeholder: "Title", onclick: (e) => e.stopPropagation(), onchange: (e) => { s.title = e.target.value; s.locked = true; segmentsChanged(); } })
        : el("span", { class: "muted small" }, (s.transcript || []).map((t) => t.text).join(" ").slice(0, 90))),
      el("td", {}, s.kind === "song" || s.kind === "interlude"
        ? el("input", { type: "text", value: s.artist || "", placeholder: "Artist", onclick: (e) => e.stopPropagation(), onchange: (e) => { s.artist = e.target.value; s.locked = true; segmentsChanged(); } }) : ""),
      el("td", {}, s.kind === "song" ? matchBadge(s) : ""),
      el("td", {}, actionOptions ? el("select", { onclick: (e) => e.stopPropagation(), onchange: (e) => { s.action = e.target.value || null; segmentsChanged(); } },
        ...actionOptions.map(([v, label]) => el("option", { value: v, selected: (s.action || "") === v }, label))) : ""),
      el("td", {}, el("button", { class: "icon-btn", title: "Listen to the original", onclick: (e) => { e.stopPropagation(); playRange("source", s.start, Math.min(30, s.end - s.start)); } }, "▶")));
  });
  return el("div", { class: "card" },
    el("div", { class: "card-head" }, el("h3", {}, "Songs and parts of the show"),
      el("div", { class: "row" },
        el("button", { class: "btn small", disabled: anyRunning(), onclick: () => runTask("identify") }, "Identify all songs again"))),
    el("table", { class: "seg-table" },
      el("thead", {}, el("tr", {}, ...["", "Part", "#", "Time", "Title / what was said", "Artist", "Match", "When…", ""].map((h) => el("th", {}, h)))),
      el("tbody", {}, rows)));
}

function matchBadge(s) {
  const ident = s.identification;
  if (s.reference && s.locked && !ident) return el("span", { class: "chip good score" }, "set by you");
  if (!ident) return el("span", { class: "chip score" }, "–");
  if (s.reference) return el("span", { class: "chip good score", title: ident.message }, `${Math.round((ident.score || 0) * 100)}% · ${s.reference.source}${s.reference.preview ? " preview" : ""}`);
  return el("span", { class: "chip warn score", title: ident.message }, "unreleased?");
}

// --- selected part panels ----------------------------------------------------------------------

function mixStemNames() {
  const aliases = state.current.aliases || {};
  const order = ["lead_vocals", "backing_vocals", "vocals", "drums", "bass", "guitar", "piano", "woodwinds", "other"];
  let names = order.filter((n) => n in aliases);
  if (names.includes("lead_vocals")) names = names.filter((n) => n !== "vocals");
  return names;
}

function renderSongPanel(seg) {
  const gains = seg.stem_gains || {};
  const names = mixStemNames();
  const offset = { value: 0 };
  const mixer = el("div", { class: "mixer" }, ...names.map((name) => {
    const muted = gains[name] !== undefined && gains[name] <= -60;
    const value = muted ? 0 : gains[name] || 0;
    const label = el("span", { class: "val" }, `${value > 0 ? "+" : ""}${value.toFixed(1)} dB`);
    return el("div", { class: "mixer-row" },
      el("span", { class: "name" }, name.replace("_", " ")),
      el("input", { type: "range", min: -12, max: 12, step: 0.5, value, oninput: (e) => { label.textContent = `${e.target.value > 0 ? "+" : ""}${Number(e.target.value).toFixed(1)} dB`; },
        onchange: (e) => { seg.stem_gains = { ...(seg.stem_gains || {}), [name]: Number(e.target.value) }; segmentsChanged(); } }),
      label,
      el("button", { class: "mute" + (muted ? " on" : ""), title: "Mute this instrument in this song", onclick: () => { seg.stem_gains = { ...(seg.stem_gains || {}), [name]: muted ? 0 : -120 }; segmentsChanged(); } }, "M"));
  }));
  const dur = seg.end - seg.start;
  const start = el("input", { type: "range", min: 0, max: Math.max(0, dur - 5), step: 1, value: 0, oninput: (e) => { offset.value = Number(e.target.value); startLabel.textContent = fmtTime(offset.value); } });
  const startLabel = el("span", { class: "muted small" }, "0:00");
  const ident = seg.identification || {};
  const ref = seg.reference;
  return el("div", { class: "card" },
    el("div", { class: "card-head" }, el("h3", {}, `Song ${seg.track ?? ""}: ${seg.title || ""}`),
      el("button", { class: "btn small", onclick: () => mergeWithNext(seg) }, "Merge with next part")),
    el("div", { class: "song-panel" },
      el("div", {},
        el("div", { class: "muted small", style: "margin-bottom:8px" }, "Extra level per instrument for this song, on top of the automatic mix."),
        mixer,
        el("div", { class: "row wrap", style: "margin-top:14px" },
          el("span", { class: "muted small" }, "Listen from"), start, startLabel),
        el("div", { class: "row wrap", style: "margin-top:8px" },
          el("button", { class: "btn", onclick: () => playRange("source", seg.start + offset.value, 30) }, "▶ Original"),
          el("button", { class: "btn", onclick: () => playRange(names.join("+"), seg.start + offset.value, 30) }, "▶ Separated, unprocessed"),
          el("button", { class: "btn primary", onclick: () => playPreview(seg, offset.value) }, "▶ Remastered preview"))),
      el("div", { class: "ref-box" },
        el("b", {}, "Studio reference"),
        el("p", { class: "muted small" }, ref
          ? `Tone and balance follow “${ref.title}”${ref.artist ? " by " + ref.artist : ""} (${ref.source}${ref.preview ? ", 30 s preview" : ""}).`
          : ident.message || "No reference: the song gets the AI clean-up and a generic studio balance."),
        (ident.candidates || []).length ? el("ul", { class: "candidates" }, ...ident.candidates.slice(0, 5).map((c) => el("li", {}, `${Math.round(c.score * 100)}% · ${c.title} – ${c.artist} (${c.source})`))) : null,
        ident.lyrics ? el("p", { class: "muted small" }, `Heard: “${ident.lyrics.slice(0, 160)}…”`) : null,
        el("div", { class: "row wrap", style: "margin-top:10px" },
          el("button", { class: "btn small", disabled: anyRunning(), onclick: () => runTask("identify", [seg.id]) }, "Identify again"),
          el("button", { class: "btn small", onclick: () => setReference(seg, "file") }, "Use a file…"),
          el("button", { class: "btn small", onclick: () => setReference(seg, "url") }, "Use a link…"),
          ref ? el("button", { class: "btn small ghost", onclick: () => setReference(seg, "clear") }, "No reference") : null))));
}

function renderOtherPanel(seg) {
  const transcript = seg.transcript || [];
  return el("div", { class: "card" },
    el("div", { class: "card-head" }, el("h3", {}, `${KIND_LABEL[seg.kind]} · ${fmtTime(seg.start)} – ${fmtTime(seg.end)}`),
      el("button", { class: "btn small", onclick: () => mergeWithNext(seg) }, "Merge with next part")),
    seg.kind === "talk" ? el("div", {},
      transcript.length ? el("div", {}, ...transcript.map((t) => el("p", {}, el("span", { class: "muted small" }, fmtTime(t.start) + "  "), t.text)))
        : el("p", { class: "muted" }, "No transcript (turn on transcription in Settings → Artist speech, then run Identify)."),
      el("div", { class: "row", style: "margin-top:10px" },
        el("button", { class: "btn", onclick: () => playRange("source", seg.start, 30) }, "▶ Original"),
        el("button", { class: "btn", onclick: () => playRange("vocals", seg.start, 30) }, "▶ Voice only")))
      : el("div", { class: "row" }, el("button", { class: "btn", onclick: () => playRange("source", seg.start, 30) }, "▶ Listen")));
}

async function setReference(seg, how) {
  let body = { segment: seg.id, title: seg.title, artist: seg.artist };
  if (how === "file") {
    const r = await api("/api/browse", { method: "POST", body: { kind: "file", title: "Choose the studio version" } });
    if (!r.path) return;
    body.file = r.path;
  } else if (how === "url") {
    const url = prompt("Paste a YouTube (or other) link to the studio version:");
    if (!url) return;
    body.url = url;
  } else body.clear = true;
  toast(how === "url" ? "Downloading…" : "Saving…");
  await guarded(async () => { await api(projectUrl("/reference"), { method: "POST", body }); await loadProject(state.current.id); toast("Reference updated"); });
}

// --- audio -----------------------------------------------------------------------------------

async function playRange(stem, start, seconds) {
  const player = $("#player");
  if (state.playing === `${stem}@${start}` && !player.paused) { player.pause(); state.playing = null; return; }
  player.src = `${projectUrl("/audio")}?stem=${encodeURIComponent(stem)}&start=${start.toFixed(2)}&end=${(start + seconds).toFixed(2)}`;
  state.playing = `${stem}@${start}`;
  trackPlayhead(start);
  await player.play().catch((e) => toast("Could not play: " + e.message, true));
}

async function playPreview(seg, offset) {
  toast("Rendering a 30-second preview…");
  const blob = await guarded(() => api(projectUrl("/preview"), { method: "POST", body: { segment: seg.id, start: offset, seconds: 30 } }));
  if (!blob) return;
  const player = $("#player");
  player.src = URL.createObjectURL(blob);
  state.playing = "preview";
  trackPlayhead(seg.start + offset);
  player.play();
}

function trackPlayhead(start) {
  const player = $("#player");
  player.ontimeupdate = () => { if (timeline) { timeline.playhead = start + player.currentTime; timeline.draw(); } };
  player.onended = player.onpause = () => { if (timeline) { timeline.playhead = null; timeline.draw(); } };
}

// --- export ---------------------------------------------------------------------------------

function renderExportCard(p) {
  const out = p.settings.output;
  const set = (key, value) => { p.settings.output[key] = value; saveProjectSettings(); };
  const toggle = (key, label) => el("label", { class: "check" }, el("input", { type: "checkbox", checked: !!out[key], onchange: (e) => set(key, e.target.checked) }), label);
  const outputs = p.outputs || [];
  const folders = {};
  for (const f of outputs) (folders[f.folder] ||= []).push(f);
  return el("div", { class: "card" },
    el("div", { class: "card-head" }, el("h3", {}, "Export"),
      el("div", { class: "row" },
        outputs.length ? el("button", { class: "btn ghost", onclick: () => guarded(() => api("/api/open", { method: "POST", body: { path: outputs[0].full_path.replace(/[\\/][^\\/]*$/, "").replace(/[\\/](Songs|Stems)([\\/].*)?$/, "") } })) }, "Open output folder") : null,
        el("button", { class: "btn primary", disabled: anyRunning(), onclick: () => runTask("export") }, p.exported ? "Export again" : "Export"))),
    el("div", { class: "row wrap", style: "gap:18px" },
      el("label", { class: "row" }, "Format ", el("select", { style: "width:auto", onchange: (e) => set("format", e.target.value) },
        ...["flac", "wav", "mp3"].map((f) => el("option", { value: f, selected: out.format === f }, f.toUpperCase())))),
      el("label", { class: "row" }, "Stems ", el("select", { style: "width:auto", onchange: (e) => set("stems", e.target.value) },
        ...[["per_song", "per song"], ["full_concert", "whole show"], ["both", "both"], ["none", "none"]].map(([v, l]) => el("option", { value: v, selected: out.stems === v }, l)))),
      toggle("songs", "One file per song"), toggle("full_concert", "Full concert"), toggle("vibes_edition", "Concert vibes edition"),
      toggle("transcript", "Speech subtitles"), toggle("tracklist", "Track list & cue")),
    outputs.length ? el("div", { class: "outputs" }, ...Object.entries(folders).flatMap(([folder, files]) => [
      el("div", { class: "output-folder" }, folder === "." ? "Main" : folder),
      ...files.map((f) => el("div", { class: "output-row" },
        /\.(flac|wav|mp3)$/i.test(f.name) ? el("button", { class: "icon-btn", onclick: () => { const pl = $("#player"); pl.src = f.url; pl.play(); } }, "▶") : el("span", { style: "width:26px" }),
        el("span", { class: "fname", title: f.full_path }, f.name), el("span", { class: "muted small" }, fmtSize(f.size)),
        el("a", { class: "icon-btn", href: f.url, download: f.name }, "⤓"),
        el("button", { class: "icon-btn", title: "Show in folder", onclick: () => guarded(() => api("/api/open", { method: "POST", body: { path: f.full_path } })) }, "📂")))]))
      : el("p", { class: "muted small" }, "Nothing exported yet."));
}

async function saveProjectSettings() {
  await guarded(() => api(projectUrl("/settings"), { method: "PUT", body: state.current.settings }));
}

// --- new project ------------------------------------------------------------------------------

function openNewDialog() {
  const dialog = $("#new-dialog");
  const presets = state.schema.presets;
  let chosen = "ultra";
  const cards = $("#new-presets");
  const draw = () => cards.replaceChildren(...Object.entries(presets).map(([name, text]) =>
    el("div", { class: "preset-card" + (name === chosen ? " selected" : ""), onclick: () => { chosen = name; draw(); } }, el("b", {}, name), el("span", {}, text))));
  draw();
  $("#new-browse").onclick = async () => {
    const r = await guarded(() => api("/api/browse", { method: "POST", body: { kind: "file", title: "Choose a concert recording" } }));
    if (r?.path) $("#new-path").value = r.path;
  };
  $("#new-form").onsubmit = async (e) => {
    if (e.submitter?.value !== "start") return;
    e.preventDefault();
    const defaults = await api("/api/defaults");
    defaults.identify.artist_hint = $("#new-artist").value.trim();
    defaults.segmentation.mode = $("#new-mode").value;
    defaults.speech.action = $("#new-speech").value;
    const body = { source: $("#new-path").value, name: $("#new-name").value.trim(), preset: chosen, settings: defaults, start: !anyRunning() };
    const project = await guarded(() => api("/api/projects", { method: "POST", body }));
    if (!project) return;
    dialog.close();
    if (anyRunning()) toast("Another job is running; start this analysis when it finishes.");
    await refreshProjects();
    await selectProject(project.id);
  };
  dialog.showModal();
}

// --- settings dialog ----------------------------------------------------------------------------

async function openSettings(project) {
  const dialog = $("#settings-dialog");
  const values = project ? structuredClone(project.settings) : await api("/api/defaults");
  $("#settings-title").textContent = project ? `Settings · ${project.name}` : "Default settings for new projects";
  $("#settings-as-default").parentElement.style.display = project ? "" : "none";
  const groups = [...state.schema.groups, { name: "stems", label: "Per-instrument processing" }];
  let active = groups[0].name;
  const tabs = $("#settings-tabs"), panel = $("#settings-panel");
  const drawTabs = () => tabs.replaceChildren(...groups.map((g) => el("button", { class: "tab" + (g.name === active ? " active" : ""), onclick: () => { active = g.name; drawTabs(); drawPanel(); } }, g.label)));
  const drawPanel = () => panel.replaceChildren(active === "stems" ? stemEditor(values) : groupForm(groups.find((g) => g.name === active), values[active]));
  const presetSelect = $("#settings-preset");
  presetSelect.replaceChildren(el("option", { value: "" }, "Load preset…"), ...Object.keys(state.schema.presets).map((n) => el("option", { value: n }, `Preset: ${n}`)));
  presetSelect.onchange = async () => {
    if (!presetSelect.value) return;
    const updated = await guarded(() => api("/api/preset", { method: "POST", body: { settings: values, preset: presetSelect.value } }));
    if (updated) { values.models = updated.models; active = "models"; drawTabs(); drawPanel(); toast(`Loaded the ${presetSelect.value} models`); }
    presetSelect.value = "";
  };
  $("#settings-reset").onclick = async () => {
    if (!confirm("Reset every setting to the built-in defaults?")) return;
    const fresh = await api("/api/preset", { method: "POST", body: { settings: null, preset: "ultra" } });
    Object.assign(values, fresh); drawPanel();
  };
  $("#settings-cancel").onclick = () => dialog.close();
  $("#settings-save").onclick = async () => {
    await guarded(async () => {
      if (project) {
        await api(`/api/projects/${encodeURIComponent(project.id)}/settings`, { method: "PUT", body: values });
        if ($("#settings-as-default").checked) await api("/api/defaults", { method: "PUT", body: values });
        await loadProject(project.id);
      } else await api("/api/defaults", { method: "PUT", body: values });
      dialog.close();
      toast("Settings saved");
    });
  };
  drawTabs(); drawPanel();
  dialog.showModal();
}

function groupForm(group, values) {
  const rows = [];
  for (const p of group.params) {
    if (p.type === "dict" || p.type === "list") continue;
    const row = el("div", { class: "param" });
    const refresh = () => { if (p.enable) row.classList.toggle("disabled", !values[p.enable]); };
    row.append(el("div", {}, el("div", { class: "p-label" }, p.label || p.name), p.help ? el("div", { class: "p-help" }, p.help) : null),
      el("div", { class: "p-control" }, control(p, values[p.name], (v) => { values[p.name] = v; rows.forEach((r) => r.refresh?.()); })));
    row.refresh = refresh; refresh();
    rows.push(row);
  }
  return el("div", {}, el("h2", {}, group.label), ...rows);
}

function control(p, value, onChange) {
  if (p.type === "bool") return el("input", { type: "checkbox", checked: !!value, onchange: (e) => onChange(e.target.checked) });
  if (p.choices) return el("select", { onchange: (e) => onChange(p.type === "int" ? Number(e.target.value) : e.target.value) },
    ...p.choices.map((c) => el("option", { value: c, selected: String(c) === String(value) }, c === "" ? "(none)" : String(c))));
  if ((p.type === "float" || p.type === "int") && p.min !== undefined) {
    const num = el("input", { type: "number", min: p.min, max: p.max, step: p.step, value });
    const range = el("input", { type: "range", min: p.min, max: p.max, step: p.step, value });
    range.oninput = () => { num.value = range.value; onChange(Number(range.value)); };
    num.onchange = () => { range.value = num.value; onChange(Number(num.value)); };
    return el("div", { class: "row", style: "width:100%" }, range, num, el("span", { class: "unit" }, p.unit || ""));
  }
  const input = el("input", { type: "text", value: value ?? "", list: p.kind === "model" ? "model-list" : null, onchange: (e) => onChange(e.target.value) });
  if (p.kind === "path") {
    return el("div", { class: "row", style: "width:100%" }, input, el("button", { class: "btn small", onclick: async () => {
      const r = await api("/api/browse", { method: "POST", body: { kind: "folder" } });
      if (r.path) { input.value = r.path; onChange(r.path); }
    } }, "Browse…"));
  }
  return input;
}

function stemEditor(values) {
  const names = Object.keys(values.stems);
  let current = names.includes("vocals") ? "vocals" : names[0];
  const body = el("div");
  const draw = () => {
    const prof = values.stems[current];
    const rows = state.schema.stem_params.filter((p) => p.name !== "eq").map((p) => {
      let ctrl;
      if (p.nullable) {
        const enabled = prof[p.name] !== null && prof[p.name] !== undefined;
        const inner = control(p, enabled ? prof[p.name] : p.min ?? 0, (v) => { prof[p.name] = v; });
        const box = el("input", { type: "checkbox", checked: enabled, title: "On/off", onchange: (e) => { prof[p.name] = e.target.checked ? Number(inner.querySelector("input[type=number]")?.value ?? p.min) : null; draw(); } });
        if (!enabled) inner.style.opacity = ".4";
        ctrl = el("div", { class: "row", style: "width:100%" }, box, inner);
      } else ctrl = control(p, prof[p.name], (v) => { prof[p.name] = v; });
      return el("div", { class: "param" }, el("div", {}, el("div", { class: "p-label" }, p.label), p.help ? el("div", { class: "p-help" }, p.help) : null), el("div", { class: "p-control" }, ctrl));
    });
    const eqRows = (prof.eq || []).map((band, i) => el("div", { class: "eq-row" },
      el("select", { onchange: (e) => (band.kind = e.target.value) }, ...["peak", "low_shelf", "high_shelf"].map((k) => el("option", { value: k, selected: band.kind === k }, k.replace("_", " ")))),
      el("input", { type: "number", value: band.freq_hz, step: 10, title: "Frequency (Hz)", onchange: (e) => (band.freq_hz = Number(e.target.value)) }),
      el("input", { type: "number", value: band.gain_db, step: 0.5, title: "Gain (dB)", onchange: (e) => (band.gain_db = Number(e.target.value)) }),
      el("input", { type: "number", value: band.q, step: 0.1, title: "Q", onchange: (e) => (band.q = Number(e.target.value)) }),
      el("button", { class: "icon-btn", title: "Remove band", onclick: () => { prof.eq.splice(i, 1); draw(); } }, "✕")));
    body.replaceChildren(
      el("div", { class: "param" }, el("div", { class: "p-label" }, "EQ bands"),
        el("div", {}, el("div", { class: "eq-row muted small" }, el("span", {}, "type"), el("span", {}, "Hz"), el("span", {}, "dB"), el("span", {}, "Q"), el("span", {})),
          ...eqRows, el("button", { class: "btn small", onclick: () => { (prof.eq ||= []).push({ kind: "peak", freq_hz: 1000, gain_db: 0, q: 1 }); draw(); } }, "+ Add band"))),
      ...rows);
  };
  const picker = el("select", { style: "width:auto", onchange: (e) => { current = e.target.value; draw(); } }, ...names.map((n) => el("option", { value: n, selected: n === current }, n.replace("_", " "))));
  const reset = el("button", { class: "btn small ghost", onclick: () => { values.stems[current] = structuredClone(state.schema.stem_defaults[current]); draw(); } }, "Reset this instrument");
  draw();
  return el("div", {}, el("h2", {}, "Per-instrument processing"),
    el("p", { class: "muted small" }, "The studio chain each separated instrument goes through. Levels here are relative to the stem's own loud parts, so they behave the same on every song."),
    el("div", { class: "row", style: "margin-bottom:8px" }, picker, reset), body);
}

// Suggest known model names in model fields.
document.body.append(el("datalist", { id: "model-list" }, ...[
  "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt", "UVR-MDX-NET_Crowd_HQ_1.onnx",
  "bs_roformer_vocals_resurrection_unwa.ckpt + melband_roformer_big_beta6x.ckpt", "vocals_mel_band_roformer.ckpt",
  "model_bs_roformer_ep_317_sdr_12.9755.ckpt", "BS-Roformer-SW.ckpt", "htdemucs_6s.yaml", "htdemucs_ft.yaml",
  "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", "UVR-DeEcho-DeReverb.pth", "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
  "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt", "MDX23C-DrumSep-aufr33-jarredou.ckpt", "17_HP-Wind_Inst-UVR.pth",
].map((m) => el("option", { value: m }))));

init().catch((e) => toast(e.message, true));
