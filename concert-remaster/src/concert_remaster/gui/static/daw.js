// Studio: one song's instruments as tracks, with a live mixer.
//
// The app's own audio engine plays the tracks through the sound card, sample-locked;
// this view is its control surface. Faders, pans, mutes and solos act at once, and
// the mix is saved with the song and used by the export.

import { el, api, toast, guarded, fmtTime } from "./util.js";

export const TRACKS = {
  lead_vocals: ["Lead vocal", "#f472b6"],
  vocals: ["Vocals", "#f472b6"],
  backing_vocals: ["Backing vocals", "#fb7185"],
  drums: ["Drums", "#f59e0b"],
  kick: ["Kick", "#f59e0b"],
  snare: ["Snare", "#fbbf24"],
  toms: ["Toms", "#fb923c"],
  hihat: ["Hi-hat", "#facc15"],
  ride: ["Ride", "#eab308"],
  crash: ["Crash", "#fde047"],
  bass: ["Bass", "#22d3ee"],
  guitar: ["Guitar", "#a3e635"],
  piano: ["Keys / piano", "#818cf8"],
  woodwinds: ["Winds", "#34d399"],
  other: ["Synths & other", "#c084fc"],
  crowd: ["Audience", "#94a3b8"],
  source: ["Original recording", "#e2e8f0"],
};
const trackLabel = (n) => (TRACKS[n] || [n.replace(/_/g, " ")])[0];
const trackColor = (n) => (TRACKS[n] || [null, "#a5b4fc"])[1];

export const EFFECTS = { co2: ["💨", "CO2 jet"], firework: ["🎆", "Firework"], confetti: ["🎊", "Confetti"] };

const FADER_MIN = -40;   // bottom of the fader = off
const FADER_MAX = 12;
const RULER = 24;
const LANE = 46;

const dbToGain = (db) => Math.pow(10, db / 20);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const fmtDb = (db) => (db <= FADER_MIN ? "−∞" : `${db > 0 ? "+" : ""}${db.toFixed(1)}`);
const fmtPan = (p) => (Math.abs(p) < 0.02 ? "C" : `${p < 0 ? "L" : "R"}${Math.round(Math.abs(p) * 100)}`);
const fmtClock = (s) => {
  s = Math.max(0, s);
  const m = Math.floor(s / 60), sec = s - m * 60;
  return `${m}:${sec < 10 ? "0" : ""}${sec.toFixed(1)}`;
};

// --- the app's audio engine, remote-controlled ------------------------------------------------
//
// Audio is mixed and played by the app itself (through the sound card), not by this
// window: here we only send commands (load, play, fader moves) and show its meters.

async function playerCommand(body) {
  const res = await fetch("/api/player", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not JSON */ }
    throw new Error(detail);
  }
  return res.json();
}

class NativePlayer {
  constructor(base, projectId, seg, info) {
    this.base = base;
    this.projectId = projectId;
    this.seg = seg;
    this.info = info;
    this.start = info.start;
    this.end = info.end;
    this.source = info.studio === "missing" ? "raw" : "studio";
    this.solo = new Set();
    this.playing = false;
    this.buffering = false;
    this.position = info.start;
    this.loop = false;
    this.loopRange = null;
    this.masterDb = 0;
    this.levels = {};
    this.masterLevel = -120;
    this.lastPoll = performance.now();
    this.listeners = new Set();
    this.mixQueued = false;
    this.poll = setInterval(() => this.refresh(), 70);
  }

  get names() {
    if (this.source === "original") return ["source"];
    return this.source === "studio" ? this.info.studio_tracks : this.info.raw_tracks;
  }
  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  emit() { for (const fn of this.listeners) fn(this); }

  track(name) {
    return (this.info.tracks[name] ||= { gain_db: 0, pan: 0, mute: false, auto_gain_db: 0 });
  }
  audible(name) {
    if (this.source === "original") return true;
    if (this.solo.size) return this.solo.has(name);
    return !this.track(name).mute;
  }
  trackGain(name) {
    if (!this.audible(name)) return 0;
    if (this.source === "original") return 1;
    const t = this.track(name);
    if (t.gain_db <= FADER_MIN) return 0;
    return dbToGain((this.source === "studio" ? t.auto_gain_db || 0 : 0) + t.gain_db);
  }
  mix() {
    const gains = {}, pans = {};
    for (const n of this.names) { gains[n] = this.trackGain(n); pans[n] = this.source === "original" ? 0 : this.track(n).pan || 0; }
    const monitor = this.info.monitor_gain_db[this.source === "studio" ? "studio" : "raw"] || 0;
    return { gains, pans, master: dbToGain(monitor + this.masterDb) };
  }
  ours(status) {
    const i = status?.info || {};
    return status?.kind === "song" && i.project === this.projectId && i.seg === this.seg.id && i.source === this.source;
  }

  async send(body) {
    try {
      this.apply(await playerCommand(body));
    } catch (e) {
      this.buffering = false;
      this.emit();
      toast(e.message.includes("audio") || e.message.includes("sound") ? e.message : `Playback: ${e.message}`, true);
    }
  }
  apply(status) {
    this.status = status;
    this.lastPoll = performance.now();
    if (this.ours(status)) {
      this.loaded = true;
      this.playing = status.playing;
      this.buffering = status.buffering;
      this.position = status.position;
      this.levels = status.levels || {};
      this.masterLevel = status.master_level;
    } else {
      this.loaded = false;
      this.playing = false;
      this.buffering = false;
      this.levels = {};
    }
    this.emit();
  }
  async refresh() {
    if (this.polling) return;
    if (!this.playing && !this.buffering && (this.idle = (this.idle || 0) + 1) % 6) return;  // slow down when idle
    this.polling = true;
    try { this.apply(await (await fetch("/api/player")).json()); } catch { /* app closing */ } finally { this.polling = false; }
  }

  async play(from) {
    const position = from ?? (this.position >= this.end - 0.05 ? this.start : this.position);
    this.buffering = true;
    this.emit();
    if (this.loaded && this.ours(this.status)) await this.send({ cmd: "play", position });
    else {
      await this.send({ cmd: "song", project: this.projectId, seg: this.seg.id, source: this.source, position, play: true,
        loop: this.loop, loop_range: this.loopRange, ...this.mix() });
    }
  }
  pause() { if (this.loaded) this.send({ cmd: "pause" }); }
  toggle() { if (this.playing || this.buffering) this.pause(); else this.play(); }
  rewind() { this.seek(this.loop && this.loopRange ? this.loopRange[0] : this.start); }
  stopToStart() {
    if (this.loaded) this.send({ cmd: "stop" });
    this.position = this.start;
    this.emit();
  }
  seek(t) {
    t = clamp(t, this.start, this.end);
    this.position = t;
    this.lastPoll = performance.now();
    if (this.loaded && this.ours(this.status)) this.send({ cmd: "seek", position: t });
    else this.emit();
  }
  setSource(source) {
    if (source === this.source) return;
    const was = this.playing || this.buffering, at = this.currentTime();
    this.source = source;
    this.loaded = false;
    if (was) this.play(at); else { this.position = at; this.emit(); }
  }
  setLoop() {
    if (this.loaded) this.send({ cmd: "loop", on: this.loop, start: this.loopRange?.[0], end: this.loopRange?.[1] });
  }
  applyTrack() { this.queueMix(); }
  applyAll() { this.queueMix(); }
  updateBus() { this.queueMix(); }
  queueMix() {
    // Fader drags produce many events; send at most one mix update per frame.
    if (this.mixQueued || !this.loaded) return;
    this.mixQueued = true;
    requestAnimationFrame(() => { this.mixQueued = false; if (this.loaded) playerCommand({ cmd: "mix", ...this.mix() }).catch(() => {}); });
  }

  currentTime() {
    if (!this.playing) return this.position;
    return Math.min(this.end, this.position + (performance.now() - this.lastPoll) / 1000);
  }
  level(name) { return name ? this.levels[name] ?? -120 : this.masterLevel; }
  halt() { this.pause(); }
  dispose() {
    clearInterval(this.poll);
    this.listeners.clear();
    if (this.loaded && this.playing) playerCommand({ cmd: "pause" }).catch(() => {});
  }
}

// --- studio view ----------------------------------------------------------------------------

let current = null;  // { segId, engine, frame }

export function stopStudio() {
  if (current) { current.engine.dispose(); cancelAnimationFrame(current.frame); current.abort?.abort(); current = null; }
}

document.addEventListener("keydown", (e) => {
  if (!current || e.target.closest?.("input, select, textarea, button, [contenteditable]")) return;
  if (e.code === "Space") { e.preventDefault(); current.engine.toggle(); }
  else if (e.code === "Home") { e.preventDefault(); current.engine.rewind(); }
});

/**
 * The studio card for one song.
 * app: { base, projectId, seg, busy, runTask(task, only), saveSegment(seg), copyMixToAll(seg), playPreview(seg, offset) }
 */
export function renderStudio(app) {
  const { seg } = app;
  const card = el("div", { class: "card studio" },
    el("div", { class: "studio-loading muted" }, "Loading the tracks…"));
  guarded(async () => {
    const info = await api(`${app.base}/tracks/${encodeURIComponent(seg.id)}`);
    let engine;
    if (current && current.segId === seg.id) {
      engine = current.engine;
      const studioNow = info.studio !== "missing";
      engine.info = info;
      if (engine.source === "studio" && !studioNow) engine.setSource("raw");
      else if (engine.source === "raw" && studioNow && !engine.playing && current.studioWas === "missing") engine.setSource("studio");
      engine.start = info.start; engine.end = info.end;
      engine.applyAll();
      cancelAnimationFrame(current.frame);
      engine.listeners.clear();
    } else {
      stopStudio();
      engine = new NativePlayer(app.base, app.projectId, seg, info);
    }
    current?.abort?.abort();
    current = { segId: seg.id, engine, frame: 0, studioWas: info.studio, abort: new AbortController() };
    buildStudio(card, app, engine, current.abort.signal);
  });
  return card;
}

function buildStudio(card, app, engine, signal) {
  const { seg } = app;
  const info = engine.info;
  let saveTimer = null;
  const save = () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      const gains = {}, pans = {}, mutes = {};
      for (const [name, t] of Object.entries(info.tracks)) {
        if (Math.abs(t.gain_db) > 0.001) gains[name] = t.gain_db;
        if (Math.abs(t.pan) > 0.001) pans[name] = t.pan;
        mutes[name] = !!t.mute;
      }
      seg.stem_gains = gains; seg.stem_pans = pans; seg.stem_mutes = mutes;
      app.saveSegment(seg);
    }, 500);
  };

  // --- header: source and studio tracks
  const sourceBtn = (value, label, title, disabled) => el("button", {
    class: "seg-btn" + (engine.source === value ? " on" : ""), title, disabled,
    onclick: () => { engine.setSource(value); rebuildLanes(); },
  }, label);
  const studioState = info.studio;
  const prepare = el("div", { class: "studio-banner " + studioState },
    studioState === "ready"
      ? el("span", {}, el("b", {}, "Studio tracks ready. "), "What you hear is what the export mixes: cleaned, effect-free and studio-processed.")
      : studioState === "stale"
        ? el("span", {}, el("b", {}, "Settings changed. "), "Update the studio tracks to hear the new processing.")
        : el("span", {}, el("b", {}, "Playing the raw AI stems. "), "Prepare the studio tracks to hear them cleaned of stage effects, EQ'd, compressed and balanced like the export."),
    el("span", { class: "spacer" }),
    studioState !== "ready" ? el("button", { class: "btn small primary", disabled: app.busy, onclick: () => app.runTask("studio", [seg.id]) },
      studioState === "stale" ? "Update studio tracks" : "Prepare studio tracks") : null,
    el("button", { class: "btn small ghost", disabled: app.busy, title: "Prepare the studio tracks of every song in the show", onclick: () => app.runTask("studio") }, "All songs"));

  // --- transport
  const playBtn = el("button", { class: "tp-btn play", title: "Play / pause (Space)", onclick: () => engine.toggle() });
  const clock = el("div", { class: "tp-clock" });
  const loopBtn = el("button", { class: "tp-btn" + (engine.loop ? " on" : ""), title: "Loop the song, or the range you drag on the ruler",
    onclick: () => { engine.loop = !engine.loop; loopBtn.classList.toggle("on", engine.loop); engine.setLoop(); draw(); } }, "⟲");
  const fxSelect = el("select", { class: "tp-select", title: "Stage effects in this song",
    onchange: (e) => { seg.fx_action = e.target.value || null; app.saveSegment(seg, true); } },
    ...[["", `Default (${info.default_fx_action})`], ["remove", "Remove"], ["reduce", "Reduce"], ["keep", "Keep"]]
      .map(([v, l]) => el("option", { value: v, selected: (seg.fx_action || "") === v }, l)));
  const master = el("input", { type: "range", min: -30, max: 6, step: 0.5, value: engine.masterDb, class: "tp-volume", title: "Monitor volume (not exported)",
    oninput: (e) => { engine.masterDb = Number(e.target.value); engine.updateBus(); } });
  const transport = el("div", { class: "transport" },
    el("button", { class: "tp-btn", title: "Back to the start (Home)", onclick: () => engine.rewind() }, "⏮"),
    playBtn,
    el("button", { class: "tp-btn", title: "Stop", onclick: () => engine.stopToStart() }, "⏹"),
    loopBtn, clock,
    el("div", { class: "seg-group" },
      sourceBtn("original", "Original", "The recording as it is"),
      sourceBtn("raw", "AI stems", "The separated instruments, untouched"),
      sourceBtn("studio", "Studio", "Cleaned and studio-processed, as exported", info.studio === "missing")),
    el("span", { class: "spacer" }),
    info.effects.length ? el("label", { class: "tp-label" }, `${info.effects.length} stage effect${info.effects.length > 1 ? "s" : ""}`, fxSelect) : null,
    el("label", { class: "tp-label", title: "Monitor volume" }, "🔊", master),
    el("button", { class: "btn small primary", title: "Render 30 s with the full mastering chain, from the playhead",
      onclick: () => app.playPreview(seg, Math.max(0, engine.currentTime() - seg.start)) }, "Hear the export"));

  // --- arrangement
  const heads = el("div", { class: "arr-heads" });
  const lanesCanvas = el("canvas", { class: "arr-canvas" });
  const overlay = el("canvas", { class: "arr-overlay" });
  const lanesBox = el("div", { class: "arr-lanes" }, lanesCanvas, overlay);
  const arrange = el("div", { class: "arrange" }, heads, lanesBox);
  const strips = el("div", { class: "strips" });
  const mixerTools = el("div", { class: "mixer-tools" },
    el("span", { class: "muted small" }, "Faders are relative to the automatic mix. Your mix is saved with the song and used by the export."),
    el("span", { class: "spacer" }),
    el("button", { class: "btn small ghost", onclick: () => { for (const t of Object.values(info.tracks)) { t.gain_db = 0; t.pan = 0; } engine.solo.clear(); resetMutes(); engine.applyAll(); save(); rebuildLanes(); } }, "Reset mix"),
    el("button", { class: "btn small", onclick: () => { save(); setTimeout(() => app.copyMixToAll(seg), 600); } }, "Use this mix for all songs"));

  card.replaceChildren(
    el("div", { class: "card-head" },
      el("h3", {}, `Studio · ${seg.track ? seg.track + ". " : ""}${seg.title || "Song"}${seg.artist ? " — " + seg.artist : ""}`),
      el("span", { class: "muted small" }, `${fmtTime(info.end - info.start)} · Space plays, Home rewinds, scroll zooms`)),
    prepare, transport, arrange, mixerTools, strips);

  function resetMutes() {
    for (const name of Object.keys(info.tracks)) info.tracks[name].mute = name === "crowd" ? info.tracks.crowd.mute_default ?? info.tracks.crowd.mute : false;
  }
  if (info.tracks.crowd && info.tracks.crowd.mute_default === undefined) info.tracks.crowd.mute_default = info.tracks.crowd.mute;

  // view (zoom) and waveforms
  const view = { t0: info.start, t1: info.end };
  const peaks = new Map();
  let lanes = [];

  function loadPeaks() {
    const dur = info.end - info.start;
    const rate = clamp(Math.round(3000 / Math.max(1, dur)), 4, 50);
    for (const name of lanes) {
      const src = engine.source === "studio" ? "studio" : "raw";
      const key = `${src}|${name}`;
      if (peaks.has(key)) continue;
      peaks.set(key, null);
      api(`${app.base}/peaks?stem=${encodeURIComponent(name)}&start=${info.start}&end=${info.end}&rate=${rate}&source=${src}&seg=${encodeURIComponent(seg.id)}`)
        .then((r) => { peaks.set(key, r); draw(); })
        .catch(() => peaks.delete(key));
    }
  }

  function rebuildLanes() {
    lanes = engine.names;
    heads.replaceChildren(el("div", { class: "arr-ruler-head" }, engine.source === "original" ? "Recording" : `${lanes.length} tracks`),
      ...lanes.map((name) => trackHead(name)));
    strips.replaceChildren(...lanes.map((name) => channelStrip(name)), masterStrip());
    lanesBox.style.height = `${RULER + lanes.length * LANE}px`;
    for (const b of transport.querySelectorAll(".seg-btn")) b.classList.toggle("on", b.textContent === { original: "Original", raw: "AI stems", studio: "Studio" }[engine.source]);
    loadPeaks();
    resize();
  }

  function toggleMute(name) {
    const t = engine.track(name);
    t.mute = !t.mute;
    engine.applyTrack(name); save(); refreshControls(); draw();
  }
  function toggleSolo(name) {
    if (engine.solo.has(name)) engine.solo.delete(name); else engine.solo.add(name);
    engine.applyAll(); refreshControls(); draw();
  }

  function trackHead(name) {
    const t = engine.track(name);
    const original = engine.source === "original";
    return el("div", { class: "arr-head", "data-track": name, style: `--c:${trackColor(name)}` },
      el("i", { class: "arr-color" }),
      el("div", { class: "arr-name", title: t.empty ? "The AI found little here in this song" : "" }, trackLabel(name), t.empty ? el("span", { class: "muted" }, " · faint") : null),
      original ? null : el("div", { class: "arr-btns" },
        el("button", { class: "ms m", title: "Mute", onclick: () => toggleMute(name) }, "M"),
        el("button", { class: "ms s", title: "Solo", onclick: () => toggleSolo(name) }, "S")));
  }

  function channelStrip(name) {
    const t = engine.track(name);
    const original = engine.source === "original";
    const readout = el("input", { class: "st-db", type: "text", value: fmtDb(t.gain_db), title: "Type a level in dB",
      onchange: (e) => { const v = parseFloat(String(e.target.value).replace("−", "-")); if (isFinite(v)) setGain(clamp(v, FADER_MIN, FADER_MAX)); } });
    const fader = el("input", { class: "st-fader", type: "range", min: FADER_MIN, max: FADER_MAX, step: 0.5, value: t.gain_db, disabled: original,
      title: "Level (double-click for 0 dB)", oninput: (e) => setGain(Number(e.target.value)), ondblclick: () => setGain(0) });
    const panLabel = el("span", { class: "st-pan-val" }, fmtPan(t.pan));
    const pan = el("input", { class: "st-pan", type: "range", min: -1, max: 1, step: 0.02, value: t.pan, disabled: original,
      title: "Pan (double-click to centre)", oninput: (e) => setPan(Number(e.target.value)), ondblclick: () => { pan.value = 0; setPan(0); } });
    const meter = el("canvas", { class: "st-meter", width: 10, height: 150 });
    function setGain(v) {
      t.gain_db = v; fader.value = v; readout.value = fmtDb(v);
      engine.applyTrack(name); save(); draw();
    }
    function setPan(v) { t.pan = Math.abs(v) < 0.03 ? 0 : v; panLabel.textContent = fmtPan(t.pan); engine.applyTrack(name); save(); }
    const strip = el("div", { class: "strip", "data-track": name, style: `--c:${trackColor(name)}` },
      el("div", { class: "st-pan-row" }, pan, panLabel),
      el("div", { class: "st-body" }, el("div", { class: "st-scale" }, ...["+12", "0", "-12", "-24", "-40"].map((s) => el("span", {}, s))), fader, meter),
      readout,
      original ? el("div", { class: "arr-btns" }) : el("div", { class: "arr-btns" },
        el("button", { class: "ms m", title: "Mute", onclick: () => toggleMute(name) }, "M"),
        el("button", { class: "ms s", title: "Solo", onclick: () => toggleSolo(name) }, "S")),
      el("div", { class: "st-name", title: trackLabel(name) }, trackLabel(name)));
    strip.meter = meter;
    return strip;
  }

  function masterStrip() {
    const meter = el("canvas", { class: "st-meter wide", width: 16, height: 150 });
    const strip = el("div", { class: "strip master" },
      el("div", { class: "st-pan-row muted small" }, "monitor"),
      el("div", { class: "st-body" }, el("div", { class: "st-scale" }, ...["0", "-6", "-12", "-24", "-48"].map((s) => el("span", {}, s))), meter),
      el("div", { class: "st-db muted" }, "limiter"),
      el("div", { class: "arr-btns" }),
      el("div", { class: "st-name" }, "Master"));
    strip.meter = meter;
    strip.dataset.track = "";
    return strip;
  }

  function refreshControls() {
    for (const node of card.querySelectorAll("[data-track]")) {
      const name = node.dataset.track;
      if (!name) continue;
      const t = engine.track(name);
      node.querySelector(".ms.m")?.classList.toggle("on", !!t.mute);
      node.querySelector(".ms.s")?.classList.toggle("on", engine.solo.has(name));
      node.classList.toggle("silent", !engine.audible(name));
    }
  }

  // --- drawing
  let w = 0, h = 0, dpr = 1;
  function resize() {
    const r = lanesBox.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    w = r.width; h = RULER + lanes.length * LANE;
    for (const c of [lanesCanvas, overlay]) { c.width = Math.max(1, w * dpr); c.height = Math.max(1, h * dpr); c.style.height = `${h}px`; }
    draw();
  }
  const xAt = (t) => ((t - view.t0) / (view.t1 - view.t0)) * w;
  const timeAt = (x) => view.t0 + (x / w) * (view.t1 - view.t0);

  function draw() {
    const ctx = lanesCanvas.getContext("2d");
    if (!ctx || !w) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const span = view.t1 - view.t0;
    // ruler
    ctx.fillStyle = "#11141c"; ctx.fillRect(0, 0, w, RULER);
    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300].find((s) => span / s < 14) || 600;
    ctx.font = "11px Segoe UI, system-ui, sans-serif";
    for (let t = Math.ceil((view.t0 - info.start) / step) * step; info.start + t < view.t1; t += step) {
      const x = xAt(info.start + t);
      ctx.fillStyle = "#2a3040"; ctx.fillRect(x, RULER, 1, h - RULER);
      ctx.fillStyle = "#8b93a7"; ctx.fillRect(x, RULER - 6, 1, 6);
      ctx.fillText(fmtTime(t), x + 3, 14);
    }
    // loop range
    if (engine.loopRange) {
      const [a, b] = engine.loopRange;
      ctx.fillStyle = engine.loop ? "rgba(34,211,238,.28)" : "rgba(148,163,184,.18)";
      ctx.fillRect(xAt(a), 0, xAt(b) - xAt(a), RULER);
      ctx.fillStyle = engine.loop ? "rgba(34,211,238,.06)" : "rgba(148,163,184,.04)";
      ctx.fillRect(xAt(a), RULER, xAt(b) - xAt(a), h - RULER);
    }
    // stage effects
    for (const e of info.effects) {
      const x0 = xAt(e.start), x1 = Math.max(x0 + 2, xAt(e.end));
      if (x1 < 0 || x0 > w) continue;
      ctx.fillStyle = "rgba(251,146,60,.10)"; ctx.fillRect(x0, RULER, x1 - x0, h - RULER);
      ctx.fillStyle = "rgba(251,146,60,.55)"; ctx.fillRect(x0, RULER - 3, x1 - x0, 3);
      ctx.font = "13px Segoe UI Emoji, Apple Color Emoji, sans-serif";
      ctx.fillText(EFFECTS[e.kind]?.[0] || "✦", x0 + 1, RULER - 6);
    }
    // lanes
    const src = engine.source === "studio" ? "studio" : "raw";
    let norm = 0;
    const lanePeaks = lanes.map((name) => peaks.get(`${src}|${name}`));
    lanes.forEach((name, i) => {
      const p = lanePeaks[i];
      if (!p?.peaks?.length) return;
      const t = engine.track(name);
      const g = engine.source === "studio" ? dbToGain(t.auto_gain_db || 0) : 1;
      for (const v of p.peaks) if (v * g > norm) norm = v * g;
    });
    norm = norm || 1;
    lanes.forEach((name, i) => {
      const y0 = RULER + i * LANE, mid = y0 + LANE / 2;
      ctx.fillStyle = i % 2 ? "#12151d" : "#141822";
      ctx.fillRect(0, y0, w, LANE);
      ctx.fillStyle = "#1f2430"; ctx.fillRect(0, y0 + LANE - 1, w, 1);
      const p = lanePeaks[i];
      if (!p?.peaks?.length) {
        ctx.fillStyle = "#3a4152"; ctx.fillText(p === null ? "loading…" : "", 8, mid + 4);
        return;
      }
      const t = engine.track(name);
      const auto = engine.source === "studio" ? t.auto_gain_db || 0 : 0;
      const gain = engine.source === "original" ? 1 : t.gain_db <= FADER_MIN ? 0 : dbToGain(auto + t.gain_db);
      ctx.globalAlpha = engine.audible(name) ? 0.95 : 0.25;
      ctx.fillStyle = trackColor(name);
      const data = p.peaks, rate = p.rate;
      for (let x = 0; x < w; x++) {
        const a = Math.floor((timeAt(x) - info.start) * rate), b = Math.max(a + 1, Math.floor((timeAt(x + 1) - info.start) * rate));
        let m = 0;
        for (let j = Math.max(0, a); j < b && j < data.length; j++) if (data[j] > m) m = data[j];
        const hgt = Math.max(0.5, Math.min(1, (m * gain) / norm) * (LANE / 2 - 4));
        ctx.fillRect(x, mid - hgt, 1, hgt * 2);
      }
      ctx.globalAlpha = 1;
    });
    drawOverlay();
  }

  function drawOverlay() {
    const ctx = overlay.getContext("2d");
    if (!ctx || !w) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const x = xAt(engine.currentTime());
    ctx.fillStyle = "#f472b6";
    ctx.fillRect(x - 0.75, 0, 1.5, h);
    ctx.beginPath(); ctx.moveTo(x - 6, 0); ctx.lineTo(x + 6, 0); ctx.lineTo(x, 8); ctx.fill();
  }

  function drawMeters() {
    for (const strip of strips.children) {
      const c = strip.meter;
      if (!c) continue;
      const db = engine.level(strip.dataset.track || null);
      const ctx = c.getContext("2d");
      const floor = strip.classList.contains("master") ? -48 : -40, top = strip.classList.contains("master") ? 0 : 12;
      const fill = clamp((db - floor) / (top - floor), 0, 1);
      const hold = (strip.hold = Math.max((strip.hold || 0) * 0.97, fill));
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.fillStyle = "#0b0d12"; ctx.fillRect(0, 0, c.width, c.height);
      const grad = ctx.createLinearGradient(0, c.height, 0, 0);
      grad.addColorStop(0, "#22c55e"); grad.addColorStop(0.7, "#eab308"); grad.addColorStop(0.9, "#ef4444");
      ctx.fillStyle = grad;
      ctx.fillRect(0, c.height * (1 - fill), c.width, c.height * fill);
      ctx.fillStyle = "#e7e9ee"; ctx.fillRect(0, c.height * (1 - hold), c.width, 1);
    }
  }

  function updateTransport() {
    playBtn.textContent = engine.buffering ? "…" : engine.playing ? "⏸" : "▶";
    playBtn.classList.toggle("on", engine.playing);
    clock.replaceChildren(el("b", {}, fmtClock(engine.currentTime() - info.start)), el("span", {}, ` / ${fmtClock(info.end - info.start)}`));
  }

  function loop() {
    updateTransport();
    drawOverlay();
    drawMeters();
    if (current?.engine === engine) current.frame = requestAnimationFrame(loop);
  }

  // --- interaction on the lanes
  let drag = null;
  overlay.addEventListener("mousedown", (e) => {
    const x = e.offsetX, y = e.offsetY;
    drag = y < RULER ? { kind: "loop", from: timeAt(x), x } : { kind: "pan", x, t0: view.t0, t1: view.t1, moved: false };
  });
  window.addEventListener("mousemove", (e) => {
    if (!drag) return;
    const r = overlay.getBoundingClientRect();
    const x = e.clientX - r.left;
    if (Math.abs(x - drag.x) > 3) drag.moved = true;
    if (drag.kind === "loop") {
      if (drag.moved) { const a = drag.from, b = timeAt(x); engine.loopRange = [Math.max(info.start, Math.min(a, b)), Math.min(info.end, Math.max(a, b))]; draw(); }
    } else if (drag.moved) {
      const span = drag.t1 - drag.t0, dt = ((x - drag.x) / w) * span;
      view.t0 = clamp(drag.t0 - dt, info.start, info.end - span); view.t1 = view.t0 + span; draw();
    }
  }, { signal });
  window.addEventListener("mouseup", (e) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    if (!d.moved) {
      if (d.kind === "loop" && engine.loopRange) { engine.loopRange = null; engine.setLoop(); draw(); }
      const r = overlay.getBoundingClientRect();
      engine.seek(timeAt(e.clientX - r.left));
    } else if (d.kind === "loop" && engine.loopRange && engine.loopRange[1] - engine.loopRange[0] > 0.5) {
      engine.loop = true; loopBtn.classList.add("on"); engine.setLoop(); engine.seek(engine.loopRange[0]);
    }
  }, { signal });
  overlay.addEventListener("wheel", (e) => {
    e.preventDefault();
    const around = timeAt(e.offsetX), factor = e.deltaY > 0 ? 1.25 : 0.8;
    const span = clamp((view.t1 - view.t0) * factor, 4, info.end - info.start);
    view.t0 = clamp(around - (around - view.t0) * (span / (view.t1 - view.t0)), info.start, info.end - span);
    view.t1 = view.t0 + span;
    draw();
  }, { passive: false });
  overlay.addEventListener("dblclick", () => { view.t0 = info.start; view.t1 = info.end; draw(); });

  const observer = new ResizeObserver(() => resize());
  observer.observe(lanesBox);
  signal.addEventListener("abort", () => observer.disconnect());
  engine.on(() => { updateTransport(); refreshControls(); });
  rebuildLanes();
  refreshControls();
  current.frame = requestAnimationFrame(loop);
}

/** Small stage-effect markers on the whole-show timeline. */
export function drawShowEffects(ctx, effects, xAt, top) {
  ctx.font = "11px Segoe UI Emoji, Apple Color Emoji, sans-serif";
  for (const e of effects || []) {
    const x = xAt(e.start);
    ctx.fillStyle = "rgba(251,146,60,.8)";
    ctx.fillRect(x, top, Math.max(2, xAt(e.end) - x), 3);
  }
}
