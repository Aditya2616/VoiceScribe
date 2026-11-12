import { download, mmss } from "./export.js";

const $ = (s) => document.querySelector(s);
const audio = $("#audio");
const canvas = $("#wave");
const ctx = canvas.getContext("2d");
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

const S = {
  mode: "demo",
  backend: null, // health payload when the local server answers
  meeting: null,
  peaks: [],
  pps: 50,
  dur: 0,
  sents: [], // {el, start, end}
  words: [], // {el, start, end}
  items: [], // {el, kind, start, end, shown, data}
  kept: new Set(),
  wi: 0, // number of words currently revealed
  si: -1, // active sentence
  sn: 0, // number of sentences whose first word has been reached
  lastT: 0,
  clock: { at: 0, wall: 0 },
  playing: false,
  visible: true,
  summaryShown: false,
  objectUrl: null,
  pendingFile: null,
  userScrollUntil: 0,
};

/* ---------- helpers ---------- */
const fmtTime = (s) => mmss(Math.max(0, s));
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
function fmtDate(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  const dt = new Date(Date.UTC(y, m - 1, d));
  return `${DAYS[dt.getUTCDay()]} ${d} ${MONTHS[m - 1]} ${y}`;
}
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
function animateIn(node) {
  if (reduce || !node.animate) return;
  node.animate(
    [{ opacity: 0, transform: "translateY(16px) scale(.985)" }, { opacity: 1, transform: "translateY(0) scale(1)" }],
    { duration: 560, easing: "cubic-bezier(.19,1,.22,1)" },
  );
}
function setNote(text, err = false) {
  const n = $("#intake-note");
  n.textContent = text || "";
  n.classList.toggle("err", err);
}

/* ---------- loading a meeting ---------- */
function resetPlayback() {
  audio.pause();
  S.playing = false;
  S.kept.clear();
  S.wi = 0;
  S.si = -1;
  S.sn = 0;
  S.lastT = 0;
  S.summaryShown = false;
  updatePlayButton();
}

function loadMeeting(meeting, peaks, pps, audioUrl, opts = {}) {
  resetPlayback();
  S.meeting = meeting;
  S.peaks = peaks;
  S.pps = pps;
  S.dur = meeting.transcript.duration;
  if (S.objectUrl && S.objectUrl !== audioUrl) URL.revokeObjectURL(S.objectUrl);
  S.objectUrl = opts.objectUrl ? audioUrl : null;
  audio.src = audioUrl;
  audio.load();
  buildTranscript(meeting);
  buildItems(meeting);
  $("#t-dur").textContent = fmtTime(S.dur);
  $("#scrub").disabled = true;
  $("#play").disabled = true;
  $("#source-tag").textContent = meeting.synthetic ? "Synthetic speech · Kokoro-82M" : "Your audio · processed locally";
  $("#source-tag").className = "tag " + (meeting.synthetic ? "tag-lime" : "");
  const tm = meeting.timings || {};
  const parts = [meeting.transcript.model.replace("faster-whisper/", "whisper ")];
  if (tm.audio_s) parts.push(`${tm.audio_s.toFixed(0)} s audio`);
  // The recorded demo shows a speed figure only when it was measured on a quiet machine (see scripts/rerecord_demo.py).
  if (tm.asr_s != null && (!opts.recorded || tm.quiet_machine)) parts.push(`asr ${tm.asr_s.toFixed(1)} s (${tm.rtf}x real time, cpu int8)`);
  parts.push(`extractor: ${meeting.extractor}`);
  if (tm.diarize_s > 0) parts.push("speakers identified");
  if (meeting.participants && meeting.participants.length) parts.push(`participants given: ${meeting.participants.join(", ")}`);
  $("#run-info").textContent = $("#run-info-m").textContent = (opts.recorded ? "recorded run · " : "") + parts.join(" · ");
  if (opts.recorded) S.recordedModel = meeting.transcript.model.replace("faster-whisper/", "whisper ");
  renderEnv();
  $("#credits").textContent = meeting.synthetic
    ? "Demo audio is synthetic speech generated with Kokoro-82M (Apache-2.0). The transcript, word timestamps and items are the output of a real run of this code on that audio. This page replays that recorded run; it is not live."
    : "Your audio was processed on the machine running this server and was not stored.";
  $("#tr-meta").textContent = `${meeting.transcript.sentences.length} sentences`;
  $("#it-meta").textContent = `extractor: ${meeting.extractor}`;
  $("#sum-kind").textContent = meeting.summary ? meeting.summary.kind : "";
  $("#summary").textContent = "";
  $("#summary").hidden = true;
  $("#summary-empty").hidden = false;
  $("#summary-empty").textContent = meeting.summary ? "The summary appears when the meeting ends." : "No summary produced.";
  $("#wave-msg").hidden = true;
  sync(0, true);
  draw(0);
}

function buildTranscript(m) {
  const root = $("#transcript");
  root.replaceChildren();
  S.sents = [];
  S.words = [];
  if (!m.transcript.sentences.length) {
    root.append(el("div", "tr-empty mono", "No speech was recognised in this audio."));
    return;
  }
  m.transcript.sentences.forEach((s, i) => {
    const row = el("div", "sent");
    row.dataset.i = i;
    row.append(el("span", "ts", fmtTime(s.start)));
    const body = el("div", "txt");
    if (s.speaker) body.append(el("span", "who", s.speaker));
    s.words.forEach((w, k) => {
      const span = el("span", "w", w.text);
      span.dataset.k = S.words.length;
      S.words.push({ el: span, start: w.start, end: w.end });
      body.append(span);
      if (k < s.words.length - 1) body.append(document.createTextNode(" "));
    });
    row.append(body);
    root.append(row);
    S.sents.push({ el: row, start: s.start, end: s.end });
  });
}

function buildItems(m) {
  const acts = $("#actions");
  const decs = $("#decisions");
  acts.replaceChildren();
  decs.replaceChildren();
  S.items = [];
  const make = (kind, data) => {
    const li = el("li");
    const card = el("button", "card" + (kind === "dec" ? " dec" : ""));
    card.type = "button";
    const head = el("div", "head");
    if (kind === "act") head.append(el("span", "chip" + (data.owner ? "" : " none"), data.owner || "unassigned"));
    else head.append(el("span", "chip", "decision"));
    head.append(el("span", "chip time", fmtTime(data.source.start)));
    card.append(head);
    card.append(el("div", "task", kind === "act" ? data.task : data.text));
    if (kind === "act") {
      const due = el("div", "due" + (data.deadline_date ? "" : " none"));
      if (data.deadline_date) {
        due.append(el("span", null, "Due " + fmtDate(data.deadline_date)));
        if (data.deadline_text) due.append(el("span", "phrase", `“${data.deadline_text}”`));
      } else due.append(el("span", null, "no deadline mentioned"));
      card.append(due);
    }
    card.append(el("div", "meth", data.method));
    li.append(card);
    li.hidden = true;
    (kind === "act" ? acts : decs).append(li);
    const item = { el: li, kind, start: data.source.start, end: data.source.end, shown: false, data };
    card.addEventListener("click", () => jumpToItem(item));
    S.items.push(item);
  };
  m.action_items.forEach((a) => make("act", a));
  m.decisions.forEach((d) => make("dec", d));
  S.items.sort((a, b) => a.end - b.end);
  updateCounts();
}

function updateCounts() {
  const a = S.items.filter((i) => i.kind === "act" && i.shown).length;
  const d = S.items.filter((i) => i.kind === "dec" && i.shown).length;
  $("#n-act").textContent = a;
  $("#n-dec").textContent = d;
  $("#actions-empty").hidden = a > 0;
  $("#decisions-empty").hidden = d > 0;
}

/* ---------- clock ---------- */
function now() {
  const at = audio.currentTime || 0;
  const wall = performance.now();
  if (at !== S.clock.at) {
    S.clock.at = at;
    S.clock.wall = wall;
  }
  let t = at;
  if (S.playing && !audio.paused) t = at + ((wall - S.clock.wall) / 1000) * audio.playbackRate;
  return Math.min(S.dur || t, Math.max(0, t));
}

function bsearch(arr, t, key) {
  // number of entries with arr[i][key] <= t
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid][key] <= t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/* ---------- state from time ---------- */
function sync(t, instant = false) {
  // words
  const n = bsearch(S.words, t, "start");
  if (n !== S.wi) {
    const lo = Math.min(n, S.wi), hi = Math.max(n, S.wi);
    for (let i = lo; i < hi; i++) S.words[i].el.classList.toggle("on", i < n);
    S.wi = n;
  }
  // a row's timestamp and speaker label appear with its first word; the row keeps its space so nothing jumps
  const sn = bsearch(S.sents, t, "start");
  if (sn !== S.sn) {
    const lo = Math.min(sn, S.sn), hi = Math.max(sn, S.sn);
    for (let i = lo; i < hi; i++) S.sents[i].el.classList.toggle("started", i < sn);
    S.sn = sn;
  }
  S.words.forEach((w, i) => {
    if (w.cur) { w.el.classList.remove("cur"); w.cur = false; }
    if (i === n - 1 && t < w.end + 0.25) { w.el.classList.add("cur"); w.cur = true; }
  });
  // active sentence
  const si = Math.max(-1, bsearch(S.sents, t, "start") - 1);
  if (si !== S.si) {
    if (S.si >= 0 && S.sents[S.si]) S.sents[S.si].el.classList.remove("active");
    if (si >= 0) S.sents[si].el.classList.add("active");
    S.si = si;
    if (si >= 0) followSentence(S.sents[si].el, instant);
  }
  // items appear when their source sentence has ended
  let changed = false;
  for (const it of S.items) {
    const want = t >= it.end - 1e-6 || S.kept.has(it);
    if (want === it.shown) continue;
    it.shown = want;
    changed = true;
    it.el.hidden = !want;
    if (want && !instant) {
      animateIn(it.el);
      flashSource(it);
    }
  }
  if (changed) updateCounts();
  // summary at the end of the meeting
  const atEnd = S.dur > 0 && t >= S.dur - 0.4;
  if (atEnd !== S.summaryShown && S.meeting && S.meeting.summary) {
    S.summaryShown = atEnd;
    const p = $("#summary");
    p.textContent = atEnd ? S.meeting.summary.text : "";
    p.hidden = !atEnd;
    $("#summary-empty").hidden = atEnd;
    if (atEnd && !instant) animateIn(p);
  }
  S.lastT = t;
}

function followSentence(row, instant) {
  if (performance.now() < S.userScrollUntil) return;
  const box = $("#transcript");
  const target = row.offsetTop - box.clientHeight * 0.35;
  if (reduce || instant || !box.scrollTo) box.scrollTop = target;
  else box.scrollTo({ top: target, behavior: "smooth" });
}

function flashSource(it) {
  const ids = it.data.source.sentence_ids;
  ids.forEach((i) => {
    const row = S.sents[i] && S.sents[i].el;
    if (!row) return;
    row.classList.add("src");
    if (reduce) return;
    row.classList.remove("flash");
    void row.offsetWidth;
    row.classList.add("flash");
  });
}

/* ---------- waveform ---------- */
let dpr = 1, cw = 0, ch = 0;
function fitCanvas() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  cw = canvas.clientWidth;
  ch = canvas.clientHeight;
  canvas.width = Math.round(cw * dpr);
  canvas.height = Math.round(ch * dpr);
}
const LIME = "215,255,63";
function draw(t) {
  if (!cw) fitCanvas();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const px = cw * 0.38;
  const span = cw < 600 ? 8 : 14; // seconds on screen
  const pps = cw / span;
  const mid = ch / 2;
  const peaks = S.peaks;
  const step = 3;
  for (let x = 0; x < cw; x += step) {
    const tt = t + (x - px) / pps;
    if (tt < 0 || tt > S.dur) continue;
    const a = (peaks[Math.floor(tt * S.pps)] || 0) / 255;
    const h = Math.max(2, Math.pow(a, 0.7) * ch * 0.74);
    const past = x < px;
    const fade = past ? Math.max(0.25, 1 - (px - x) / (px * 1.3)) : Math.max(0.1, 0.34 - (x - px) / (cw * 2));
    ctx.fillStyle = past ? `rgba(${LIME},${0.95 * fade})` : `rgba(237,236,230,${fade})`;
    ctx.fillRect(x, mid - h / 2, 2, h);
  }
  // seconds ruler
  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textBaseline = "alphabetic";
  const first = Math.ceil(t - px / pps), last = Math.floor(t + (cw - px) / pps);
  for (let s = Math.max(0, first); s <= Math.min(Math.floor(S.dur), last); s++) {
    const x = px + (s - t) * pps;
    const major = s % 5 === 0;
    ctx.fillStyle = `rgba(237,236,230,${major ? 0.35 : 0.14})`;
    ctx.fillRect(x, ch - (major ? 12 : 6), 1, major ? 12 : 6);
    if (major) ctx.fillText(fmtTime(s), x + 4, ch - 4);
  }
  // sentence starts and extraction markers
  for (const s of S.sents) {
    const x = px + (s.start - t) * pps;
    if (x < -4 || x > cw + 4) continue;
    ctx.fillStyle = "rgba(237,236,230,.22)";
    ctx.fillRect(x, 6, 1, 7);
  }
  for (const it of S.items) {
    const x = px + (it.end - t) * pps;
    if (x < -8 || x > cw + 8) continue;
    const on = t >= it.end - 1e-6;
    ctx.save();
    ctx.translate(x, 12);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = it.kind === "dec" ? (on ? "#edece6" : "transparent") : on ? `rgb(${LIME})` : "transparent";
    ctx.strokeStyle = it.kind === "dec" ? "rgba(237,236,230,.7)" : `rgba(${LIME},.85)`;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.rect(-4, -4, 8, 8);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
  // playhead
  ctx.fillStyle = `rgb(${LIME})`;
  ctx.shadowColor = `rgba(${LIME},.55)`;
  ctx.shadowBlur = 10;
  ctx.fillRect(px - 1, 0, 2, ch);
  ctx.shadowBlur = 0;
  ctx.beginPath();
  ctx.moveTo(px - 6, 0);
  ctx.lineTo(px + 6, 0);
  ctx.lineTo(px, 8);
  ctx.closePath();
  ctx.fill();
}

/* ---------- loop ---------- */
let raf = 0;
function frame() {
  raf = 0;
  const t = now();
  const jump = Math.abs(t - S.lastT) > 1.2;
  sync(t, jump);
  draw(t);
  updateTransport(t);
  if (S.playing && S.visible) raf = requestAnimationFrame(frame);
}
function kick() {
  if (!raf) raf = requestAnimationFrame(frame);
}
function updateTransport(t) {
  $("#t-cur").textContent = fmtTime(t);
  const scrub = $("#scrub");
  if (S.dur) {
    const v = Math.round((t / S.dur) * 1000);
    if (!scrub.matches(":active")) scrub.value = v;
    scrub.style.setProperty("--p", `${v / 10}%`);
  }
}
function updatePlayButton() {
  const b = $("#play");
  b.classList.toggle("playing", S.playing);
  b.setAttribute("aria-label", S.playing ? "Pause" : "Play");
}

function seek(t, { keepItems = false } = {}) {
  t = Math.max(0, Math.min(S.dur, t));
  if (!keepItems) S.kept.clear();
  audio.currentTime = t;
  S.clock.at = t;
  S.clock.wall = performance.now();
  sync(t, true);
  draw(t);
  updateTransport(t);
}

function jumpToItem(it) {
  S.kept.clear();
  S.kept.add(it);
  const t = Math.max(0, it.start - 0.15);
  seek(t, { keepItems: true });
  S.userScrollUntil = 0;
  it.data.source.sentence_ids.forEach((i) => {
    const row = S.sents[i] && S.sents[i].el;
    if (!row) return;
    row.classList.add("src");
    row.classList.remove("flash");
    void row.offsetWidth;
    row.classList.add("flash");
  });
  play();
}

/* Phone: the stage is pinned at the top and the player bar at the bottom. Size the transcript to the space
   between them and bring it into view on the first tap of play, so no sentence sits under the bar. */
const phone = window.matchMedia("(max-width: 759px)");
function measureChrome() {
  const root = document.documentElement.style;
  root.setProperty("--stage-h", `${Math.ceil($(".stage").getBoundingClientRect().height)}px`);
  root.setProperty("--bar-h", `${Math.ceil($("#transport").getBoundingClientRect().height)}px`);
}
new ResizeObserver(measureChrome).observe($(".stage"));
new ResizeObserver(measureChrome).observe($("#transport"));
function bringTranscriptIntoView() {
  if (!phone.matches) return;
  const panel = $(".transcript");
  const top = panel.getBoundingClientRect().top;
  const want = Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--stage-h")) + 8;
  if (Math.abs(top - want) > 6) window.scrollBy({ top: top - want, behavior: reduce ? "auto" : "smooth" });
}

async function play() {
  try {
    if (audio.ended || S.lastT >= S.dur - 0.05) seek(0);
    await audio.play();
    bringTranscriptIntoView();
  } catch (e) {
    setNote("The browser blocked playback. Tap play again.", true);
  }
}

audio.addEventListener("play", () => { S.playing = true; S.clock.wall = performance.now(); updatePlayButton(); kick(); });
audio.addEventListener("pause", () => { S.playing = false; updatePlayButton(); kick(); });
audio.addEventListener("ended", () => { S.playing = false; updatePlayButton(); sync(S.dur, true); draw(S.dur); });
audio.addEventListener("loadedmetadata", () => {
  $("#play").disabled = false;
  $("#scrub").disabled = false;
});
audio.addEventListener("error", () => {
  $("#wave-msg").hidden = false;
  $("#wave-msg").textContent = "This browser could not play the audio file.";
});
audio.addEventListener("seeked", () => kick());
audio.addEventListener("ratechange", () => { $("#rate").textContent = `${audio.playbackRate}x`; });

$("#play").addEventListener("click", () => (audio.paused ? play() : audio.pause()));
$("#rate").addEventListener("click", () => {
  const rates = [1, 1.5, 2, 0.75];
  audio.playbackRate = rates[(rates.indexOf(audio.playbackRate) + 1) % rates.length];
});
$("#scrub").addEventListener("input", (e) => seek((e.target.value / 1000) * S.dur));

// transcript: click a word or sentence to seek; remember manual scrolling so auto-follow does not fight it
$("#transcript").addEventListener("click", (e) => {
  const row = e.target.closest(".sent");
  if (!row) return;
  const w = e.target.closest(".w");
  const i = Number(row.dataset.i);
  const t = w ? S.words[Number(w.dataset.k)].start : S.sents[i].start;
  seek(Math.max(0, t - 0.05));
  S.userScrollUntil = 0;
  play();
});
["wheel", "touchmove"].forEach((ev) =>
  $("#transcript").addEventListener(ev, () => { S.userScrollUntil = performance.now() + 4000; }, { passive: true }),
);

// waveform: drag horizontally to scrub
{
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    if (!S.dur) return;
    drag = { x: e.clientX, t: now(), id: e.pointerId, moved: false };
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    const dx = e.clientX - drag.x;
    if (!drag.moved && Math.abs(dx) < 6) return;
    if (!drag.moved) { drag.moved = true; canvas.setPointerCapture(e.pointerId); }
    const pps = cw / (cw < 600 ? 8 : 14);
    seek(drag.t - dx / pps);
  });
  const end = (e) => { if (drag && e.pointerId === drag.id) drag = null; };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
}

new ResizeObserver(() => { fitCanvas(); draw(now()); }).observe(canvas);
new IntersectionObserver((entries) => {
  S.visible = entries[0].isIntersecting;
  if (S.visible) kick();
}, { threshold: 0 }).observe(canvas);
document.addEventListener("visibilitychange", () => { if (!document.hidden) kick(); });

/* ---------- exports ---------- */
document.querySelectorAll(".exports [data-fmt]").forEach((b) =>
  b.addEventListener("click", () => S.meeting && download(S.meeting, b.dataset.fmt)),
);

/* ---------- modes ---------- */
function setMode(mode) {
  S.mode = mode;
  renderEnv();
  document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  const intake = $("#intake");
  intake.hidden = mode === "demo";
  $("#upload-ui").hidden = mode !== "upload";
  $("#record-ui").hidden = mode !== "record";
  $("#fields").hidden = mode === "demo";
  $("#pipe").hidden = true;
  if (mode === "demo") {
    loadDemo();
    return;
  }
  audio.pause();
  if (!S.backend) {
    setNote("Upload and record work when you run VoiceScribe on your own machine (`voicescribe serve`, then open http://127.0.0.1:8107). This page only replays the bundled recorded run.", true);
    $("#process").disabled = true;
    return;
  }
  setNote(mode === "upload" ? "Audio is processed on this machine and deleted afterwards. Limit: " + S.backend.max_upload_mb + " MB, " + S.backend.max_minutes + " minutes." : "Press record, speak, then stop. Nothing is sent anywhere except this local server.");
  $("#process").disabled = !S.pendingFile;
}

document.querySelectorAll(".mode").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));

async function loadDemo() {
  setNote("");
  try {
    // The recording is under 1 MB: load it as a blob so seeking works even on static hosts without Range support.
    const [m, p, blob] = await Promise.all([
      fetch("demo/meeting.json").then((r) => r.json()),
      fetch("demo/peaks.json").then((r) => r.json()),
      fetch("demo/meeting.mp3").then((r) => r.blob()),
    ]);
    loadMeeting(m, p.peaks, p.per_second, URL.createObjectURL(blob), { recorded: true });
  } catch (e) {
    $("#wave-msg").hidden = false;
    $("#wave-msg").textContent = "Could not load the bundled demo run.";
  }
}

/* ---------- upload / record ---------- */
$("#file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  if (S.backend && f.size > S.backend.max_upload_mb * 1024 * 1024) {
    setNote(`That file is larger than ${S.backend.max_upload_mb} MB.`, true);
    return;
  }
  S.pendingFile = f;
  $("#file-name").textContent = f.name;
  $("#process").disabled = !S.backend;
  setNote("");
});

let rec = null;
$("#rec").addEventListener("click", async () => {
  if (rec) { rec.stop(); return; }
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    setNote("This browser cannot record audio. Upload a file instead.", true);
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"].find((m) => MediaRecorder.isTypeSupported(m)) || "";
    const mr = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const chunks = [];
    const started = performance.now();
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const an = ac.createAnalyser();
    an.fftSize = 256;
    ac.createMediaStreamSource(stream).connect(an);
    const buf = new Uint8Array(an.frequencyBinCount);
    const mctx = $("#meter").getContext("2d");
    let live = true;
    const tick = () => {
      if (!live) return;
      an.getByteTimeDomainData(buf);
      let peak = 0;
      for (const v of buf) peak = Math.max(peak, Math.abs(v - 128));
      mctx.clearRect(0, 0, 160, 28);
      mctx.fillStyle = `rgb(${LIME})`;
      mctx.fillRect(0, 10, Math.min(160, (peak / 128) * 320), 8);
      const sec = (performance.now() - started) / 1000;
      $("#rec-time").textContent = fmtTime(sec);
      if (sec > 600) mr.stop();
      requestAnimationFrame(tick);
    };
    mr.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    mr.onstop = () => {
      live = false;
      stream.getTracks().forEach((t) => t.stop());
      ac.close();
      rec = null;
      $("#rec").classList.remove("on");
      $("#rec-label").textContent = "Record again";
      const type = mr.mimeType || "audio/webm";
      const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
      S.pendingFile = new File(chunks, `recording.${ext}`, { type });
      $("#process").disabled = !S.backend;
      setNote(`Recorded ${$("#rec-time").textContent}. Press "Transcribe and extract".`);
    };
    mr.start(500);
    rec = mr;
    $("#rec").classList.add("on");
    $("#rec-label").textContent = "Stop";
    setNote("");
    tick();
  } catch (e) {
    setNote("Microphone access was denied or is unavailable.", true);
  }
});

$("#process").addEventListener("click", async () => {
  if (!S.pendingFile || !S.backend) return;
  const file = S.pendingFile;
  const fd = new FormData();
  fd.append("file", file);
  fd.append("date", $("#date").value || new Date().toISOString().slice(0, 10));
  fd.append("diarize", $("#diar").checked ? "true" : "false");
  fd.append("extractor", $("#extractor").value);
  if ($("#participants").value.trim()) fd.append("participants", $("#participants").value.trim());
  $("#process").disabled = true;
  setNote("");
  showPipe({ stage: "decode", progress: 0 });
  try {
    const r = await fetch("api/jobs", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
    const { id } = await r.json();
    await follow(id);
    const job = await fetch(`api/jobs/${id}`).then((x) => x.json());
    if (job.state === "error") throw new Error(job.error || "processing failed");
    $("#pipe").hidden = true;
    const url = URL.createObjectURL(file);
    loadMeeting(job.meeting, job.peaks, job.peaks_per_second, url, { objectUrl: true });
  } catch (e) {
    $("#pipe").hidden = true;
    setNote(String(e.message || e), true);
  } finally {
    $("#process").disabled = !S.pendingFile;
  }
});

function showPipe(st) {
  $("#pipe").hidden = false;
  const order = ["decode", "transcribe", "diarize", "extract"];
  const idx = order.indexOf(st.stage);
  document.querySelectorAll("#pipe .steps span").forEach((s) => {
    const k = order.indexOf(s.dataset.s);
    s.classList.toggle("on", k === idx);
    s.classList.toggle("done", idx >= 0 && k < idx);
  });
  const stageShare = { decode: [0, 0.05], transcribe: [0.05, 0.85], diarize: [0.85, 0.92], extract: [0.92, 1] };
  const [a, b] = stageShare[st.stage] || [0, 0];
  $("#pipe-bar").style.width = `${(a + (b - a) * (st.progress || 0)) * 100}%`;
  $("#pipe-msg").textContent = `${st.stage} ${Math.round((st.progress || 0) * 100)}%`;
}

function follow(id) {
  return new Promise((resolve) => {
    if (!window.EventSource) return poll(id).then(resolve);
    const es = new EventSource(`api/jobs/${id}/events`);
    es.onmessage = (e) => {
      const st = JSON.parse(e.data);
      showPipe(st);
      if (st.state === "done" || st.state === "error") { es.close(); resolve(); }
    };
    es.onerror = () => { es.close(); poll(id).then(resolve); };
  });
}
async function poll(id) {
  for (;;) {
    const st = await fetch(`api/jobs/${id}`).then((r) => r.json());
    showPipe(st);
    if (st.state === "done" || st.state === "error") return;
    await new Promise((r) => setTimeout(r, 500));
  }
}

/* ---------- boot ---------- */
/* The top bar describes what is on screen: the recorded run in Demo, the local server in Upload and Record. */
function renderEnv() {
  const env = $("#env");
  env.textContent = "";
  if (S.mode === "demo") {
    env.append(S.backend ? "recorded run" : "static demo · recorded run", S.recordedModel ? " · " : "");
    if (S.recordedModel) env.append(Object.assign(el("b", null, S.recordedModel)));
  } else if (S.backend) {
    env.append("local server · ", Object.assign(el("b", null, `whisper ${S.backend.whisper_model}`)), ` · extractor ${S.backend.extractor}`);
  } else {
    env.textContent = "upload and record: run it locally";
  }
}

async function boot() {
  $("#date").value = new Date().toISOString().slice(0, 10);
  const staticSite = !!document.querySelector('meta[name="voicescribe-static"]'); // set by scripts/build_docs.py
  try {
    if (staticSite) throw new Error("static demo");
    const r = await fetch("api/health", { headers: { accept: "application/json" } });
    if (r.ok && (r.headers.get("content-type") || "").includes("json")) S.backend = await r.json();
  } catch (e) { /* static hosting: no backend */ }
  if (S.backend) {
    $("#diar").disabled = !S.backend.diarization_available;
    $("#diar-wrap").classList.toggle("off", !S.backend.diarization_available);
    if (!S.backend.diarization_available) $("#diar-wrap").title = "Run: voicescribe fetch-models --diarize";
  } else {
    $("#diar").disabled = true;
    $("#diar-wrap").classList.add("off");
  }
  renderEnv();
  setMode("demo");
}
boot();
