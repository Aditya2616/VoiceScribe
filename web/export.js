// Exports built in the browser from the meeting JSON. Mirrors src/voicescribe/export.py
// (a test compares the two on the bundled demo run).

export function ts(seconds, sep = ",") {
  let ms = Math.round(seconds * 1000);
  const h = Math.floor(ms / 3600000); ms -= h * 3600000;
  const m = Math.floor(ms / 60000); ms -= m * 60000;
  const s = Math.floor(ms / 1000); ms -= s * 1000;
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${p(h)}:${p(m)}:${p(s)}${sep}${p(ms, 3)}`;
}

export function mmss(seconds) {
  const s = Math.floor(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function cues(sentences, maxChars = 84, maxSeconds = 6.0) {
  const out = [];
  const text = (ws) => ws.map((w) => w.text).join(" ");
  for (const s of sentences) {
    if (!s.words.length) { out.push([s.start, Math.max(s.end, s.start + 0.5), s.text]); continue; }
    let cur = [];
    for (const w of s.words) {
      const trial = text([...cur, w]);
      if (cur.length && (trial.length > maxChars || w.end - cur[0].start > maxSeconds)) {
        out.push([cur[0].start, cur[cur.length - 1].end, text(cur)]);
        cur = [];
      }
      cur.push(w);
    }
    if (cur.length) out.push([cur[0].start, cur[cur.length - 1].end, text(cur)]);
  }
  return out;
}

export const toSrt = (m) =>
  cues(m.transcript.sentences).map(([a, b, t], i) => `${i + 1}\n${ts(a)} --> ${ts(b)}\n${t}\n`).join("\n");

export const toVtt = (m) =>
  ["WEBVTT\n", ...cues(m.transcript.sentences).map(([a, b, t]) => `${ts(a, ".")} --> ${ts(b, ".")}\n${t}\n`)].join("\n");

export const toJson = (m) => JSON.stringify(m, null, 2);

const cell = (t) => String(t).replaceAll("|", "\\|").replaceAll("\n", " ");

export function toMarkdown(m) {
  const L = [`# ${m.title}`, "", `Date: ${m.date}  `];
  if (m.speakers.length) L.push(`Speakers: ${m.speakers.join(", ")}  `);
  if (m.synthetic) L.push("Audio: synthetic speech (Kokoro-82M)  ");
  L.push(`Extractor: ${m.extractor}`, "");
  if (m.summary) L.push(`## Summary (${m.summary.kind})`, "", m.summary.text, "");
  L.push("## Decisions", "");
  if (m.decisions.length) m.decisions.forEach((d) => L.push(`- ${d.text} [${mmss(d.source.start)}]`));
  else L.push("- none found");
  L.push("", "## Action items", "");
  if (m.action_items.length) {
    L.push("| Task | Owner | Due | At |", "|---|---|---|---|");
    for (const a of m.action_items) {
      let due = a.deadline_date || "";
      if (a.deadline_text && a.deadline_date) due += ` (${a.deadline_text})`;
      L.push(`| ${cell(a.task)} | ${cell(a.owner || "unassigned")} | ${cell(due)} | ${mmss(a.source.start)} |`);
    }
  } else L.push("- none found");
  if (m.topics.length > 1) {
    L.push("", "## Topics", "");
    m.topics.forEach((t) => L.push(`- ${mmss(t.start)}: ${t.label}`));
  }
  L.push("", "## Transcript", "");
  for (const s of m.transcript.sentences) {
    L.push(`[${mmss(s.start)}] ${s.speaker ? `**${s.speaker}** ` : ""}${s.text}`, "");
  }
  return L.join("\n").trimEnd() + "\n";
}

export const FORMATS = {
  md: { fn: toMarkdown, type: "text/markdown" },
  json: { fn: toJson, type: "application/json" },
  srt: { fn: toSrt, type: "application/x-subrip" },
  vtt: { fn: toVtt, type: "text/vtt" },
};

export function download(meeting, fmt) {
  const { fn, type } = FORMATS[fmt];
  const url = URL.createObjectURL(new Blob([fn(meeting)], { type: `${type};charset=utf-8` }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `meeting.${fmt}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
