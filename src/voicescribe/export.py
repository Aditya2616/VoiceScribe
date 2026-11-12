"""Exports: Markdown minutes, JSON, SRT and VTT subtitles."""

from __future__ import annotations

from .schema import Meeting, Sentence


def ts(seconds: float, sep: str = ",") -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def mmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def cues(sentences: list[Sentence], max_chars: int = 84, max_seconds: float = 6.0) -> list[tuple[float, float, str]]:
    """Subtitle cues built from the real word timings, split on length and duration."""
    out: list[tuple[float, float, str]] = []
    for s in sentences:
        if not s.words:
            out.append((s.start, max(s.end, s.start + 0.5), s.text))
            continue
        cur: list = []
        for w in s.words:
            trial = " ".join(x.text for x in [*cur, w])
            if cur and (len(trial) > max_chars or w.end - cur[0].start > max_seconds):
                out.append((cur[0].start, cur[-1].end, _text(cur)))
                cur = []
            cur.append(w)
        if cur:
            out.append((cur[0].start, cur[-1].end, _text(cur)))
    return out


def _text(ws) -> str:
    return " ".join(w.text for w in ws)


def to_srt(meeting: Meeting) -> str:
    blocks = []
    for i, (a, b, text) in enumerate(cues(meeting.transcript.sentences), 1):
        blocks.append(f"{i}\n{ts(a)} --> {ts(b)}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(meeting: Meeting) -> str:
    blocks = ["WEBVTT\n"]
    for a, b, text in cues(meeting.transcript.sentences):
        blocks.append(f"{ts(a, '.')} --> {ts(b, '.')}\n{text}\n")
    return "\n".join(blocks)


def to_json(meeting: Meeting) -> str:
    return meeting.model_dump_json(indent=2)


def to_markdown(meeting: Meeting) -> str:
    lines = [f"# {meeting.title}", "", f"Date: {meeting.date.isoformat()}  "]
    if meeting.speakers:
        lines.append(f"Speakers: {', '.join(meeting.speakers)}  ")
    if meeting.synthetic:
        lines.append("Audio: synthetic speech (Kokoro-82M)  ")
    lines.append(f"Extractor: {meeting.extractor}")
    lines.append("")
    if meeting.summary:
        lines += [f"## Summary ({meeting.summary.kind})", "", meeting.summary.text, ""]
    lines += ["## Decisions", ""]
    if meeting.decisions:
        lines += [f"- {d.text} [{mmss(d.source.start)}]" for d in meeting.decisions]
    else:
        lines.append("- none found")
    lines += ["", "## Action items", ""]
    if meeting.action_items:
        lines += ["| Task | Owner | Due | At |", "|---|---|---|---|"]
        for a in meeting.action_items:
            due = a.deadline_date.isoformat() if a.deadline_date else ""
            if a.deadline_text and a.deadline_date:
                due += f" ({a.deadline_text})"
            lines.append(
                f"| {_cell(a.task)} | {_cell(a.owner or 'unassigned')} | {_cell(due)} | {mmss(a.source.start)} |"
            )
    else:
        lines.append("- none found")
    if meeting.topics and len(meeting.topics) > 1:
        lines += ["", "## Topics", ""]
        lines += [f"- {mmss(t.start)}: {t.label}" for t in meeting.topics]
    lines += ["", "## Transcript", ""]
    for s in meeting.transcript.sentences:
        who = f"**{s.speaker}** " if s.speaker else ""
        lines.append(f"[{mmss(s.start)}] {who}{s.text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def export(meeting: Meeting, fmt: str) -> tuple[str, str]:
    """Return (content, media type) for md, json, srt or vtt."""
    table = {
        "md": (to_markdown, "text/markdown"),
        "json": (to_json, "application/json"),
        "srt": (to_srt, "application/x-subrip"),
        "vtt": (to_vtt, "text/vtt"),
    }
    if fmt not in table:
        raise ValueError(f"unknown format {fmt!r}; choose from {sorted(table)}")
    fn, media = table[fmt]
    return fn(meeting), media
