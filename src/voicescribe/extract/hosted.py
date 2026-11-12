"""Shared pieces of the hosted-LLM adapters (OpenAI, Anthropic): prompt, JSON schema, and mapping the
model's answer back onto the transcript with validation. These adapters are tested against stub HTTP
transports only; they have not been run against the live APIs."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from ..dates import find_deadline
from ..schema import ActionItem, Decision, Method, Summary, Transcript, make_span

SYSTEM = (
    "You turn a meeting transcript into structured minutes. Sentences are numbered like [12]. "
    "Return a short abstractive summary (2 to 4 sentences), the decisions the group made, and the action items. "
    "An action item is a concrete task that someone commits to or is asked to do. Give the owner's name if it is "
    "clear from the transcript, otherwise null. Copy the deadline wording exactly as spoken (deadline_text) and "
    "do not compute dates. Every decision and action item must cite the sentence numbers it came from. "
    "Do not invent items that are not in the transcript."
)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "decisions", "action_items"],
    "properties": {
        "summary": {"type": "string"},
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "sentence_ids"],
                "properties": {"text": {"type": "string"}, "sentence_ids": {"type": "array", "items": {"type": "integer"}}},
            },
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["task", "owner", "deadline_text", "sentence_ids"],
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": ["string", "null"]},
                    "deadline_text": {"type": ["string", "null"]},
                    "sentence_ids": {"type": "array", "items": {"type": "integer"}},
                },
            },
        },
    },
}


def render_transcript(transcript: Transcript, meeting_date: dt.date, people: list[str] | None) -> str:
    lines = [f"Meeting date: {meeting_date.isoformat()} ({meeting_date.strftime('%A')})"]
    if people:
        lines.append(f"Known participants: {', '.join(people)}")
    lines.append("Transcript:")
    for s in transcript.sentences:
        who = f"{s.speaker}: " if s.speaker else ""
        lines.append(f"[{s.id}] {who}{s.text}")
    return "\n".join(lines)


def to_results(
    payload: dict[str, Any], transcript: Transcript, meeting_date: dt.date, method: Method
) -> tuple[list[ActionItem], list[Decision], Summary | None]:
    """Validate a model payload. Items citing sentences that do not exist are dropped, never repaired."""
    valid = set(transcript.by_id())

    def ids_of(raw: Any) -> list[int]:
        return sorted({i for i in raw if isinstance(i, int) and i in valid}) if isinstance(raw, list) else []

    actions: list[ActionItem] = []
    for it in payload.get("action_items") or []:
        ids = ids_of(it.get("sentence_ids"))
        task = (it.get("task") or "").strip()
        if not ids or len(task) < 2:
            continue
        span = make_span(transcript, ids)
        phrase = (it.get("deadline_text") or "").strip() or None
        found = find_deadline(phrase, meeting_date) if phrase else None
        if found is None and phrase is None:
            found = find_deadline(span.quote, meeting_date)
        owner = (it.get("owner") or "").strip() or None
        actions.append(
            ActionItem(
                task=task, owner=owner, deadline_text=(found[0] if found else phrase),
                deadline_date=found[1] if found else None, source=span, confidence=0.8, method=method,
            )
        )  # fmt: skip
    decisions: list[Decision] = []
    for it in payload.get("decisions") or []:
        ids = ids_of(it.get("sentence_ids"))
        text = (it.get("text") or "").strip()
        if ids and len(text) >= 2:
            decisions.append(Decision(text=text, source=make_span(transcript, ids), method=method))
    summary_text = re.sub(r"\s+", " ", payload.get("summary") or "").strip()
    summary = Summary(text=summary_text, kind="abstractive") if summary_text else None
    return actions, decisions, summary
