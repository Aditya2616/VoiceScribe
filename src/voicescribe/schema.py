"""Pydantic models for the transcript and the structured meeting output."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Method = Literal["rules", "local", "openai", "anthropic"]


class Word(BaseModel):
    text: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    prob: float | None = None

    @model_validator(mode="after")
    def _ordered(self) -> Word:
        if self.end < self.start:
            raise ValueError("word end before start")
        return self


class Sentence(BaseModel):
    id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    speaker: str | None = None
    words: list[Word] = Field(default_factory=list)


class Transcript(BaseModel):
    duration: float = Field(ge=0)
    model: str = "unknown"
    sentences: list[Sentence] = Field(default_factory=list)

    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)

    def by_id(self) -> dict[int, Sentence]:
        return {s.id: s for s in self.sentences}


class SourceSpan(BaseModel):
    """Where in the audio an item came from."""

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    sentence_ids: list[int] = Field(min_length=1)
    quote: str


class ActionItem(BaseModel):
    task: str = Field(min_length=2)
    owner: str | None = None
    deadline_text: str | None = None
    deadline_date: dt.date | None = None
    source: SourceSpan
    confidence: float = Field(default=0.5, ge=0, le=1)
    method: Method = "rules"


class Decision(BaseModel):
    text: str = Field(min_length=2)
    source: SourceSpan
    method: Method = "rules"


class Topic(BaseModel):
    start_sentence: int
    start: float
    label: str


class Summary(BaseModel):
    text: str
    kind: Literal["extractive", "abstractive"] = "extractive"
    sentence_ids: list[int] = Field(default_factory=list)


class Meeting(BaseModel):
    title: str = "Meeting"
    date: dt.date
    synthetic: bool = False
    audio_name: str | None = None
    speakers: list[str] = Field(default_factory=list)
    transcript: Transcript
    summary: Summary | None = None
    topics: list[Topic] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    extractor: str = "rules"
    timings: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _spans_exist(self) -> Meeting:
        """Every cited sentence must exist and the quote must come from the transcript."""
        ids = self.transcript.by_id()
        for item in [*self.action_items, *self.decisions]:
            for i in item.source.sentence_ids:
                if i not in ids:
                    raise ValueError(f"source sentence {i} not in transcript")
            joined = " ".join(ids[i].text for i in item.source.sentence_ids)
            if norm_ws(item.source.quote) not in norm_ws(joined):
                raise ValueError("source quote not found in the cited sentences")
        return self


def norm_ws(s: str) -> str:
    return " ".join(s.split())


def make_span(transcript: Transcript, sentence_ids: list[int]) -> SourceSpan:
    """Build a span covering the given sentences (quote is the exact transcript text)."""
    by = transcript.by_id()
    ids = sorted(set(sentence_ids))
    sents = [by[i] for i in ids]
    return SourceSpan(
        start=sents[0].start,
        end=sents[-1].end,
        sentence_ids=ids,
        quote=" ".join(s.text for s in sents),
    )
