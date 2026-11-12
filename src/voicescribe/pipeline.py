"""End-to-end pipeline: audio -> words -> sentences -> structured meeting."""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import audio as audio_mod
from .asr import FasterWhisperTranscriber, Transcriber
from .extract.base import Extractor, get_extractor
from .extract.rules import name_speakers
from .extract.summary import extractive_summary
from .names import apply_map, apply_to_owner, canonical_map
from .schema import Meeting, Transcript, Word
from .segment import assign_speakers, clean_words, split_sentences, topic_boundaries

ProgressFn = Callable[[str, float], None]


def build_meeting(
    words: list[Word],
    duration: float,
    meeting_date: dt.date,
    extractor: Extractor,
    *,
    title: str = "Meeting",
    model: str = "unknown",
    turns: list[tuple[float, float, str]] | None = None,
    people: list[str] | None = None,
    synthetic: bool = False,
    audio_name: str | None = None,
) -> Meeting:
    """Pure function from words to a validated Meeting (no audio, no models except the extractor)."""
    t0 = time.perf_counter()
    words = clean_words(words)
    speakers = assign_speakers(words, turns) if turns else None
    sentences = split_sentences(words, speakers)
    name_map = canonical_map(sentences, people)  # one person, one spelling (closed set when participants are given)
    if name_map:
        words = apply_map(words, name_map)
        sentences = split_sentences(words, speakers)
    speaker_names: dict[str, str] = {}
    if turns:  # name the speakers the dialogue pins down; the rest stay "Speaker N"
        speaker_names = name_speakers(sentences, people)
        for s in sentences:
            s.speaker = speaker_names.get(s.speaker or "", s.speaker)
    transcript = Transcript(duration=duration, model=model, sentences=sentences)
    t1 = time.perf_counter()
    actions, decisions, abstractive = extractor.extract(transcript, meeting_date, people)
    for a in actions:
        a.owner = apply_to_owner(a.owner, people)
    t2 = time.perf_counter()
    summary = abstractive or extractive_summary(transcript)
    names = sorted({s.speaker for s in sentences if s.speaker})
    return Meeting(
        title=title,
        date=meeting_date,
        synthetic=synthetic,
        audio_name=audio_name,
        speakers=names,
        participants=list(people or []),
        speaker_names=speaker_names,
        name_map=name_map,
        transcript=transcript,
        summary=summary,
        topics=topic_boundaries(sentences),
        decisions=decisions,
        action_items=sorted(actions, key=lambda a: a.source.start),
        extractor=getattr(extractor, "last_method", extractor.method),
        timings={"segment_s": round(t1 - t0, 3), "extract_s": round(t2 - t1, 3)},
    )


def process_file(
    path: str | Path,
    meeting_date: dt.date | None = None,
    *,
    transcriber: Transcriber | None = None,
    extractor: Extractor | None = None,
    diarise: bool = False,
    models_dir: Path = Path("models"),
    title: str | None = None,
    synthetic: bool = False,
    people: list[str] | None = None,
    progress: ProgressFn | None = None,
    on_words: Callable[[list[Word]], None] | None = None,
) -> tuple[Meeting, np.ndarray]:
    """Decode, transcribe, (optionally diarise), segment and extract. Returns the meeting and the samples."""

    def tick(stage: str, frac: float) -> None:
        if progress:
            progress(stage, frac)

    path = Path(path)
    tick("decode", 0.0)
    t0 = time.perf_counter()
    samples = audio_mod.decode(path)
    duration = audio_mod.duration(samples)
    decode_s = time.perf_counter() - t0
    tick("decode", 1.0)

    transcriber = transcriber or FasterWhisperTranscriber()
    t0 = time.perf_counter()
    words = transcriber.transcribe(samples, progress=lambda f: tick("transcribe", f))
    asr_s = time.perf_counter() - t0
    if on_words:
        on_words(list(words))  # the raw recogniser output, before name correction

    turns = None
    diar_s = 0.0
    if diarise:
        from .diarize import SENTENCE_DEFAULTS, diarize_sentences

        tick("diarize", 0.0)
        t0 = time.perf_counter()
        spans = [(s.start, s.end) for s in split_sentences(clean_words(words))]
        # a participants list fixes the number of speakers; otherwise it is estimated from the embeddings
        turns = diarize_sentences(samples, spans, models_dir, num_speakers=len(people) if people else None, **SENTENCE_DEFAULTS)
        diar_s = time.perf_counter() - t0
        tick("diarize", 1.0)

    tick("extract", 0.0)
    meeting = build_meeting(
        words,
        duration,
        meeting_date or dt.date.today(),
        extractor or get_extractor(),
        title=title or path.stem.replace("_", " ").title(),
        model=getattr(transcriber, "name", "unknown"),
        turns=turns,
        people=people,
        synthetic=synthetic,
        audio_name=path.name,
    )
    meeting.timings.update(
        {"decode_s": round(decode_s, 3), "asr_s": round(asr_s, 3), "diarize_s": round(diar_s, 3),
         "audio_s": round(duration, 3), "rtf": round(asr_s / duration, 3) if duration else 0.0}
    )  # fmt: skip
    tick("extract", 1.0)
    return meeting, samples
