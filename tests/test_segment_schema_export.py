import datetime as dt
import re

import pytest
from conftest import MEETING_DATE, transcript_from_lines, words_from_text
from pydantic import ValidationError

from voicescribe.export import cues, to_markdown, to_srt, to_vtt, ts
from voicescribe.extract.base import RulesExtractor
from voicescribe.pipeline import build_meeting
from voicescribe.schema import ActionItem, Meeting, SourceSpan, Word, make_span
from voicescribe.segment import (
    assign_speakers,
    clean_words,
    split_sentences,
    topic_boundaries,
)


def test_clean_words_makes_times_monotonic_and_bounded():
    raw = [
        Word(text=" Hello", start=0.0, end=3.0),  # stretched over a pause
        Word(text="world.", start=1.0, end=1.4),  # overlaps the previous word's end
        Word(text="  ", start=1.5, end=1.6),  # blank token is dropped
        Word(text="Next", start=0.5, end=0.9),  # start goes backwards
    ]
    out = clean_words(raw)
    assert [w.text for w in out] == ["Hello", "world.", "Next"]
    starts = [w.start for w in out]
    assert starts == sorted(starts)
    for a, b in zip(out, out[1:], strict=False):
        assert a.end <= b.start + 1e-9
    assert out[0].end - out[0].start <= 1.0


def test_sentences_split_on_punctuation_and_keep_abbreviations():
    ws = words_from_text("Dr. Rao will join us. Is that fine? Yes it is!")
    s = split_sentences(ws)
    assert [x.text for x in s] == ["Dr. Rao will join us.", "Is that fine?", "Yes it is!"]
    assert [x.id for x in s] == [0, 1, 2]
    assert s[0].start == ws[0].start and s[0].end == ws[4].end


def test_sentences_split_on_pause_and_length_without_punctuation():
    a = words_from_text("we should ship it", 0.0)
    b = words_from_text("then we can rest", a[-1].end + 2.0)  # 2 s of silence
    assert len(split_sentences(a + b)) == 2
    long = words_from_text(" ".join(["word"] * 100))
    assert all(len(s.words) <= 45 for s in split_sentences(long))


def test_sentences_split_on_speaker_change_and_majority_speaker():
    ws = words_from_text("yes agreed let me check")
    spk = ["A", "A", "B", "B", "B"]
    s = split_sentences(ws, spk)
    assert [x.speaker for x in s] == ["A", "B"]


def test_assign_speakers_by_overlap_and_nearest_turn():
    ws = words_from_text("one two three", 0.0, per_word=0.5, gap=0.0)
    turns = [(0.0, 0.9, "S1"), (1.0, 3.0, "S2")]
    assert assign_speakers(ws, turns) == ["S1", "S1", "S2"]
    gap_word = [Word(text="x", start=5.0, end=5.2)]
    assert assign_speakers(gap_word, turns) == ["S2"]


def test_topic_boundaries_find_a_vocabulary_shift():
    a = ["database migration replica cutover rollback plan"] * 6
    b = ["marketing campaign budget forecast audience launch"] * 6
    t = transcript_from_lines([(None, x + f" item{i}.") for i, x in enumerate(a + b)])
    topics = topic_boundaries(t.sentences)
    assert len(topics) == 2 and topics[1].start_sentence in (5, 6, 7)


def test_srt_and_vtt_timestamp_formats():
    assert ts(3723.456) == "01:02:03,456"
    assert ts(3723.456, ".") == "01:02:03.456"
    t = transcript_from_lines([("Ana", "Hello there."), ("Ben", "Hi.")])
    m = Meeting(date=MEETING_DATE, transcript=t)
    srt = to_srt(m)
    assert srt.startswith("1\n00:00:00,000 --> ")
    assert re.search(r"\n\n2\n", srt)
    vtt = to_vtt(m)
    assert vtt.startswith("WEBVTT\n") and "-->" in vtt and "," not in vtt.split("-->")[0]


def test_cues_split_long_sentences():
    t = transcript_from_lines([(None, " ".join(["lorem"] * 40) + ".")])
    parts = cues(t.sentences, max_chars=60)
    assert len(parts) > 2 and all(len(text) <= 60 for _, _, text in parts)


def test_markdown_minutes_contain_items_and_escape_pipes():
    t = transcript_from_lines([("Ana", "Ben, can you fix the a|b parser by Friday?"), ("Ben", "Sure, I will.")])
    m = build_meeting(
        [w for s in t.sentences for w in s.words], t.duration, MEETING_DATE, RulesExtractor(), title="Sync", synthetic=True
    )
    md = to_markdown(m)
    assert md.startswith("# Sync") and "synthetic speech" in md
    assert "a\\|b" in md and "2026-11-13" in md and "Ben" in md


def test_meeting_rejects_spans_that_do_not_exist():
    t = transcript_from_lines([(None, "We will ship on Friday.")])
    good = SourceSpan(start=0, end=1, sentence_ids=[0], quote="We will ship on Friday.")
    ok = ActionItem(task="Ship", source=good)
    Meeting(date=MEETING_DATE, transcript=t, action_items=[ok])
    with pytest.raises(ValidationError):
        Meeting(date=MEETING_DATE, transcript=t, action_items=[ActionItem(task="Ship", source=good.model_copy(update={"sentence_ids": [7]}))])
    with pytest.raises(ValidationError):
        Meeting(date=MEETING_DATE, transcript=t, action_items=[ActionItem(task="Ship", source=good.model_copy(update={"quote": "made up quote"}))])


def test_make_span_and_word_validation():
    t = transcript_from_lines([(None, "One."), (None, "Two."), (None, "Three.")])
    span = make_span(t, [2, 0])
    assert span.sentence_ids == [0, 2] and span.quote == "One. Three."
    with pytest.raises(ValidationError):
        Word(text="x", start=2.0, end=1.0)


def test_build_meeting_roundtrips_through_json():
    t = transcript_from_lines([("Ana", "Okay let us begin."), ("Ana", "Ben, please send the notes by tomorrow.")])
    m = build_meeting([w for s in t.sentences for w in s.words], t.duration, dt.date(2026, 11, 12), RulesExtractor())
    again = Meeting.model_validate_json(m.model_dump_json())
    assert again.action_items[0].deadline_date == dt.date(2026, 11, 13)
    assert again.summary is not None and again.summary.kind == "extractive"
