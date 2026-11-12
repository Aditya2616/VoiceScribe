"""Model-backed tests (run with: pytest -m slow). They need models downloaded by `voicescribe fetch-models`."""

import json
from pathlib import Path

import pytest

from voicescribe import audio
from voicescribe.asr import FasterWhisperTranscriber
from voicescribe.eval.wer import wer

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


def test_real_whisper_gives_ordered_word_timestamps_and_low_wer():
    clip = ROOT / "data" / "eval" / "wer" / "260-123440-0018.flac"
    samples = audio.decode(clip)
    words = FasterWhisperTranscriber("tiny.en").transcribe(samples)
    assert words and all(a.start <= b.start for a, b in zip(words, words[1:], strict=False))
    assert words[-1].end <= audio.duration(samples) + 0.5
    ref = "I AM VERY TIRED OF SWIMMING ABOUT HERE O MOUSE"
    assert wer(ref, " ".join(w.text for w in words)) < 0.3


def test_pipeline_on_the_demo_audio_finds_the_dated_action_item():
    import datetime as dt

    from voicescribe.pipeline import process_file

    meeting, _ = process_file(
        ROOT / "data" / "demo" / "meeting.mp3", dt.date(2026, 11, 12), transcriber=FasterWhisperTranscriber("tiny.en")
    )
    due = {a.deadline_date.isoformat() for a in meeting.action_items if a.deadline_date}
    assert "2026-11-20" in due


def test_sentence_diarisation_separates_the_demo_speakers():
    from voicescribe import diarize
    from voicescribe.diarize import SENTENCE_DEFAULTS, diarize_sentences
    from voicescribe.eval.diar_metrics import der

    models = ROOT / "models"
    if not diarize.available(models):
        pytest.skip("speaker embedding model not downloaded")
    truth = json.loads((ROOT / "data" / "demo" / "meeting.timing.json").read_text())["sentences"]
    ref = [(t["start"], t["end"], t["speaker"]) for t in truth]
    turns = diarize_sentences(audio.decode(ROOT / "data" / "demo" / "meeting.mp3"), [(a, b) for a, b, _ in ref], models, **SENTENCE_DEFAULTS)
    assert 3 <= len({t[2] for t in turns}) <= 5 and der(ref, turns)["der"] < 0.1  # 4 true speakers
