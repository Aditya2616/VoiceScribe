import datetime as dt
import wave
from pathlib import Path

import numpy as np
import pytest

from voicescribe.schema import Transcript, Word
from voicescribe.segment import split_sentences

ROOT = Path(__file__).resolve().parents[1]
MEETING_DATE = dt.date(2026, 11, 12)  # a Thursday


def words_from_text(text: str, start: float = 0.0, per_word: float = 0.3, gap: float = 0.05) -> list[Word]:
    out, t = [], start
    for tok in text.split():
        out.append(Word(text=tok, start=round(t, 3), end=round(t + per_word, 3)))
        t += per_word + gap
    return out


def transcript_from_lines(lines: list[tuple[str | None, str]]) -> Transcript:
    """Build a Transcript with one sentence per line; (speaker, text)."""
    from voicescribe.schema import Sentence

    sents, t = [], 0.0
    for i, (spk, text) in enumerate(lines):
        ws = words_from_text(text, t)
        sents.append(Sentence(id=i, start=ws[0].start, end=ws[-1].end, text=text, speaker=spk, words=ws))
        t = ws[-1].end + 0.4
    return Transcript(duration=t, model="test", sentences=sents)


@pytest.fixture
def meeting_date() -> dt.date:
    return MEETING_DATE


@pytest.fixture
def tiny_wav(tmp_path: Path) -> Path:
    path = tmp_path / "tone.wav"
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    pcm = (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    return path


__all__ = ["MEETING_DATE", "ROOT", "split_sentences", "transcript_from_lines", "words_from_text"]
