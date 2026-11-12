"""Speech recognition with word-level timestamps."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Protocol

import numpy as np

from .audio import SAMPLE_RATE, default_threads
from .schema import Word

ProgressFn = Callable[[float], None]


class Transcriber(Protocol):
    name: str

    def transcribe(self, samples: np.ndarray, progress: ProgressFn | None = None) -> list[Word]: ...


class FasterWhisperTranscriber:
    """faster-whisper (CTranslate2, int8 on CPU) with attention-based word timestamps."""

    def __init__(self, model: str | None = None, threads: int | None = None, compute_type: str = "int8"):
        self.model_name = model or os.environ.get("VOICESCRIBE_WHISPER", "base.en")
        self.name = f"faster-whisper/{self.model_name}"
        self.threads = threads or default_threads()
        self.compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_name, device="cpu", compute_type=self.compute_type, cpu_threads=self.threads
            )
        return self._model

    def transcribe(self, samples: np.ndarray, progress: ProgressFn | None = None) -> list[Word]:
        model = self._load()
        total = max(len(samples) / SAMPLE_RATE, 1e-6)
        segments, _info = model.transcribe(
            samples,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
            beam_size=1,
        )
        words: list[Word] = []
        for seg in segments:
            for w in seg.words or []:
                words.append(Word(text=w.word, start=float(w.start), end=float(w.end), prob=float(w.probability)))
            if progress:
                progress(min(1.0, float(seg.end) / total))
        return words


class StubTranscriber:
    """Returns canned words; used by tests and by offline replays."""

    name = "stub"

    def __init__(self, words: list[Word]):
        self._words = words

    def transcribe(self, samples: np.ndarray, progress: ProgressFn | None = None) -> list[Word]:
        if progress:
            progress(1.0)
        return list(self._words)
