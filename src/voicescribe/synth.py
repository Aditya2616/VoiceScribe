"""Synthesise the demo meeting from a script with Kokoro-82M (Apache-2.0), recording true timings.

The audio is synthetic speech, labelled as such everywhere it is shown. Because the script is ours, the
ground truth for words, speakers, sentence timings and action items is exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KOKORO_SR = 24000


def trim(a: np.ndarray, thresh: float = 0.01, pad: int = 600) -> tuple[np.ndarray, int]:
    """Trim leading/trailing silence; return (audio, samples removed from the front)."""
    idx = np.flatnonzero(np.abs(a) > thresh)
    if idx.size == 0:
        return a, 0
    lo, hi = max(0, idx[0] - pad), min(len(a), idx[-1] + pad)
    return a[lo:hi], lo


VOICE_POOL = ["af_heart", "am_adam", "bf_emma", "bm_george", "af_nova", "am_michael", "bf_isabella", "bm_lewis",
              "af_sarah", "am_liam"]  # fmt: skip


def assign_voices(speakers: list[str], offset: int = 0) -> dict[str, str]:
    """Distinct Kokoro voices for the speakers of one meeting (steps of 3 through a 10-voice pool never repeat)."""
    return {s: VOICE_POOL[(offset + 3 * k) % len(VOICE_POOL)] for k, s in enumerate(speakers)}


def synthesize(
    script: Path, model: Path, voices: Path, out_dir: Path, name: str = "meeting",
    voice_map: dict[str, str] | None = None, threads: int | None = None,
) -> dict:  # fmt: skip
    from kokoro_onnx import Kokoro

    from .eval.extract_eval import load_meeting

    meeting = load_meeting(script, with_speakers=True)
    header = {}
    for line in script.read_text(encoding="utf8").splitlines():
        if line.startswith("# voices:"):
            header = dict(p.strip().split("=") for p in line.split(":", 1)[1].split(","))
    if voice_map:
        header = voice_map
    if threads:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        kokoro = Kokoro.from_session(ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"]), str(voices))
    else:
        kokoro = Kokoro(str(model), str(voices))
    gap_same, gap_turn = int(0.22 * KOKORO_SR), int(0.55 * KOKORO_SR)
    chunks: list[np.ndarray] = []
    timing = []
    t = 0
    prev_speaker = None
    gold = {g.sentence_id: g for g in meeting.gold}
    for s in meeting.transcript.sentences:
        audio, _sr = kokoro.create(s.text, voice=header[s.speaker], speed=1.0, lang="en-us")
        audio, _ = trim(np.asarray(audio, dtype=np.float32))
        gap = 0 if prev_speaker is None else (gap_same if s.speaker == prev_speaker else gap_turn)
        chunks.append(np.zeros(gap, dtype=np.float32))
        t += gap
        timing.append(
            {
                "id": s.id,
                "speaker": s.speaker,
                "text": s.text,
                "start": round(t / KOKORO_SR, 3),
                "end": round((t + len(audio)) / KOKORO_SR, 3),
                "label": gold[s.id].kind if s.id in gold else None,
            }
        )
        chunks.append(audio)
        t += len(audio)
        prev_speaker = s.speaker
    full = np.concatenate(chunks + [np.zeros(int(0.6 * KOKORO_SR), dtype=np.float32)])
    out_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    wav = out_dir / f"{name}.wav"
    sf.write(wav, full, KOKORO_SR)
    info = {"sample_rate": KOKORO_SR, "duration": round(len(full) / KOKORO_SR, 3), "voices": header, "sentences": timing}
    (out_dir / f"{name}.timing.json").write_text(json.dumps(info, indent=1), encoding="utf8")
    return info
