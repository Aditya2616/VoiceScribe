"""Audio decoding (PyAV, no ffmpeg on PATH needed) and waveform peaks for the UI."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
MAX_DURATION_S = 30 * 60


class AudioError(ValueError):
    pass


def decode(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any container PyAV understands to mono float32 at sample_rate."""
    import av

    try:
        with av.open(str(path), mode="r") as container:
            if not container.streams.audio:
                raise AudioError("file has no audio stream")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
            chunks: list[np.ndarray] = []
            total = 0
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    a = out.to_ndarray().reshape(-1)
                    chunks.append(a)
                    total += a.size
                    if total > MAX_DURATION_S * sample_rate:
                        raise AudioError(f"audio longer than {MAX_DURATION_S // 60} minutes")
            for out in resampler.resample(None) or []:
                chunks.append(out.to_ndarray().reshape(-1))
    except AudioError:
        raise
    except Exception as exc:  # PyAV raises many exception types for bad input
        raise AudioError(f"could not decode audio: {exc}") from exc
    if not chunks:
        raise AudioError("no audio samples decoded")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def peaks(samples: np.ndarray, per_second: int = 50, sample_rate: int = SAMPLE_RATE) -> list[int]:
    """Peak amplitude envelope, 0..255, per_second values per second."""
    hop = sample_rate // per_second
    n = len(samples) // hop
    if n == 0:
        return []
    env = np.abs(samples[: n * hop]).reshape(n, hop).max(axis=1)
    top = float(np.percentile(env, 99.5)) or 1.0
    return np.clip(env / top * 255, 0, 255).astype(np.uint8).tolist()


def duration(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return len(samples) / sample_rate


def default_threads() -> int:
    """About a quarter of the logical cores (the CPU is shared), override with VOICESCRIBE_THREADS."""
    env = os.environ.get("VOICESCRIBE_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, (os.cpu_count() or 4) // 4)


def encode_mp3(
    samples: np.ndarray, path: str | Path, in_rate: int = SAMPLE_RATE, sample_rate: int = 22050, bitrate: int = 48000
) -> None:
    """Write a compact mono mp3 (plays in Chromium and Safari) using PyAV's bundled encoder."""
    import av

    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with av.open(str(path), mode="w", format="mp3") as out:
        stream = out.add_stream("mp3", rate=sample_rate)
        stream.bit_rate = bitrate
        stream.layout = "mono"
        frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = in_rate
        resampler = av.AudioResampler(format="s16p", layout="mono", rate=sample_rate)
        for f in resampler.resample(frame):
            for packet in stream.encode(f):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
