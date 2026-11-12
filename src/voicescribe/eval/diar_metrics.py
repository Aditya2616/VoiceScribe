"""Diarisation metrics on 10 ms frames: diarisation error rate, frame accuracy and speaker-count accuracy.

Reference and hypothesis are lists of (start, end, speaker). DER = (missed speech + false alarm + speaker confusion)
over reference speech time, after the best one-to-one mapping of hypothesis speakers to reference speakers, ignoring
frames within `collar` seconds of a reference turn boundary (the usual 0.25 s forgiveness).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _labels(turns: list[tuple[float, float, str]], frames: int, step: float) -> tuple[np.ndarray, list[str]]:
    names = sorted({t[2] for t in turns})
    lab = np.full(frames, -1, dtype=np.int32)
    for s0, e0, spk in turns:
        lab[int(round(s0 / step)) : min(frames, int(round(e0 / step)))] = names.index(spk)
    return lab, names


def der(ref: list[tuple[float, float, str]], hyp: list[tuple[float, float, str]], collar: float = 0.25, step: float = 0.01) -> dict:
    end = max([t[1] for t in ref] + [t[1] for t in hyp]) + 1.0
    frames = int(end / step) + 1
    r, rn = _labels(ref, frames, step)
    h, hn = _labels(hyp, frames, step)
    keep = np.ones(frames, dtype=bool)
    c = int(round(collar / step))
    for s0, e0, _ in ref:
        for b in (int(round(s0 / step)), int(round(e0 / step))):
            keep[max(0, b - c) : b + c] = False
    r, h = r[keep], h[keep]
    speech = r >= 0
    total = int(speech.sum())
    if total == 0:
        return {"der": 0.0, "frame_accuracy": 0.0, "miss": 0.0, "false_alarm": 0.0, "confusion": 0.0}
    both = speech & (h >= 0)
    cost = np.zeros((len(rn), max(1, len(hn))))
    for i in range(len(rn)):
        for j in range(len(hn)):
            cost[i, j] = -int(((r == i) & (h == j)).sum())
    rows, cols = linear_sum_assignment(cost) if len(hn) else ([], [])
    mapping = {int(j): int(i) for i, j in zip(rows, cols, strict=False)}
    correct = sum(int(((r == i) & (h == j)).sum()) for j, i in mapping.items())
    miss = int((speech & (h < 0)).sum())
    fa = int(((~speech) & (h >= 0)).sum())
    confusion = int(both.sum()) - correct
    return {
        "der": (miss + fa + confusion) / total,
        "frame_accuracy": correct / total,
        "miss": miss / total,
        "false_alarm": fa / total,
        "confusion": confusion / total,
        "true_speakers": len(rn),
        "predicted_speakers": len(hn),
    }


def summarise(per_meeting: list[dict]) -> dict:
    """Micro-average over meetings (weighted by reference speech) plus speaker-count accuracy."""
    n = len(per_meeting)
    w = [m["speech"] for m in per_meeting]
    tot = sum(w)
    out = {k: sum(m[k] * wi for m, wi in zip(per_meeting, w, strict=True)) / tot for k in ("der", "frame_accuracy", "miss", "false_alarm", "confusion")}
    out["meetings"] = n
    out["speaker_count_exact"] = sum(m["predicted_speakers"] == m["true_speakers"] for m in per_meeting) / n
    out["speaker_count_mean_abs_error"] = sum(abs(m["predicted_speakers"] - m["true_speakers"]) for m in per_meeting) / n
    return out
