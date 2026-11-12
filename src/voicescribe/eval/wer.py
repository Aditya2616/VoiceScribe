"""Word error rate (own implementation, cross-checked against jiwer in the tests)."""

from __future__ import annotations

import re

_CONTRACTIONS = {"it's": "it is", "that's": "that is", "i'm": "i am", "don't": "do not", "can't": "cannot"}


def normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation (keep apostrophes inside words), split on whitespace."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", text)
    return text.split()


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) from a Levenshtein alignment."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    i, j, s, de, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            s += 0 if ref[i - 1] == hyp[j - 1] else 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            de += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return s, de, ins


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    s, d, i = edit_counts(ref, hyp)
    return (s + d + i) / len(ref)


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    """Micro-averaged WER: total errors over total reference words."""
    errors = words = 0
    for ref, hyp in pairs:
        r, h = normalize(ref), normalize(hyp)
        s, d, i = edit_counts(r, h)
        errors += s + d + i
        words += len(r)
    return errors / words if words else 0.0
