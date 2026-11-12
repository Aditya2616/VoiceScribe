"""Word error rate (own implementation, cross-checked against jiwer in the tests) and word alignment."""

from __future__ import annotations

import re

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
    "ninth": 9, "tenth": 10, "twentieth": 20, "thirtieth": 30,
}  # fmt: skip


def _suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def numbers_to_digits(tokens: list[str]) -> list[str]:
    """Spoken numbers to digits ("two hundred" -> "200", "twenty fifth" -> "25th") so that Whisper's digit
    output and a spelled-out reference compare equal. Applied to both sides."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t in TENS:
            if nxt in UNITS and 1 <= UNITS[nxt] <= 9:
                out.append(str(TENS[t] + UNITS[nxt]))
                i += 2
                continue
            if nxt in ORDINALS and 1 <= ORDINALS[nxt] <= 9:
                n = TENS[t] + ORDINALS[nxt]
                out.append(f"{n}{_suffix(n)}")
                i += 2
                continue
            out.append(str(TENS[t]))
        elif t in UNITS and nxt == "hundred":
            out.append(str(UNITS[t] * 100))
            i += 2
            continue
        elif t in UNITS and t != "one":
            out.append(str(UNITS[t]))
        elif t in ORDINALS and t not in {"first", "second"}:
            out.append(f"{ORDINALS[t]}{_suffix(ORDINALS[t])}")
        else:
            out.append({"1st": "first", "2nd": "second"}.get(t, t))
        i += 1
    return out


def normalize(text: str, digits: bool = True) -> list[str]:
    """Lowercase, drop punctuation (keep apostrophes inside words), split on whitespace, numbers to digits."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", text)
    tokens = text.split()
    return numbers_to_digits(tokens) if digits else tokens


def _dp(ref: list[str], hyp: list[str]) -> list[list[int]]:
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
    return d


def alignment(ref: list[str], hyp: list[str]) -> tuple[list[tuple[int, int]], tuple[int, int, int]]:
    """Levenshtein alignment: (matched (ref_idx, hyp_idx) pairs, (substitutions, deletions, insertions))."""
    d = _dp(ref, hyp)
    i, j = len(ref), len(hyp)
    pairs: list[tuple[int, int]] = []
    s = de = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] == hyp[j - 1]:
                pairs.append((i - 1, j - 1))
            else:
                s += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            de += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    pairs.reverse()
    return pairs, (s, de, ins)


def edit_counts(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions)."""
    return alignment(ref, hyp)[1]


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
