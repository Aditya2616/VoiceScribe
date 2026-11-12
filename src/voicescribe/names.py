"""Name canonicalisation: one person, one spelling.

Speech recognition spells an unusual name differently from one sentence to the next ("Rohan" / "Rohen",
"Lena" / "Lina"), which reads as two people. This module finds the names that the dialogue itself marks as
names (a request addressed to someone, "thanks, X", "X will ..."), clusters near-identical spellings and
rewrites every occurrence to one spelling.

- With a participants list, that list is the closed set: each detected name maps to the closest participant.
- Without one, a cluster takes its most frequent spelling.

Two spellings are merged when they have the same consonant skeleton and differ by at most two edits (both at
least four letters long), which covers vowel slips but keeps "Ben" and "Ken" apart. Limitation: two real people
called, say, "Mara" and "Mary" in one meeting would be merged unless a participants list is given.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

from .extract.rules import ASSIGN_ADDRESS, ASSIGN_THIRD, NAME, NOT_NAMES
from .schema import Sentence, Word

THANKS = re.compile(rf"\b(?i:thanks|thank you)\s*,?\s+(?P<n>{NAME})\b")
VOCATIVE_END = re.compile(rf",\s*(?P<n>{NAME})\s*[.?!]*$")
_VOWELS = set("aeiouy")


def skeleton(name: str) -> str:
    n = name.lower().replace("ph", "f")
    n = n.translate(str.maketrans("cq", "kk"))  # c, k and q sound alike at the start of a name ("Kwame" / "Quame")
    return n[:1] + "".join(c for c in n[1:] if c not in _VOWELS)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similar(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    return skeleton(a) == skeleton(b) and edit_distance(a.lower(), b.lower()) <= 2


def name_evidence(sentences: list[Sentence]) -> Counter[str]:
    """Tokens that the dialogue treats as personal names, with how often. Weak cues ("..., Lena.") only count
    for a token that also has a strong cue or appears twice."""
    strong: Counter[str] = Counter()
    weak: Counter[str] = Counter()
    for s in sentences:
        m = ASSIGN_ADDRESS.match(s.text)
        if m and (n := m.group("name") or m.group("name2")) and n not in NOT_NAMES:
            strong[n] += 1
        for m in ASSIGN_THIRD.finditer(s.text):
            if m.group("name") not in NOT_NAMES:
                strong[m.group("name")] += 1
        m = THANKS.search(s.text)
        if m and m.group("n") not in NOT_NAMES:
            strong[m.group("n")] += 1
        m = VOCATIVE_END.search(s.text)
        if m and m.group("n") not in NOT_NAMES:
            weak[m.group("n")] += 1
        if s.speaker and re.fullmatch(NAME, s.speaker):
            strong[s.speaker] += 1
    out = Counter(strong)
    for n, c in weak.items():
        if n in strong or c >= 2:
            out[n] += c
    return out


def _closest(token: str, participants: list[str]) -> str | None:
    best, score = None, 0.0
    for p in participants:
        if p == token:
            return p
        if similar(p, token):
            s = 1.0 - edit_distance(p.lower(), token.lower()) / max(len(p), len(token))
        else:  # inside a closed list a looser match is safe: same first letter and mostly the same letters
            ratio = SequenceMatcher(None, p.lower(), token.lower()).ratio()
            s = ratio if ratio >= 0.8 or (ratio >= 0.7 and p[:1].lower() == token[:1].lower()) else 0.0
        if s > score:
            best, score = p, s
    return best


def canonical_map(
    sentences: list[Sentence], participants: list[str] | None = None, text: str | None = None
) -> dict[str, str]:
    """Map each variant spelling to its canonical one (only entries that change)."""
    evidence = name_evidence(sentences)
    body = text if text is not None else " ".join(s.text for s in sentences)
    freq = Counter(re.findall(NAME, body))
    mapping: dict[str, str] = {}
    if participants:
        for tok in evidence:
            target = _closest(tok, participants)
            if target and target != tok:
                mapping[tok] = target
        return mapping
    tokens = sorted(evidence, key=lambda t: (-freq[t], t))
    groups: list[list[str]] = []
    for tok in tokens:
        for g in groups:
            if any(similar(tok, other) for other in g):
                g.append(tok)
                break
        else:
            groups.append([tok])
    for g in groups:
        if len(g) > 1:
            canon = max(g, key=lambda t: (freq[t], evidence[t], -g.index(t)))
            mapping.update({t: canon for t in g if t != canon})
    return mapping


def apply_map(words: list[Word], mapping: dict[str, str]) -> list[Word]:
    if not mapping:
        return words
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, mapping)) + r")\b")
    return [w.model_copy(update={"text": pattern.sub(lambda m: mapping[m.group(1)], w.text)}) for w in words]


def apply_to_owner(owner: str | None, participants: list[str] | None) -> str | None:
    """With a participants list, an owner outside it becomes unassigned (a wrong name is worse than none)."""
    if owner and participants and owner not in participants:
        return None
    return owner


def parse_people(text: str | None) -> list[str] | None:
    """Parse a participants field ("Ana, Ben; Chloe"): capitalised first names, at most 30."""
    if not text:
        return None
    out: list[str] = []
    for part in re.split(r"[,;" + chr(92) + r"n]+", text):
        name = part.strip().title() if part.strip().isalpha() else ""
        if 2 <= len(name) <= 20 and name not in out:
            out.append(name)
    return out[:30] or None
