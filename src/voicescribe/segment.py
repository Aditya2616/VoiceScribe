"""Word cleanup, sentence segmentation, speaker assignment and topic segmentation."""

from __future__ import annotations

import math
import re
from collections import Counter

from .schema import Sentence, Topic, Word

ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "etc.", "e.g.", "i.e.", "no.", "approx."}
TERMINAL = (".", "?", "!", "…")
MAX_WORD_SECONDS = 1.0


def clean_words(words: list[Word]) -> list[Word]:
    """Make word timings monotonic and bounded.

    Whisper's attention alignment sometimes stretches a word across a pause or lets neighbours
    overlap. Starts are made non-decreasing, a word never ends after the next one starts, and a
    word is capped at MAX_WORD_SECONDS (except the last one).
    """
    out: list[Word] = []
    for w in words:
        text = w.text.strip()
        if not text:
            continue
        start = max(w.start, out[-1].start if out else 0.0)
        out.append(Word(text=text, start=start, end=max(w.end, start), prob=w.prob))
    for i, w in enumerate(out):
        end = w.end
        if i + 1 < len(out):
            end = min(end, out[i + 1].start)
        if i + 1 < len(out):
            end = min(end, w.start + MAX_WORD_SECONDS)
        out[i] = Word(text=w.text, start=w.start, end=max(end, w.start), prob=w.prob)
    return out


def _is_terminal(token: str) -> bool:
    if token.lower() in ABBREVIATIONS:
        return False
    return token.rstrip("\"')]”’").endswith(TERMINAL)


def split_sentences(
    words: list[Word],
    speakers: list[str | None] | None = None,
    pause: float = 1.2,
    max_words: int = 45,
) -> list[Sentence]:
    """Group words into sentences.

    A sentence ends at terminal punctuation. Without punctuation it also ends on a long pause,
    on a speaker change, or after max_words words, so unpunctuated ASR output still segments.
    """
    sentences: list[Sentence] = []
    cur: list[int] = []

    def flush() -> None:
        if not cur:
            return
        ws = [words[i] for i in cur]
        spk = None
        if speakers:
            c = Counter(speakers[i] for i in cur if speakers[i])
            spk = c.most_common(1)[0][0] if c else None
        text = _join(ws)
        sentences.append(
            Sentence(id=len(sentences), start=ws[0].start, end=ws[-1].end, text=text, speaker=spk, words=ws)
        )
        cur.clear()

    for i, w in enumerate(words):
        if cur:
            gap = w.start - words[cur[-1]].end
            changed = speakers and speakers[i] and speakers[cur[-1]] and speakers[i] != speakers[cur[-1]]
            if gap >= pause or changed or len(cur) >= max_words:
                flush()
        cur.append(i)
        if _is_terminal(w.text):
            flush()
    flush()
    return sentences


def _join(ws: list[Word]) -> str:
    text = " ".join(w.text for w in ws)
    return re.sub(r"\s+([,.;:?!])", r"\1", text)


def assign_speakers(words: list[Word], turns: list[tuple[float, float, str]]) -> list[str | None]:
    """Give each word the speaker whose diarisation turn overlaps it most."""
    out: list[str | None] = []
    for w in words:
        best, best_ov = None, 0.0
        mid = (w.start + w.end) / 2
        for s, e, spk in turns:
            ov = min(w.end, e) - max(w.start, s)
            if ov > best_ov:
                best, best_ov = spk, ov
        if best is None:  # word fell in a gap: nearest turn to the midpoint
            near = min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])), default=None)
            best = near[2] if near else None
        out.append(best)
    return out


STOPWORDS = set(
    "a an the and or but if so to of in on for with at by from up about as is are was were be been it this that these "
    "those i you he she we they me my our your their his her its do does did have has had will would can could should "
    "not no yes okay ok just also then than there here what which who how when where why all any some more very "
    "really like get got going go let us".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in STOPWORDS and len(t) > 2]


def topic_boundaries(sentences: list[Sentence], window: int = 4, min_gap: int = 4) -> list[Topic]:
    """TextTiling-style topic segmentation over sentence windows (lexical cohesion, no model).

    The cosine similarity between the bag of words before and after each gap is computed;
    gaps that are deep local minima below (mean - 0.5 sd) become topic starts.
    """
    n = len(sentences)
    toks = [Counter(tokens(s.text)) for s in sentences]
    topics = [Topic(start_sentence=0, start=sentences[0].start, label="")] if sentences else []
    if n < 2 * window:
        return _label(sentences, topics)
    sims: list[float] = []
    for g in range(window, n - window + 1):
        a = sum((toks[i] for i in range(g - window, g)), Counter())
        b = sum((toks[i] for i in range(g, g + window)), Counter())
        sims.append(_cos(a, b))
    mean = sum(sims) / len(sims)
    sd = math.sqrt(sum((x - mean) ** 2 for x in sims) / len(sims))
    cutoff = mean - 0.5 * sd
    last = 0
    for k, v in enumerate(sims):
        g = k + window
        left = sims[k - 1] if k > 0 else 1.0
        right = sims[k + 1] if k + 1 < len(sims) else 1.0
        if v < cutoff and v <= left and v <= right and g - last >= min_gap:
            topics.append(Topic(start_sentence=g, start=sentences[g].start, label=""))
            last = g
    return _label(sentences, topics)


# words that say nothing about the subject of a topic: meeting talk, filler and generic verbs
GENERIC = set(
    "let lets start started starting need needs needed want wants think thought know say said says tell thing things "
    "one two three also yes yeah okay great good nice thanks thank please will can could would just really next first "
    "today tomorrow yesterday week month day time still much many lot bit sure fine right well maybe probably sorry "
    "send write look looks looked take took make made give gave check see seen come came put keep kept talk talked "
    "side book everything anything something nothing everyone everybody someone anyone team meeting update updates item items point points sync".split()
)


def _content(text: str, names: set[str]) -> list[str]:
    """Lower-case content words in order: no stopwords, contractions, generic words or names."""
    out = []
    for t in re.findall(r"[a-z0-9']+", text.lower()):
        if "'" in t or len(t) < 3 or t.isdigit() or t in STOPWORDS or t in GENERIC or t in names:
            out.append("")  # a gap, so bigrams never span removed words
        else:
            out.append(t)
    return out


def _title(phrase: str) -> str:
    return " ".join(w.capitalize() for w in phrase.split())


def _label(sentences: list[Sentence], topics: list[Topic]) -> list[Topic]:
    """Label each topic with its most distinctive noun-like phrase (bigram when repeated, else top words);
    when nothing distinctive exists, the first content sentence trimmed to a few words."""
    from .names import name_evidence

    names = {n.lower() for n in name_evidence(sentences)} | {s.speaker.lower() for s in sentences if s.speaker}
    bounds = [t.start_sentence for t in topics] + [len(sentences)]
    segs = [sentences[bounds[k] : bounds[k + 1]] for k in range(len(topics))]
    uni: list[Counter] = []
    bi: list[Counter] = []
    for seg in segs:
        u: Counter = Counter()
        b2: Counter = Counter()
        for sent in seg:
            words = _content(sent.text, names)
            u.update(w for w in words if w)
            b2.update(f"{x} {y}" for x, y in zip(words, words[1:], strict=False) if x and y)
        uni.append(u)
        bi.append(b2)
    df = Counter(w for u in uni for w in u)
    n_seg = max(1, len(segs))
    for k, t in enumerate(topics):
        weight = lambda w, c: c * (1 + math.log(1 + n_seg / df[w]))  # noqa: E731
        ranked = sorted(uni[k], key=lambda w: (-weight(w, uni[k][w]), -len(w), w))
        phrases: list[str] = []
        pairs = [p for p, c in bi[k].items() if c >= 2]
        if pairs:
            best = max(pairs, key=lambda p: (bi[k][p] * (weight(p.split()[0], 1) + weight(p.split()[1], 1)), p))
            phrases.append(best)
        for w in ranked:
            if len(phrases) >= 2 or uni[k][w] < 2:
                break  # a second phrase must be repeated in the segment, otherwise it is noise
            if not any(w in p.split() for p in phrases):
                phrases.append(w)
        if not phrases and ranked:
            phrases.append(ranked[0])
        t.label = " / ".join(_title(p) for p in phrases[:2]) if phrases else _first_sentence_label(segs[k])
    return topics


def _first_sentence_label(seg: list[Sentence]) -> str:
    for sent in seg:
        words = re.sub(r"^(?:(?:okay|ok|so|well|right|and|but|yes|yeah)[,.]?\s+)+", "", sent.text, flags=re.I).split()
        if len(words) >= 3:
            return " ".join(words[:5]).rstrip(",.;:!?") + ("..." if len(words) > 5 else "")
    return seg[0].text[:30] if seg else ""


def _cos(a: Counter, b: Counter) -> float:
    dot = sum(a[k] * b[k] for k in a if k in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0
