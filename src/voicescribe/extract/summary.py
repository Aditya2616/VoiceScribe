"""Extractive summary: TF-IDF centrality over sentences plus a position prior (offline, deterministic)."""

from __future__ import annotations

import math
from collections import Counter

from ..schema import Summary, Transcript
from ..segment import tokens


def extractive_summary(transcript: Transcript, max_sentences: int = 4, min_words: int = 6) -> Summary | None:
    sents = [s for s in transcript.sentences if len(s.text.split()) >= min_words]
    if not sents:
        return None
    docs = [Counter(tokens(s.text)) for s in sents]
    df: Counter = Counter()
    for d in docs:
        df.update(d.keys())
    n = len(docs)
    vecs = [{t: c * math.log((n + 1) / (df[t] + 0.5)) for t, c in d.items()} for d in docs]
    norms = [math.sqrt(sum(v * v for v in vec.values())) or 1.0 for vec in vecs]
    centrality = []
    for i, vi in enumerate(vecs):
        total = 0.0
        for j, vj in enumerate(vecs):
            if i != j:
                total += sum(w * vj.get(t, 0.0) for t, w in vi.items()) / (norms[i] * norms[j])
        centrality.append(total / max(1, n - 1))
    # a mild prior for the opening and closing minutes, and for sentences that are not tiny
    scored = []
    for i, s in enumerate(sents):
        pos = 1.0 + (0.15 if i < max(2, n // 8) else 0.0)
        length = min(1.0, len(s.text.split()) / 14)
        scored.append((centrality[i] * pos * (0.6 + 0.4 * length), i))
    top = sorted(sorted(scored, reverse=True)[:max_sentences], key=lambda x: x[1])
    chosen = [sents[i] for _, i in top]
    return Summary(text=" ".join(s.text for s in chosen), kind="extractive", sentence_ids=[s.id for s in chosen])
