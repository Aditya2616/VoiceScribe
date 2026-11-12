"""Speech-side evaluation: LibriSpeech WER, and the synthetic demo meeting against its script."""

from __future__ import annotations

import datetime as dt
import json
import statistics
import time
from pathlib import Path

from .. import audio as audio_mod
from ..asr import Transcriber
from ..extract.base import Extractor
from ..schema import Meeting, Word
from .extract_eval import load_meeting
from .wer import alignment, corpus_wer, normalize, wer


def librispeech_wer(directory: Path, transcriber: Transcriber) -> dict:
    rows = [line.split("\t", 1) for line in (directory / "refs.tsv").read_text(encoding="utf8").splitlines() if line]
    pairs, per, total_audio, total_asr = [], [], 0.0, 0.0
    for uid, ref in rows:
        samples = audio_mod.decode(directory / f"{uid}.flac")
        t0 = time.perf_counter()
        words = transcriber.transcribe(samples)
        total_asr += time.perf_counter() - t0
        total_audio += audio_mod.duration(samples)
        hyp = " ".join(w.text for w in words)
        pairs.append((ref, hyp))
        per.append({"id": uid, "wer": round(wer(ref, hyp), 4), "ref": ref.lower(), "hyp": hyp.strip()})
    return {
        "model": transcriber.name,
        "utterances": len(rows),
        "reference_words": sum(len(normalize(r)) for r, _ in pairs),
        "wer": corpus_wer(pairs),
        "audio_seconds": round(total_audio, 1),
        "rtf": round(total_asr / total_audio, 3),
        "per_utterance": per,
    }


def demo_eval(meeting: Meeting, script: Path, timing: Path, extractor: Extractor) -> dict:
    """Score a processed demo meeting against its script: WER, sentence boundary timing, extraction."""
    truth = json.loads(timing.read_text(encoding="utf8"))["sentences"]
    ref_text = " ".join(t["text"] for t in truth)
    hyp_words: list[Word] = [w for s in meeting.transcript.sentences for w in s.words]
    hyp_text = " ".join(w.text for w in hyp_words)
    ref_tokens = normalize(ref_text)
    # hyp tokens with the originating Word index (a Word may normalise to several tokens or none)
    hyp_tokens: list[str] = []
    hyp_owner: list[int] = []
    for k, w in enumerate(hyp_words):
        for tok in normalize(w.text, digits=False):
            hyp_tokens.append(tok)
            hyp_owner.append(k)
    # compare after digit normalisation on both sides, keeping the word index mapping
    from .wer import numbers_to_digits

    hyp_norm = numbers_to_digits(hyp_tokens)
    if len(hyp_norm) != len(hyp_tokens):  # number merging changed the token count; map by position scan
        hyp_owner = _remap(hyp_tokens, hyp_norm, hyp_owner)
    pairs, (s, d, i) = alignment(ref_tokens, hyp_norm)
    wer_value = (s + d + i) / len(ref_tokens)

    # reference token index range of each true sentence
    bounds, pos = [], 0
    for t in truth:
        n = len(normalize(t["text"]))
        bounds.append((pos, pos + n - 1))
        pos += n
    match = dict(pairs)
    start_err, end_err = [], []
    for t, (lo, hi) in zip(truth, bounds, strict=True):
        if lo in match:
            start_err.append(hyp_words[hyp_owner[match[lo]]].start - t["start"])
        if hi in match:
            end_err.append(hyp_words[hyp_owner[match[hi]]].end - t["end"])
    both = start_err + end_err

    # extraction against the script labels, matched by time overlap with the true sentence spans
    gold_script = load_meeting(script, with_speakers=True)
    gold = {g.sentence_id: g for g in gold_script.gold}
    tp_a = tp_d = 0
    owner_ok = due_ok = 0
    used: set[int] = set()
    for a in meeting.action_items:
        hit = _overlap(a.source.start, a.source.end, truth, {k for k, g in gold.items() if g.kind == "action"} - used)
        if hit is not None:
            used.add(hit)
            tp_a += 1
            owner_ok += (a.owner or None) == gold[hit].owner
            due_ok += a.deadline_date == gold[hit].due
    used_d: set[int] = set()
    for dcs in meeting.decisions:
        hit = _overlap(dcs.source.start, dcs.source.end, truth, {k for k, g in gold.items() if g.kind == "decision"} - used_d)
        if hit is not None:
            used_d.add(hit)
            tp_d += 1
    n_a = sum(g.kind == "action" for g in gold.values())
    n_d = sum(g.kind == "decision" for g in gold.values())
    return {
        "asr_model": meeting.transcript.model,
        "reference_words": len(ref_tokens),
        "wer": wer_value,
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "boundary_error_ms": {
            "n": len(both),
            "median_abs": round(1000 * statistics.median(abs(x) for x in both), 0) if both else None,
            "p90_abs": round(1000 * sorted(abs(x) for x in both)[int(0.9 * (len(both) - 1))], 0) if both else None,
            "within_250ms": round(sum(abs(x) <= 0.25 for x in both) / len(both), 3) if both else None,
            "mean_start_signed": round(1000 * statistics.mean(start_err), 0) if start_err else None,
            "mean_end_signed": round(1000 * statistics.mean(end_err), 0) if end_err else None,
        },
        "action_items": _prf(tp_a, len(meeting.action_items), n_a),
        "decisions": _prf(tp_d, len(meeting.decisions), n_d),
        "owner_accuracy": owner_ok / tp_a if tp_a else None,
        "due_date_accuracy": due_ok / tp_a if tp_a else None,
        "hypothesis": hyp_text,
        "extractor": extractor.method,
    }


def _remap(before: list[str], after: list[str], owner: list[int]) -> list[int]:
    """Approximate owner indices after number merging (merged tokens take the first token's owner)."""
    out, i = [], 0
    for tok in after:
        out.append(owner[min(i, len(owner) - 1)])
        i += max(1, _consumed(before, i, tok))
    return out


def _consumed(before: list[str], i: int, merged: str) -> int:
    for n in (3, 2, 1):
        from .wer import numbers_to_digits

        if i + n <= len(before) and numbers_to_digits(before[i : i + n]) == [merged]:
            return n
    return 1


def _overlap(start: float, end: float, truth: list[dict], candidates: set[int]) -> int | None:
    best, best_ov = None, 0.0
    for k in candidates:
        t = truth[k]
        ov = min(end, t["end"]) - max(start, t["start"])
        if ov > best_ov and ov >= 0.3 * (t["end"] - t["start"]):
            best, best_ov = k, ov
    return best


def _prf(tp: int, pred: int, gold: int) -> dict:
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp, "predicted": pred, "gold": gold}


def run_demo_eval(
    audio: Path, script: Path, timing: Path, transcriber: Transcriber, extractor: Extractor, diarise: bool = False
) -> tuple[dict, Meeting]:
    from ..pipeline import process_file

    gold = load_meeting(script)
    meeting, _ = process_file(
        audio, gold.date, transcriber=transcriber, extractor=extractor, diarise=diarise, synthetic=True
    )
    assert isinstance(meeting.date, dt.date)
    return demo_eval(meeting, script, timing, extractor), meeting
