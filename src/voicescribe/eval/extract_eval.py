"""Precision / recall for action items and decisions on the hand-labelled meeting set.

File format (data/eval/extract/*.txt): header lines "# key: value" (title, date, split) then one
sentence per line, "Speaker: text", optionally followed by " => ACTION owner=X due=YYYY-MM-DD" or
" => DECISION". Empty owner / due means none.

Matching: a predicted item matches a gold item of the same kind when their sentence sets intersect;
each gold item is matched at most once. Precision = matched / predicted, recall = matched / gold.
Owner accuracy and due-date accuracy are scored over matched action items only.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..extract.base import Extractor
from ..schema import Sentence, Transcript

LABEL = re.compile(r"\s*=>\s*(ACTION|DECISION)(.*)$")


@dataclass
class Gold:
    kind: str
    sentence_id: int
    owner: str | None = None
    due: dt.date | None = None


@dataclass
class EvalMeeting:
    name: str
    title: str
    date: dt.date
    split: str
    transcript: Transcript
    gold: list[Gold] = field(default_factory=list)


def load_meeting(path: Path, with_speakers: bool = False) -> EvalMeeting:
    meta: dict[str, str] = {}
    sents: list[Sentence] = []
    gold: list[Gold] = []
    t = 0.0
    for raw in path.read_text(encoding="utf8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            k, _, v = line[1:].partition(":")
            meta[k.strip()] = v.strip()
            continue
        label = LABEL.search(line)
        body = LABEL.sub("", line)
        speaker, _, text = body.partition(":")
        sid = len(sents)
        dur = max(1.0, len(text.split()) * 0.4)
        sents.append(
            Sentence(id=sid, start=t, end=t + dur, text=text.strip(), speaker=speaker.strip() if with_speakers else None)
        )
        t += dur + 0.3
        if label:
            kind = label.group(1).lower()
            fields = dict(re.findall(r"(\w+)=(\S*)", label.group(2)))
            owner = fields.get("owner") or None
            due = dt.date.fromisoformat(fields["due"]) if fields.get("due") else None
            gold.append(Gold(kind, sid, owner, due))
    return EvalMeeting(
        name=path.stem,
        title=meta.get("title", path.stem),
        date=dt.date.fromisoformat(meta["date"]),
        split=meta.get("split", "dev"),
        transcript=Transcript(duration=t, sentences=sents),
        gold=gold,
    )


def load_all(directory: Path, split: str | None = None, with_speakers: bool = False) -> list[EvalMeeting]:
    meetings = [load_meeting(p, with_speakers) for p in sorted(directory.glob("*.txt"))]
    wanted = set(split.split(",")) if split else None
    return [m for m in meetings if wanted is None or m.split in wanted]


def _prf(tp: int, pred: int, gold: int) -> dict[str, float]:
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "predicted": pred, "gold": gold}


def score(meetings: list[EvalMeeting], extractor: Extractor, people_from_speakers: bool = False) -> dict:
    counts = {"action": [0, 0, 0], "decision": [0, 0, 0]}  # tp, pred, gold
    owner_ok = owner_n = due_ok = due_n = 0
    own = {"correct": 0, "wrong": 0, "unassigned": 0}  # over matched items: a wrong name is worse than unassigned
    errors: list[str] = []
    for m in meetings:
        people = sorted({s.speaker for s in m.transcript.sentences if s.speaker}) if people_from_speakers else None
        actions, decisions, _ = extractor.extract(m.transcript, m.date, people)
        gold_a = [g for g in m.gold if g.kind == "action"]
        gold_d = [g for g in m.gold if g.kind == "decision"]
        used: set[int] = set()
        for a in actions:
            hit = next(
                (i for i, g in enumerate(gold_a) if i not in used and g.sentence_id in a.source.sentence_ids), None
            )
            counts["action"][1] += 1
            if hit is None:
                errors.append(f"[{m.name}] FP action: {a.source.quote!r} -> {a.task!r}")
                continue
            used.add(hit)
            counts["action"][0] += 1
            g = gold_a[hit]
            owner_n += 1
            owner_ok += (a.owner or None) == g.owner
            if g.owner:
                own["correct" if a.owner == g.owner else "unassigned" if not a.owner else "wrong"] += 1
            elif a.owner:  # a name where the gold has none is also a wrong owner
                own["wrong"] += 1
            due_n += 1
            due_ok += a.deadline_date == g.due
            if (a.owner or None) != g.owner or a.deadline_date != g.due:
                errors.append(
                    f"[{m.name}] wrong field: owner {a.owner!r} vs {g.owner!r}, due {a.deadline_date} vs {g.due}: {a.source.quote!r}"
                )
        for i, g in enumerate(gold_a):
            if i not in used:
                errors.append(f"[{m.name}] FN action: {m.transcript.sentences[g.sentence_id].text!r}")
        counts["action"][2] += len(gold_a)
        used_d: set[int] = set()
        for d in decisions:
            hit = next(
                (i for i, g in enumerate(gold_d) if i not in used_d and g.sentence_id in d.source.sentence_ids), None
            )
            counts["decision"][1] += 1
            if hit is None:
                errors.append(f"[{m.name}] FP decision: {d.source.quote!r}")
                continue
            used_d.add(hit)
            counts["decision"][0] += 1
        for i, g in enumerate(gold_d):
            if i not in used_d:
                errors.append(f"[{m.name}] FN decision: {m.transcript.sentences[g.sentence_id].text!r}")
        counts["decision"][2] += len(gold_d)
    return {
        "meetings": len(meetings),
        "sentences": sum(len(m.transcript.sentences) for m in meetings),
        "action_items": _prf(*counts["action"]),
        "decisions": _prf(*counts["decision"]),
        "owner_accuracy": owner_ok / owner_n if owner_n else None,
        "owners": {**own, "n": sum(own.values())},
        "due_date_accuracy": due_ok / due_n if due_n else None,
        "matched_actions": owner_n,
        "errors": errors,
    }
