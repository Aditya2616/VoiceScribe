"""Deterministic rule-based extractor: the offline baseline and the fallback for every other extractor.

It works on sentences (with optional speaker labels). Cues are explicit commitments ("I'll ..."),
assignments ("Rohan, can you ..." / "Maya will ..."), requests, obligations ("we need to ...") and
decisions ("we agreed to ..."). Deadlines are resolved against the meeting date by dates.py.
"""

from __future__ import annotations

import datetime as dt
import itertools
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..dates import find_deadline
from ..schema import ActionItem, Decision, Sentence, Transcript, make_span

TASK_VERBS = (
    "send|write|draft|review|update|book|schedule|prepare|check|share|fix|ship|test|confirm|follow up|talk|call|email|"
    "ping|set up|create|build|deploy|finalize|finalise|investigate|look into|look at|collect|run|publish|post|file|"
    "order|arrange|pull|own|handle|take|coordinate|reach out|get|make|add|remove|migrate|document|circulate|sync|"
    "organize|organise|finish|complete|kick off|present|demo|clean|plan|research|benchmark|measure|rotate|renew|"
    "approve|sign|design|mock up|rewrite|record|summarize|summarise|compile|gather|raise|open|close|cut|merge|"
    "release|upgrade|install|configure|verify|validate|prototype|escalate|notify|inform|introduce|loop in|chase|"
    "track|submit|log|revise|refactor|monitor|archive|prioritize|prioritise|do|bring|put together|figure out|"
    "sort out|pick up|message|ask|find|list|write down|reach|schedule|decide|talk to|speak"
)
VERB = rf"(?:{TASK_VERBS})\b"
NAME = r"[A-Z][a-z]{1,15}"
NOT_NAMES = frozenset(
    "I We You They He She It The This That Okay Ok So And But Sure Yes No Thanks Thank Right Alright Great Good Well "
    "Please Let Can Could Would Will Should Also Then Now Next Monday Tuesday Wednesday Thursday Friday Saturday "
    "Sunday January February March April May June July August September October November December Yeah Hi Hello "
    "Anyway Maybe Perhaps If Do Did Does What Who When Where Why How One Two Three Action Decision Team Everyone "
    "Someone Anyone Nobody Fine Cool Done Agreed Perfect Absolutely Definitely Just Not Today Tomorrow".split()
)
FILLER = (
    r"(?i:(?:ok|okay|so|and|right|alright|all right|great|thanks|thank you|yeah|yes|sure|well|also|then|now|cool|"
    r"perfect|fine)\b[,.]?\s*)*"
)

SELF = re.compile(
    r"\b(?:i'll|i will|i'm going to|i am going to|i'm gonna|i can|i could|let me|i'd be happy to|i should|"
    r"i need to|i volunteer to|i'm on it)\b",
    re.I,
)
OBLIGATION = re.compile(
    r"\b(?:we|someone|somebody|everyone|the team)\s+(?:need|needs|have|has|should|must|ought)\s+to\b|"
    r"\b(?:needs|need) to be\b|\baction item\b|\bdon't forget to\b|\bremember to\b|\bmake sure (?:we|to|that)\b|"
    r"\btodo\b|\bto-do\b",
    re.I,
)
REQUEST = re.compile(r"\b(?:can|could|would|will) you\b|\bplease\b|\bif you could\b|\bdo you mind\b", re.I)
ACK_WORDS = re.compile(
    r"^\s*(?:(?:yes|yeah|yep|sure|okay|ok|absolutely|of course|got it|no problem|sounds good|will do|on it|"
    r"happy to|definitely|right|fine|good idea|good point|understood|noted|of course)\b[,.!]?\s*)+",
    re.I,
)
HEDGE = re.compile(r"\b(?:probably|maybe|might|perhaps|possibly|i guess|i was thinking|someday|at some point)\b", re.I)
LIGHT_VERBS = {"do", "take", "handle", "get", "make", "ask"}
ASSIGN_ADDRESS = re.compile(
    rf"^{FILLER}(?P<name>{NAME})\s*,?\s+(?:can|could|would|will|should|please|if you|do you mind|you(?:'ll| will)|"
    rf"you can|you could|you should|you need to|you might want to)\b|"
    rf"^{FILLER}(?P<name2>{NAME})\s*,\s+(?={VERB})",
)
ASSIGN_THIRD = re.compile(
    rf"\b(?P<name>{NAME})\s+(?:will|is going to|'s going to|can|should|needs to|has to|to|owns|is taking|takes|"
    rf"agreed to|volunteered to|is gonna)\s+(?P<verb>{VERB})"
)
ASSIGN_HAVE = re.compile(rf"\b(?:have|ask|get)\s+(?P<name>{NAME})\s+(?:to\s+)?(?P<verb>{VERB})")
ASSIGN_TO = re.compile(rf"\b(?:assign(?:ed)?|give|hand)\s+(?:it |this |that )?to\s+(?P<name>{NAME})\b")
DECISION = re.compile(
    r"\bwe(?:'ve| have)?\s+(?:decided|agreed|settled|resolved)\b|"
    r"\bwe(?:'re| are)\s+going\s+(?:with|to\s+(?:go|ship|move|use|drop|keep|switch|stick|postpone|cancel|push|launch))\b|"
    r"\blet's\s+(?:go with|ship|move|keep|drop|use|stick|switch|push|postpone|cancel|launch|pick|delay|skip|hold)\b|"
    r"\b(?:the|our)\s+decision\s+is\b|\bdecision:|\b(?:it's|that's|that is|it is)\s+(?:decided|settled|approved)\b|"
    r"\bwe'll\s+(?:go with|use|keep|ship|stick|drop|switch|postpone|push|delay|skip|launch|hold)\b|"
    r"\bwe(?:'re| are)\s+(?:dropping|postponing|moving|switching|keeping|cancelling|canceling|delaying|going ahead)\b|"
    r"\bis approved\b|\bit's a go\b",
    re.I,
)
CONFIRM = re.compile(r"\b(?:approved|agreed|settled|decided|it's a go)\b", re.I)
QUESTION_START = re.compile(r"^\s*(?:should|shall|do|does|did|is|are|what|why|how|when|which|who|where)\b", re.I)
NEGATED = re.compile(r"\b(?:i can't|i cannot|i won't|i will not|i couldn't|we can't|not going to|no need to)\b", re.I)


ACK_TAIL = re.compile(
    r"^(?:go ahead and\s+)?(?:(?:do|take|handle|get|make)\s+(?:care of\s+)?(?:it|that|this|them|those)\b|"
    r"have\s+[\w ]{1,24}?\s+ready\b)?"
    r"(?:[\s,.!]+(?:yes|yep|sure|today|tomorrow|now|then|later|too|right away|this \w+|next \w+|"
    r"by \w+(?: \w+)?|before \w+(?: \w+)?|on \w+|at \w+))*[\s,.!]*$",
    re.I,
)
TRAILING_VOCATIVE = re.compile(rf",\s*(?P<name>{NAME})\s*[?.!]*$")
THANKS_NAME = re.compile(
    rf"^{FILLER}(?i:thanks|thank you|great|good|perfect|nice|cool)[,.!]?\s*(?i:thanks|thank you)?[,\s]*(?P<name>{NAME})[.!]*$"
)
SELF_START = re.compile(rf"^{FILLER}(?:i'll|i will|i can|i'm on it|let me|will do)\b", re.I)


def is_ack(text: str) -> bool:
    """A short reply that accepts a request: "Sure, I'll do that tomorrow.", "Will do.", "I can take that, yes."."""
    if len(text.split()) > 12 or text.rstrip().endswith("?"):
        return False
    m = ACK_WORDS.match(text)
    body = text[m.end() :] if m else text
    cue = SELF.match(body.strip())
    if cue:
        return bool(ACK_TAIL.match(body.strip()[cue.end() :].strip()))
    if not m:
        return False
    rest = body.strip(" .,!")
    return not rest or (len(rest.split()) <= 3 and not re.search(VERB, rest, re.I))


def is_reply(prev_text: str, text: str) -> bool:
    """Is `text` the answer to the request in `prev_text`? Covers heavy-verb acceptances such as
    "Sure, I'll reach out today." that only make sense as replies to a question or an addressed request."""
    if is_ack(text):
        return True
    m = ACK_WORDS.match(text)
    if not m or not SELF.match(text[m.end() :].strip()):
        return False
    return prev_text.rstrip().endswith("?") or ASSIGN_ADDRESS.match(prev_text) is not None


@dataclass
class Candidate:
    kind: str  # "action" | "decision"
    sentence_ids: list[int]
    task: str
    owner: str | None
    deadline_text: str | None
    deadline_date: dt.date | None
    confidence: float
    context: list[int] = field(default_factory=list)  # neighbouring sentence ids (for the local model)


def clean_task(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip(" ,.;:?!-")
    t = re.sub(r"^(?:and|so|then|also|to|please|you)\s+", "", t, flags=re.I)
    t = re.sub(r"\s+(?:for|by|on|until|before|at|in|to|due|within)$", "", t, flags=re.I)
    return t[:1].upper() + t[1:] if t else t


def _strip_deadline(text: str, phrase: str | None) -> str:
    if phrase:
        text = re.sub(re.escape(phrase), "", text, count=1, flags=re.I)
    return re.sub(r"\s+,", ",", text)


SELF_INTRO = re.compile(
    rf"^{FILLER}(?i:hi|hello|hey)?[,.!]?\s*(?i:this is|it's|i'm|i am|my name is)\s+(?P<name>{NAME})\b"
)
MIN_NAME_CONFIDENCE = 0.6  # a speaker is only named when this share of its name evidence agrees


MIN_VOTE_SECONDS = 1.0  # a sentence shorter than this is too short for its speaker label to be trusted


def infer_speaker_names(
    sentences: list[Sentence], min_confidence: float = MIN_NAME_CONFIDENCE, min_seconds: float = MIN_VOTE_SECONDS
) -> dict[str, str]:  # fmt: skip
    """Bind diarisation labels to first names, only where the dialogue is clear.

    Evidence (one vote each): a request addressed to X ("X, can you ...?") answered by a different speaker; "Thanks, X."
    said right after a speaker's turn; a self-introduction ("I'm X", "this is X"). A label gets a name when at least
    `min_confidence` of its votes agree, votes only come from sentences of at least `min_seconds` (a one-word "Sure." is
    often clustered with the wrong speaker), and each name goes to one label only (the stronger claim wins, a tie names
    neither), so a merged or split cluster leaves speakers unnamed rather than wrongly named.
    """
    votes: dict[str, Counter] = defaultdict(Counter)
    for i, s in enumerate(sentences):
        intro = SELF_INTRO.match(s.text.strip())
        if intro and s.speaker and s.end - s.start >= min_seconds and intro.group("name") not in NOT_NAMES:
            votes[s.speaker][intro.group("name")] += 1
        t = THANKS_NAME.match(s.text.strip())
        prev = sentences[i - 1] if i else None
        if (t and prev and s.speaker and prev.speaker not in (None, s.speaker) and prev.end - prev.start >= min_seconds
                and t.group("name") not in NOT_NAMES):  # fmt: skip
            votes[prev.speaker][t.group("name")] += 1  # "Thanks, Lena." names the previous speaker
        m = ASSIGN_ADDRESS.match(s.text)
        if not (m and s.speaker and (m.group("name") or m.group("name2")) not in NOT_NAMES):
            continue
        # the reply is the next sentence (or the one after a one-word "Sure."), said by someone other than the asker
        for j in (i + 1, i + 2):
            if j >= len(sentences) or not sentences[j].speaker:
                break
            nxt = sentences[j]
            if nxt.speaker == s.speaker:
                break  # the asker kept talking (or a short reply was clustered with them): no reliable evidence
            if nxt.end - nxt.start < min_seconds and j == i + 1:
                continue
            if nxt.end - nxt.start >= min_seconds and (is_ack(nxt.text) or SELF.search(nxt.text)):
                votes[nxt.speaker][m.group("name") or m.group("name2")] += 1
            break
    claims = sorted(((c, label, name) for label, cnt in votes.items() for name, c in cnt.items()
                     if c / sum(cnt.values()) >= min_confidence), key=lambda x: -x[0])  # fmt: skip
    out: dict[str, str] = {}
    taken: dict[str, int] = {}
    for c, label, name in claims:
        if label in out or name in taken:
            if name in taken and taken[name] == c and out.get(label) != name:
                # a tie between two speakers for one name: name neither
                for lbl in [k for k, v in out.items() if v == name]:
                    del out[lbl]
            continue
        out[label] = name
        taken[name] = c
    return out


def ruled_out_names(sentences: list[Sentence]) -> set[tuple[str, str]]:
    """(speaker, name) pairs the dialogue rules out: someone who addresses X, thanks X or calls out X is not X."""
    out: set[tuple[str, str]] = set()
    for s in sentences:
        if not s.speaker:
            continue
        names: set[str] = set()
        if m := ASSIGN_ADDRESS.match(s.text):
            names.add(m.group("name") or m.group("name2"))
        if m := THANKS_NAME.match(s.text.strip()):
            names.add(m.group("name"))
        if m := TRAILING_VOCATIVE.search(s.text):
            names.add(m.group("name"))
        out |= {(s.speaker, n) for n in names if n and n not in NOT_NAMES}
    return out


def bind_by_elimination(
    labels: set[str], people: list[str], named: dict[str, str], ruled_out: set[tuple[str, str]], max_labels: int = 8
) -> dict[str, str]:  # fmt: skip
    """With as many speakers as participants, bind only what the evidence forces: every one-to-one assignment that keeps
    the names already bound and breaks no ruled-out pair is tried, and a speaker gets a name only when all the consistent
    assignments agree. Inconsistent evidence, or a speaker/participant count mismatch, binds nothing."""
    if not people or len(labels) != len(people) or len(labels) > max_labels:
        return {}
    order = sorted(labels)
    solutions = []
    for perm in itertools.permutations(people):
        assign = dict(zip(order, perm, strict=True))
        if all(assign.get(label) == name for label, name in named.items() if name in people and label in assign) and not any(
            assign.get(label) == name for label, name in ruled_out
        ):  # fmt: skip
            solutions.append(assign)
    if not solutions:
        return {}
    return {label: solutions[0][label] for label in order if all(sol[label] == solutions[0][label] for sol in solutions)}


def name_speakers(sentences: list[Sentence], people: list[str] | None = None) -> dict[str, str]:
    """Diarisation label -> name for the speakers the dialogue (and a participants list) pins down."""
    return RuleExtractor._speaker_names(sentences, people)


class RuleExtractor:
    """Rule-based action-item and decision extraction."""

    method = "rules"

    def __init__(self, strict: bool = True):
        # strict=False also yields weak candidates, meant for a model to verify (never shown as final)
        self.strict = strict

    def candidates(
        self, transcript: Transcript, meeting_date: dt.date, people: list[str] | None = None
    ) -> list[Candidate]:
        sents = transcript.sentences
        speaker_names = self._speaker_names(sents, people)
        out: list[Candidate] = []
        for idx, s in enumerate(sents):
            text = s.text
            last = out[-1] if out and out[-1].sentence_ids[-1] == s.id - 1 else None
            # a short acceptance belongs to the request just before it
            if last and last.kind == "action" and is_reply(sents[idx - 1].text, text):
                last.sentence_ids.append(s.id)
                last.owner = last.owner or speaker_names.get(s.speaker or "")
                if last.deadline_date is None and (dl := find_deadline(text, meeting_date)):
                    last.deadline_text, last.deadline_date = dl
                last.confidence = min(0.95, last.confidence + 0.1)
                continue
            if is_ack(text) and not DECISION.search(text):
                continue  # a bare acceptance with no request before it carries no task of its own
            owner, rest, strength = self._owner_and_rest(s, idx, speaker_names)
            if DECISION.search(text) and not (owner and strength >= 0.8) and not QUESTION_START.match(text):
                if last and last.kind == "decision" and len(text.split()) <= 8 and CONFIRM.search(text):
                    # a short confirmation ("Yes, that is approved.") belongs to the decision before it
                    last.sentence_ids.append(s.id)
                    continue
                dl = find_deadline(text, meeting_date)
                out.append(
                    Candidate("decision", [s.id], clean_task(text), None, dl[0] if dl else None,
                              dl[1] if dl else None, 0.8, self._ctx(sents, idx))
                )  # fmt: skip
                continue
            if rest is None:
                continue
            dl = find_deadline(text, meeting_date)
            task = clean_task(_strip_deadline(rest, dl[0] if dl else None))
            if len(task.split()) < 2:
                continue
            out.append(
                Candidate("action", [s.id], task, owner, dl[0] if dl else None, dl[1] if dl else None, strength,
                          self._ctx(sents, idx))
            )  # fmt: skip
        self._bind_thanked_owner(out, sents)
        return out

    @staticmethod
    def _bind_thanked_owner(cands: list[Candidate], sents: list[Sentence]) -> None:
        """A first-person commitment with no known speaker, answered by "Thanks, Mei.", belongs to Mei: the
        reply is addressed to whoever just made the commitment."""
        by_id = {s.id: s for s in sents}
        for c in cands:
            if c.kind != "action" or c.owner:
                continue
            if not any(SELF.search(by_id[i].text) or SELF_START.match(by_id[i].text) for i in c.sentence_ids):
                continue
            nxt = by_id.get(c.sentence_ids[-1] + 1)
            m = THANKS_NAME.match(nxt.text.strip()) if nxt and len(nxt.text.split()) <= 4 else None
            if m and m.group("name") not in NOT_NAMES:
                c.owner = m.group("name")

    def extract(
        self, transcript: Transcript, meeting_date: dt.date, people: list[str] | None = None
    ) -> tuple[list[ActionItem], list[Decision]]:
        actions: list[ActionItem] = []
        decisions: list[Decision] = []
        for c in self.candidates(transcript, meeting_date, people):
            if self.strict and c.confidence < 0.5:
                continue
            span = make_span(transcript, c.sentence_ids)
            if c.kind == "decision":
                decisions.append(Decision(text=c.task, source=span, method="rules"))
            else:
                actions.append(
                    ActionItem(
                        task=c.task, owner=c.owner, deadline_text=c.deadline_text, deadline_date=c.deadline_date,
                        source=span, confidence=c.confidence, method="rules",
                    )
                )  # fmt: skip
        return actions, decisions

    @staticmethod
    def _ctx(sents: list[Sentence], idx: int) -> list[int]:
        return [sents[j].id for j in range(max(0, idx - 1), min(len(sents), idx + 2))]

    @staticmethod
    def _speaker_names(sents: list[Sentence], people: list[str] | None) -> dict[str, str]:
        """Speaker label -> display name. Labels that already look like names map to themselves."""
        mapping = infer_speaker_names(sents)
        labels = {s.speaker for s in sents if s.speaker and not re.fullmatch(NAME, s.speaker)}
        if people:
            for label, name in bind_by_elimination(labels, people, mapping, ruled_out_names(sents)).items():
                mapping.setdefault(label, name)
        for s in sents:
            if s.speaker and re.fullmatch(NAME, s.speaker):
                mapping[s.speaker] = s.speaker
        return mapping

    def _owner_and_rest(
        self, s: Sentence, idx: int, speaker_names: dict[str, str]
    ) -> tuple[str | None, str | None, float]:
        """Return (owner, remaining task text, strength); rest is None when the sentence is not an action."""
        text = s.text
        if NEGATED.search(text):
            return None, None, 0.0
        m = ASSIGN_ADDRESS.match(text)
        if m and (m.group("name") or m.group("name2")) not in NOT_NAMES:
            who = m.group("name") or m.group("name2")
            after = text[m.end("name") if m.group("name") else m.end("name2") :].lstrip(" ,")
            task = re.sub(
                r"^(?:can|could|would|will|should|please|if you could|do you mind|you(?:'ll| will| can| could| should|"
                r" need to| might want to)?)\s+(?:you\s+)?(?:please\s+)?(?:also\s+)?(?:be able to\s+)?",
                "", after, flags=re.I,
            )  # fmt: skip
            if re.search(VERB, task, re.I):
                return who, task, 0.9
        for pattern, strength in ((ASSIGN_HAVE, 0.85), (ASSIGN_THIRD, 0.85)):
            m = pattern.search(text)
            if m and m.group("name") not in NOT_NAMES:
                return m.group("name"), text[m.start("verb") :], strength
        m = ASSIGN_TO.search(text)
        if m and m.group("name") not in NOT_NAMES:
            return m.group("name"), text, 0.7
        owner = speaker_names.get(s.speaker or "")
        m = SELF.search(text)
        if m and not QUESTION_START.match(text) and not HEDGE.search(text):
            rest = text[m.end() :]
            if re.search(VERB, rest, re.I):
                return owner, rest, 0.85 if owner else 0.75
            return (owner, rest, 0.3) if not self.strict else (None, None, 0.0)
        m = REQUEST.search(text)
        if m and not QUESTION_START.match(text):
            rest = text[m.end() :]
            if re.search(VERB, rest, re.I):
                tail = TRAILING_VOCATIVE.search(rest)  # "Could you pull the numbers, Sam?" is addressed to Sam
                if tail and tail.group("name") not in NOT_NAMES:
                    return tail.group("name"), rest[: tail.start()], 0.85
                return None, rest, 0.6
        m = OBLIGATION.search(text)
        if m:
            rest = text[m.end() :]
            if re.search(VERB, rest, re.I):
                return None, rest, 0.6
            if not self.strict:
                return None, rest or text, 0.3
        return None, None, 0.0
