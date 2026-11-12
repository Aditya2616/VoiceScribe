"""Deterministic rule-based extractor: the offline baseline and the fallback for every other extractor.

It works on sentences (with optional speaker labels). Cues are explicit commitments ("I'll ..."),
assignments ("Rohan, can you ..." / "Maya will ..."), requests, obligations ("we need to ...") and
decisions ("we agreed to ..."). Deadlines are resolved against the meeting date by dates.py.
"""

from __future__ import annotations

import datetime as dt
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


def infer_speaker_names(sentences: list[Sentence]) -> dict[str, str]:
    """Map diarisation labels to first names from addressing: "Rohan, can you ..." followed by a
    short acceptance from another speaker means that speaker is Rohan."""
    votes: dict[str, Counter] = defaultdict(Counter)
    for i, s in enumerate(sentences):
        m = ASSIGN_ADDRESS.match(s.text)
        if not (m and s.speaker and m.group("name") and m.group("name") not in NOT_NAMES):
            continue
        for nxt in sentences[i + 1 : i + 3]:
            if nxt.speaker and nxt.speaker != s.speaker:
                if is_ack(nxt.text) or SELF.search(nxt.text):
                    votes[nxt.speaker][m.group("name")] += 1
                break
    return {label: c.most_common(1)[0][0] for label, c in votes.items()}


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
        return out

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
                return None, rest, 0.6
        m = OBLIGATION.search(text)
        if m:
            rest = text[m.end() :]
            if re.search(VERB, rest, re.I):
                return None, rest, 0.6
            if not self.strict:
                return None, rest or text, 0.3
        return None, None, 0.0
