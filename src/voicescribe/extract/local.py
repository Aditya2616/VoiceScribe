"""Local small-model extractors built on the rules.

The rules propose candidates. A small instruct model (Qwen2.5 0.5B to 1.5B) is then used in two narrow,
closed-choice ways instead of as a free generator, which at this size invents content:

* `owners` (the useful one): for an action item the rules left unassigned, the model reads the surrounding
  dialogue and picks the owner from the closed list of names in the meeting, or "unknown". A choice is only taken
  when the model is confident and the name is actually mentioned near the item.
* `verify` (kept for comparison): the model labels each candidate action / decision / none.

Choices are made by scoring each allowed answer under the model (log-probability of the option text), so there
is no JSON to parse and no way to answer outside the list. Deadlines always come from the date resolver.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import re
from collections.abc import Callable

from ..audio import default_threads
from ..names import name_evidence
from ..schema import ActionItem, Decision, Summary, Transcript, make_span
from .rules import NOT_NAMES, Candidate, RuleExtractor

# (messages, options) -> probability of each option (sums to 1)
ChooseFn = Callable[[list[dict[str, str]], list[str]], dict[str, float]]

OWNER_SYSTEM = (
    "You read part of a meeting transcript and decide who is responsible for one task. The task sentence is marked "
    ">>. Answer with exactly one word: a name from the list, or unknown. Transcripts have no speaker labels, so a "
    "sentence like 'I'll do X' only has a known owner when the nearby text shows who is speaking: the sentence just "
    "before addresses that person by name ('Ana, can you ...?'), or the sentence just after thanks or answers "
    "them by name ('Thanks, Mei.'). People mentioned elsewhere in the conversation are NOT the owner. "
    "If the text does not show who is speaking, answer unknown. Most of the time the answer is unknown."
)
def _shot(people: str, lines: list[str], answer: str) -> tuple[str, str]:
    """One few-shot example: lines starting with '>>' are the task sentence, the others are context."""
    body = "\n".join(line if line.startswith(">>") else "   " + line for line in lines)
    return f"People: {people}\n{body}", answer


OWNER_SHOTS = [
    _shot("Ana, Ben, Mei", ["Someone needs to send out a survey.", ">> I'll do it this week.", "Thanks, Mei."], "Mei"),
    _shot("Ana, Ben, Mei", ["Ana, can you review the draft before Monday?", ">> Yes, I will.", "Ok."], "Ana"),
    _shot("Ana, Ben, Mei", ["The staging run finished last night.", ">> I'll write up the rollback plan by Friday.", "Great."], "unknown"),
    _shot(
        "Ana, Ben, Mei",
        ["Ben, the dashboard is down again.", "Mei, please check the alerts.", ">> I'll fix the dashboard after lunch.", "Ok."],
        "unknown",
    ),
    _shot("Ana, Ben, Mei", ["Ana will present the roadmap.", ">> I'll book the room for the review on the 10th.", "Thanks."], "unknown"),
]
LABEL_SYSTEM = (
    "You label one sentence from a meeting transcript with exactly one word: action, decision or none. "
    "action = someone commits to or is asked to do a concrete task. decision = the group decides or agrees on "
    "something. none = discussion, status, opinion or a vague wish."
)
LABEL_SHOTS = [
    ("People: Ana, Ben\n>> Ben, can you send the report by Friday?", "action"),
    ("People: Ana, Ben\n>> We decided to keep the old server for a month.", "decision"),
    ("People: Ana, Ben\n>> The meeting went well and people were happy.", "none"),
    ("People: Ana, Ben\n>> I'll write the summary tomorrow.", "action"),
    ("People: Ana, Ben\n>> Let's go with the second vendor.", "decision"),
    ("People: Ana, Ben\n>> That is not what we expected.", "none"),
]


class LocalChooser:
    """transformers causal LM that picks among given options by log-probability. Device: VOICESCRIBE_DEVICE, else
    cuda when available, else cpu (float16 on cuda, float32 on cpu). CPU threads are capped."""

    def __init__(self, model: str | None = None, threads: int | None = None):
        self.model_name = model or os.environ.get("VOICESCRIBE_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        self.threads = threads or default_threads()
        self._tok = None
        self._model = None
        self.device = "cpu"
        self.dtype = "float32"

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch.set_num_threads(self.threads)
            want = os.environ.get("VOICESCRIBE_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
            self.device = want
            self.dtype = "float16" if want.startswith("cuda") else "float32"
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch.float16 if want.startswith("cuda") else torch.float32
            ).to(want)
            self._model.eval()
        return self._tok, self._model

    def __call__(self, messages: list[dict[str, str]], options: list[str]) -> dict[str, float]:
        import torch

        tok, model = self._load()
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        scores: list[float] = []
        with torch.no_grad():
            for opt in options:
                opt_ids = tok(opt, add_special_tokens=False)["input_ids"]
                ids = torch.tensor([prompt_ids + opt_ids], device=model.device)
                logits = model(ids).logits[0, len(prompt_ids) - 1 : -1].float()
                logp = torch.log_softmax(logits, dim=-1)
                scores.append(float(sum(logp[i, t] for i, t in enumerate(opt_ids))))
        top = max(scores)
        weights = [math.exp(s - top) for s in scores]
        total = sum(weights)
        return {o: w / total for o, w in zip(options, weights, strict=True)}


def candidate_people(transcript: Transcript, people: list[str] | None) -> list[str]:
    """Closed list of names for the model: participants if given, else names the dialogue itself marks as names."""
    if people:
        return sorted(set(people))
    return sorted(n for n in name_evidence(transcript.sentences) if n not in NOT_NAMES)


def _window(by_id: dict, ids: list[int], before: int = 5, after: int = 2) -> list[int]:
    lo, hi = min(ids) - before, max(ids) + after
    return [i for i in sorted(by_id) if lo <= i <= hi]


def _render(by_id: dict, window: list[int], marked: list[int]) -> str:
    out = []
    for i in window:
        s = by_id[i]
        who = f"{s.speaker}: " if s.speaker else ""
        out.append(f"{'>> ' if i in marked else '   '}{who}{s.text}")
    return "\n".join(out)


class LocalExtractor:
    """Rules + a small local model. `owners=True` resolves unassigned owners; `verify=True` relabels candidates."""

    method = "local"

    def __init__(
        self,
        choose: ChooseFn | None = None,
        model: str | None = None,
        *,
        owners: bool = True,
        verify: bool = False,
        owner_threshold: float = 0.8,
    ):
        self._choose = choose or LocalChooser(model)
        self.model_name = getattr(self._choose, "model_name", "custom")
        self.owners, self.verify, self.owner_threshold = owners, verify, owner_threshold

    def extract(
        self, transcript: Transcript, meeting_date: dt.date, people: list[str] | None = None
    ) -> tuple[list[ActionItem], list[Decision], Summary | None]:
        cands = RuleExtractor(strict=not self.verify).candidates(transcript, meeting_date, people)
        names = candidate_people(transcript, people)
        by_id = transcript.by_id()
        actions: list[ActionItem] = []
        decisions: list[Decision] = []
        for c in cands:
            rule_label = "action" if c.kind == "action" else "decision"
            if not self.verify and c.kind == "action" and c.confidence < 0.5:
                continue  # same filter as the strict rule extractor
            label = self._label(c, by_id, names) if self.verify else rule_label
            if label == "none":
                continue
            span = make_span(transcript, c.sentence_ids)
            if label == "decision":
                decisions.append(Decision(text=c.task, source=span, method="local" if self.verify else "rules"))
                continue
            owner, method = c.owner, ("local" if self.verify else "rules")
            if owner is None and self.owners and names:
                picked = self._owner(c, by_id, names)
                if picked:
                    owner, method = picked, "local"
            actions.append(
                ActionItem(
                    task=c.task, owner=owner, deadline_text=c.deadline_text, deadline_date=c.deadline_date,
                    source=span, confidence=c.confidence, method=method,
                )
            )  # fmt: skip
        return actions, decisions, None

    def _label(self, c: Candidate, by_id: dict, names: list[str]) -> str:
        ids = [i for i in sorted(set(c.context) | set(c.sentence_ids)) if i in by_id]
        user = f"People: {', '.join(names) or 'unknown'}\n{_render(by_id, ids, c.sentence_ids)}"
        messages = [{"role": "system", "content": LABEL_SYSTEM}]
        for q, a in LABEL_SHOTS:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": user})
        try:
            probs = self._choose(messages, ["action", "decision", "none"])
        except Exception:  # model missing or failed: keep the rule label
            return "none" if c.kind == "action" and c.confidence < 0.5 else c.kind
        return max(probs, key=probs.get)

    def _owner(self, c: Candidate, by_id: dict, names: list[str]) -> str | None:
        window = _window(by_id, c.sentence_ids)
        user = f"People: {', '.join(names)}\n{_render(by_id, window, c.sentence_ids)}"
        messages = [{"role": "system", "content": OWNER_SYSTEM}]
        for q, a in OWNER_SHOTS:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": user})
        try:
            probs = self._choose(messages, [*names, "unknown"])
        except Exception:
            return None
        best = max(probs, key=probs.get)
        if best == "unknown" or probs[best] < self.owner_threshold:
            return None
        near = " ".join(by_id[i].text for i in window)
        speakers = {by_id[i].speaker for i in window}
        if not re.search(rf"\b{re.escape(best)}\b", near) and best not in speakers:
            return None  # a name the dialogue never mentions is not an owner
        return best
