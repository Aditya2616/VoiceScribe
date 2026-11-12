"""Local small-model extractor: a verifier and normaliser over rule-based candidates.

Small instruct models (0.5B to 1.5B) are unreliable as free generators over a whole transcript, so the
rules propose candidate sentences (including weak ones) and the model labels each one with a tiny JSON
object: {"label": action|decision|none, "task": "...", "owner": "<name from the list>|null"}.
The owner must come from a closed list of names found in the meeting; anything that fails to parse
falls back to the rule-based answer for that candidate. Deadlines always come from the date resolver.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from collections.abc import Callable

from ..audio import default_threads
from ..schema import ActionItem, Decision, Summary, Transcript, make_span
from .rules import NAME, NOT_NAMES, Candidate, RuleExtractor, clean_task

GenerateFn = Callable[[list[dict[str, str]], str], str]

SYSTEM = (
    "You label one sentence from a meeting transcript. Reply with a single JSON object and nothing else: "
    '{"label": "action" | "decision" | "none", "task": "<short imperative phrase>", "owner": "<name>" | null}. '
    "action = someone commits to or is asked to do a concrete task. decision = the group decides or agrees on "
    "something. none = discussion, status, opinion or a vague wish. The owner must be one of the listed people, "
    "otherwise null. For a decision, task is the decision in a few words."
)
PREFILL = '{"label": "'
FEW_SHOT = [
    (
        "People: Ana, Ben\n>> Ben, can you send the report by Friday?",
        '{"label": "action", "task": "send the report", "owner": "Ben"}',
    ),
    (
        "People: Ana, Ben\n>> We decided to keep the old server for a month.",
        '{"label": "decision", "task": "keep the old server for a month", "owner": null}',
    ),
    (
        "People: Ana, Ben\n>> The meeting went well and people were happy.",
        '{"label": "none", "task": "", "owner": null}',
    ),
]


class LocalGenerator:
    """transformers causal LM, greedy decoding, CPU, thread-capped."""

    def __init__(self, model: str | None = None, threads: int | None = None, max_new_tokens: int = 48):
        self.model_name = model or os.environ.get("VOICESCRIBE_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        self.threads = threads or default_threads()
        self.max_new_tokens = max_new_tokens
        self._tok = None
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch.set_num_threads(self.threads)
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=torch.float32)
            self._model.eval()
        return self._tok, self._model

    def __call__(self, messages: list[dict[str, str]], prefill: str) -> str:
        import torch

        tok, model = self._load()
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) + prefill
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
        return prefill + tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True)


def parse_label(raw: str, people: list[str]) -> dict | None:
    """Leniently parse the model reply. Returns {"label", "task", "owner"} or None."""
    text = raw.strip()
    m = re.search(r"\{.*?\}", text, re.S)
    data = None
    if m:
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            data = None
    if data is None:
        label = re.search(r'"label"\s*:\s*"(\w+)', text)
        if not label:
            return None
        task = re.search(r'"task"\s*:\s*"([^"]*)', text)
        owner = re.search(r'"owner"\s*:\s*"([^"]*)', text)
        data = {"label": label.group(1), "task": task.group(1) if task else "", "owner": owner.group(1) if owner else None}
    label = str(data.get("label", "")).lower().strip()
    if label not in {"action", "decision", "none"}:
        return None
    owner = data.get("owner")
    owner = owner.strip() if isinstance(owner, str) else None
    by_lower = {p.lower(): p for p in people}
    owner = by_lower.get((owner or "").lower())
    task = data.get("task") if isinstance(data.get("task"), str) else ""
    return {"label": label, "task": task.strip(), "owner": owner}


def candidate_people(transcript: Transcript, cands: list[Candidate], people: list[str] | None) -> list[str]:
    names = {p for p in (people or [])}
    names |= {s.speaker for s in transcript.sentences if s.speaker and re.fullmatch(NAME, s.speaker)}
    names |= {c.owner for c in cands if c.owner}
    for s in transcript.sentences:
        m = re.match(rf"^(?:\w+,?\s+)?({NAME}),", s.text)
        if m and m.group(1) not in NOT_NAMES:
            names.add(m.group(1))
    return sorted(names)


class LocalExtractor:
    method = "local"

    def __init__(self, generate: GenerateFn | None = None, model: str | None = None):
        self._generate = generate or LocalGenerator(model)
        self.model_name = getattr(self._generate, "model_name", "custom")

    def extract(
        self, transcript: Transcript, meeting_date: dt.date, people: list[str] | None = None
    ) -> tuple[list[ActionItem], list[Decision], Summary | None]:
        cands = RuleExtractor(strict=False).candidates(transcript, meeting_date, people)
        names = candidate_people(transcript, cands, people)
        by_id = transcript.by_id()
        actions: list[ActionItem] = []
        decisions: list[Decision] = []
        for c in cands:
            verdict = self._ask(c, by_id, names, meeting_date)
            label = verdict["label"] if verdict else ("action" if c.kind == "action" and c.confidence >= 0.5 else
                                                      "decision" if c.kind == "decision" else "none")  # fmt: skip
            if label == "none":
                continue
            span = make_span(transcript, c.sentence_ids)
            task = c.task
            if verdict and verdict["task"] and 1 <= len(verdict["task"].split()) <= 16 and _overlaps(verdict["task"], span.quote):
                task = clean_task(verdict["task"])
            if label == "decision":
                decisions.append(Decision(text=task if verdict else c.task, source=span, method="local" if verdict else "rules"))
            else:
                owner = (verdict["owner"] if verdict else None) or c.owner
                actions.append(
                    ActionItem(
                        task=task, owner=owner, deadline_text=c.deadline_text, deadline_date=c.deadline_date,
                        source=span, confidence=0.7 if verdict else c.confidence, method="local" if verdict else "rules",
                    )
                )  # fmt: skip
        return actions, decisions, None

    def _ask(self, c: Candidate, by_id: dict, names: list[str], meeting_date: dt.date) -> dict | None:
        lines = "\n".join(f"{'>> ' if i in c.sentence_ids else '   '}{by_id[i].text}" for i in _window(c, by_id))
        user = (
            f"People: {', '.join(names) if names else 'unknown'}\n"
            f"Transcript (the sentence(s) marked >> are the ones to label):\n{lines}"
        )
        try:
            messages = [{"role": "system", "content": SYSTEM}]
            for q, a in FEW_SHOT:
                messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            messages.append({"role": "user", "content": user})
            raw = self._generate(messages, PREFILL)
        except Exception:  # model missing or generation failure: caller falls back to the rule answer
            return None
        verdict = parse_label(raw, names)
        if verdict and verdict["owner"]:
            # an owner the model names must actually be mentioned near the sentence (no invented owners)
            near = " ".join(by_id[i].text for i in _window(c, by_id))
            if not re.search(rf"\b{re.escape(verdict['owner'])}\b", near):
                verdict["owner"] = None
        return verdict


def _window(c: Candidate, by_id: dict) -> list[int]:
    ids = sorted(set(c.context) | set(c.sentence_ids))
    return [i for i in ids if i in by_id]


def _overlaps(task: str, quote: str) -> bool:
    """A rewritten task must reuse words from the quoted sentence (guards against invented tasks)."""
    a = {w for w in re.findall(r"[a-z]{3,}", task.lower())}
    b = {w for w in re.findall(r"[a-z]{3,}", quote.lower())}
    return bool(a) and len(a & b) / len(a) >= 0.5
