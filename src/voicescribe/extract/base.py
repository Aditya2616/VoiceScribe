"""Extractor interface and registry (chosen with VOICESCRIBE_EXTRACTOR)."""

from __future__ import annotations

import datetime as dt
import os
from typing import Protocol

from ..schema import ActionItem, Decision, Summary, Transcript


class Extractor(Protocol):
    method: str

    def extract(
        self, transcript: Transcript, meeting_date: dt.date, people: list[str] | None = None
    ) -> tuple[list[ActionItem], list[Decision], Summary | None]:
        """Return (action items, decisions, optional abstractive summary)."""
        ...


class RulesExtractor:
    """Adapter giving the rule extractor the common return shape."""

    method = "rules"

    def extract(self, transcript, meeting_date, people=None):
        from .rules import RuleExtractor

        actions, decisions = RuleExtractor().extract(transcript, meeting_date, people)
        return actions, decisions, None


def get_extractor(name: str | None = None) -> Extractor:
    name = (name or os.environ.get("VOICESCRIBE_EXTRACTOR") or "rules").lower()
    if name == "rules":
        return RulesExtractor()
    if name == "local":
        from .local import LocalExtractor

        return LocalExtractor()
    if name == "openai":
        from .openai_adapter import OpenAIExtractor

        return OpenAIExtractor()
    if name == "anthropic":
        from .anthropic_adapter import AnthropicExtractor

        return AnthropicExtractor()
    raise ValueError(f"unknown extractor {name!r}; choose rules, local, openai or anthropic")
