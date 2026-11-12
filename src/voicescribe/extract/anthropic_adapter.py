"""Anthropic Messages adapter (forced tool call returns the structured minutes) over httpx. Stub-tested only."""

from __future__ import annotations

import datetime as dt
import os

import httpx

from ..schema import ActionItem, Decision, Summary, Transcript
from .hosted import SCHEMA, SYSTEM, render_transcript, to_results
from .openai_adapter import AdapterError, ConfigError
from .rules import RuleExtractor

TOOL = "record_meeting_minutes"


class AnthropicExtractor:
    method = "anthropic"

    def __init__(self, client: httpx.Client | None = None, api_key: str | None = None, model: str | None = None,
                 base_url: str = "https://api.anthropic.com/v1", fallback: bool = True):  # fmt: skip
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "")
        if not self.model:
            raise ConfigError("set ANTHROPIC_MODEL (or pass model=) to the model id you want to use")
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.Client(timeout=60)
        self.fallback = fallback
        self.last_method = "anthropic"

    def request_body(self, transcript: Transcript, meeting_date: dt.date, people: list[str] | None) -> dict:
        return {
            "model": self.model,
            "max_tokens": 2000,
            "temperature": 0,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": render_transcript(transcript, meeting_date, people)}],
            "tools": [{"name": TOOL, "description": "Record the structured meeting minutes.", "input_schema": SCHEMA}],
            "tool_choice": {"type": "tool", "name": TOOL},
        }

    def extract(
        self, transcript: Transcript, meeting_date: dt.date, people: list[str] | None = None
    ) -> tuple[list[ActionItem], list[Decision], Summary | None]:
        try:
            if not self.api_key:
                raise AdapterError("ANTHROPIC_API_KEY is not set")
            resp = self.client.post(
                f"{self.base_url}/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                json=self.request_body(transcript, meeting_date, people),
            )
            if resp.status_code != 200:
                raise AdapterError(f"Anthropic returned HTTP {resp.status_code}")
            block = next(b for b in resp.json()["content"] if b.get("type") == "tool_use" and b.get("name") == TOOL)
            self.last_method = "anthropic"
            return to_results(block["input"], transcript, meeting_date, "anthropic")
        except (AdapterError, httpx.HTTPError, KeyError, StopIteration, ValueError, TypeError):
            if not self.fallback:
                raise
            self.last_method = "rules"  # visible to the pipeline: the hosted call failed
            actions, decisions = RuleExtractor().extract(transcript, meeting_date, people)
            return actions, decisions, None
