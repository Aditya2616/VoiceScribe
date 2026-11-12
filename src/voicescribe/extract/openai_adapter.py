"""OpenAI chat-completions adapter (JSON-schema structured output) over httpx. Stub-tested only."""

from __future__ import annotations

import datetime as dt
import json
import os

import httpx

from ..schema import ActionItem, Decision, Summary, Transcript
from .hosted import SCHEMA, SYSTEM, render_transcript, to_results
from .rules import RuleExtractor


class AdapterError(RuntimeError):
    pass


class ConfigError(AdapterError):
    """A required setting (such as the model id) is missing."""


class OpenAIExtractor:
    method = "openai"

    def __init__(self, client: httpx.Client | None = None, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, fallback: bool = True):  # fmt: skip
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "")
        if not self.model:
            raise ConfigError("set OPENAI_MODEL (or pass model=) to the model id you want to use")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.client = client or httpx.Client(timeout=60)
        self.fallback = fallback
        self.last_method = "openai"

    def request_body(self, transcript: Transcript, meeting_date: dt.date, people: list[str] | None) -> dict:
        return {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": render_transcript(transcript, meeting_date, people)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "meeting_minutes", "strict": True, "schema": SCHEMA},
            },
        }

    def extract(
        self, transcript: Transcript, meeting_date: dt.date, people: list[str] | None = None
    ) -> tuple[list[ActionItem], list[Decision], Summary | None]:
        try:
            if not self.api_key:
                raise AdapterError("OPENAI_API_KEY is not set")
            resp = self.client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=self.request_body(transcript, meeting_date, people),
            )
            if resp.status_code != 200:
                raise AdapterError(f"OpenAI returned HTTP {resp.status_code}")
            content = resp.json()["choices"][0]["message"]["content"]
            self.last_method = "openai"
            return to_results(json.loads(content), transcript, meeting_date, "openai")
        except (AdapterError, httpx.HTTPError, KeyError, IndexError, ValueError, TypeError):
            if not self.fallback:
                raise
            self.last_method = "rules"  # visible to the pipeline: the hosted call failed
            actions, decisions = RuleExtractor().extract(transcript, meeting_date, people)
            return actions, decisions, None
