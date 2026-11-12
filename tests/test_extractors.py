import json

import httpx
import pytest
from conftest import MEETING_DATE, transcript_from_lines

from voicescribe.extract.anthropic_adapter import TOOL, AnthropicExtractor
from voicescribe.extract.base import get_extractor
from voicescribe.extract.hosted import to_results
from voicescribe.extract.local import LocalExtractor
from voicescribe.extract.openai_adapter import AdapterError, OpenAIExtractor

LINES = [
    ("Maya", "Okay, let's start."),
    ("Maya", "Rohan, can you ship the invite flow by next Friday?"),
    ("Rohan", "Sure, I'll do that."),
    ("Maya", "We decided to keep the free plan."),
    ("Theo", "I can see why that matters."),
]


def transcript():
    return transcript_from_lines(LINES)


# ---- local model (closed-choice owner resolution and verification) ---------------------------------


def chooser(answers: dict[str, str], seen: list | None = None):
    """Fake model: answers[key] is chosen when key is in the final user message, else the last option."""

    def choose(messages, options):
        user = messages[-1]["content"]
        if seen is not None:
            seen.append((user, options))
        for key, ans in answers.items():
            if key in user and ans in options:
                return {o: (0.9 if o == ans else 0.1 / (len(options) - 1)) for o in options}
        return {o: (0.9 if o == options[-1] else 0.1 / (len(options) - 1)) for o in options}

    return choose


OWNER_LINES = [
    (None, "Someone needs to send out a survey."),
    (None, "I'll do it this week."),
    (None, "Is that fine, Maya?"),
    (None, "Rohan, can you ship the invite flow by next Friday?"),
    (None, "Sure, I'll do that."),
    (None, "Good."),
]


def test_owner_resolution_fills_only_unassigned_items_from_the_closed_list():
    seen: list = []
    t = transcript_from_lines(OWNER_LINES)
    ex = LocalExtractor(choose=chooser({"survey": "Maya"}, seen), owners=True)
    actions, _, _ = ex.extract(t, MEETING_DATE, ["Maya", "Rohan"])
    assert any(a.owner == "Maya" and a.method == "local" for a in actions)
    invite = next(a for a in actions if "invite" in a.task.lower())
    assert invite.owner == "Rohan" and invite.method == "rules"  # rule owners are never overridden
    options = seen[0][1]
    assert options[-1] == "unknown" and set(options[:-1]) <= {"Maya", "Rohan"}


def test_owner_resolution_respects_unknown_threshold_and_mention_guard():
    t = transcript_from_lines(OWNER_LINES)
    unknown = LocalExtractor(choose=chooser({}), owners=True).extract(t, MEETING_DATE, ["Maya", "Rohan"])[0]
    assert all(a.owner in (None, "Rohan") for a in unknown)

    def weak(messages, options):
        return {o: (0.5 if o == "Maya" else 0.5 / (len(options) - 1)) for o in options}

    low = LocalExtractor(choose=weak, owners=True, owner_threshold=0.6).extract(t, MEETING_DATE, ["Maya", "Rohan"])[0]
    assert all(a.owner != "Maya" for a in low)

    def invents(messages, options):
        return {o: 1.0 if o == "Rohan" else 0.0 for o in options}

    far = transcript_from_lines([(None, "Rohan, can you ship it?")] + [(None, f"Filler sentence number {i} here.") for i in range(9)]
                                + [(None, "I'll write the report tomorrow.")])
    acts = LocalExtractor(choose=invents, owners=True).extract(far, MEETING_DATE)[0]
    assert [a.owner for a in acts if "report" in a.task.lower()] == [None]  # Rohan is not mentioned near it


def test_verify_mode_uses_model_labels_and_survives_model_failure():
    def labeller(messages, options):
        marked = " ".join(ln for ln in messages[-1]["content"].splitlines() if ln.startswith(">>"))
        want = "decision" if "free plan" in marked else "action" if "invite flow" in marked else "none"
        return {o: 1.0 if o == want else 0.0 for o in options}

    actions, decisions, summary = LocalExtractor(choose=labeller, owners=False, verify=True).extract(transcript(), MEETING_DATE)
    assert summary is None and len(actions) == 1 and "invite" in actions[0].task.lower()
    assert len(decisions) == 1 and actions[0].deadline_date.isoformat() == "2026-11-20"  # dates from the resolver

    def broken(messages, options):
        raise RuntimeError("model missing")

    actions, decisions, _ = LocalExtractor(choose=broken, owners=True, verify=True).extract(transcript(), MEETING_DATE)
    assert len(actions) == 1 and actions[0].method == "local" and len(decisions) == 1  # rule labels kept



# ---- hosted adapters (stub transport, never the live API) -----------------------------------------

PAYLOAD = {
    "summary": "They agreed pricing and assigned the invite flow.",
    "decisions": [{"text": "Keep the free plan", "sentence_ids": [3]}, {"text": "bogus", "sentence_ids": [99]}],
    "action_items": [
        {"task": "Ship the invite flow", "owner": "Rohan", "deadline_text": "by next Friday", "sentence_ids": [1, 2]},
        {"task": "Phantom task", "owner": None, "deadline_text": None, "sentence_ids": [42]},
    ],
}


def test_to_results_drops_items_with_missing_sources_and_resolves_dates():
    actions, decisions, summary = to_results(PAYLOAD, transcript(), MEETING_DATE, "openai")
    assert [a.task for a in actions] == ["Ship the invite flow"]
    assert actions[0].deadline_date.isoformat() == "2026-11-20" and actions[0].source.sentence_ids == [1, 2]
    assert [d.text for d in decisions] == ["Keep the free plan"]
    assert summary.kind == "abstractive"


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_adapter_request_and_parse():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(PAYLOAD)}}]})

    ex = OpenAIExtractor(client=make_client(handler), api_key="sk-test", model="test-model")
    actions, decisions, summary = ex.extract(transcript(), MEETING_DATE)
    assert seen["url"].endswith("/chat/completions") and seen["auth"] == "Bearer sk-test"
    assert seen["body"]["response_format"]["json_schema"]["strict"] is True
    assert "[1] Maya: Rohan, can you ship" in seen["body"]["messages"][1]["content"]
    assert actions[0].method == "openai" and summary is not None


@pytest.mark.parametrize(
    "response",
    [httpx.Response(500, text="boom"), httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
     httpx.Response(200, json={"unexpected": True})],
)  # fmt: skip
def test_openai_adapter_falls_back_to_rules_on_failure(response):
    ex = OpenAIExtractor(client=make_client(lambda r: response), api_key="k", model="test-model")
    actions, decisions, summary = ex.extract(transcript(), MEETING_DATE)
    assert actions and actions[0].method == "rules" and summary is None


def test_openai_adapter_without_key_or_with_fallback_off_raises():
    ex = OpenAIExtractor(client=make_client(lambda r: httpx.Response(200)), api_key="", model="test-model", fallback=False)
    with pytest.raises(AdapterError):
        ex.extract(transcript(), MEETING_DATE)


def test_openai_adapter_network_error_falls_back():
    def handler(request):
        raise httpx.ConnectError("offline")

    actions, _, _ = OpenAIExtractor(client=make_client(handler), api_key="k", model="test-model").extract(transcript(), MEETING_DATE)
    assert actions[0].method == "rules"


def test_anthropic_adapter_forces_the_tool_and_parses_tool_use():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "name": TOOL, "input": PAYLOAD}]},
        )

    ex = AnthropicExtractor(client=make_client(handler), api_key="ak-test", model="test-model")
    actions, decisions, summary = ex.extract(transcript(), MEETING_DATE)
    assert seen["headers"]["x-api-key"] == "ak-test" and seen["headers"]["anthropic-version"]
    assert seen["body"]["tool_choice"] == {"type": "tool", "name": TOOL}
    assert actions[0].method == "anthropic" and decisions and summary


def test_anthropic_adapter_fallback_when_no_tool_block():
    ex = AnthropicExtractor(client=make_client(lambda r: httpx.Response(200, json={"content": []})), api_key="k", model="test-model")
    actions, _, _ = ex.extract(transcript(), MEETING_DATE)
    assert actions[0].method == "rules"


def test_registry(monkeypatch):
    assert get_extractor("rules").method == "rules"
    monkeypatch.setenv("VOICESCRIBE_EXTRACTOR", "openai")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    assert get_extractor().method == "openai"
    with pytest.raises(ValueError):
        get_extractor("nope")


@pytest.mark.parametrize("cls,var", [(OpenAIExtractor, "OPENAI_MODEL"), (AnthropicExtractor, "ANTHROPIC_MODEL")])
def test_hosted_adapters_have_no_default_model(monkeypatch, cls, var):
    from voicescribe.extract.openai_adapter import ConfigError

    monkeypatch.delenv(var, raising=False)
    with pytest.raises(ConfigError, match=var):
        cls(api_key="k")
    monkeypatch.setenv(var, "test-model")
    assert cls(api_key="k").model == "test-model"
