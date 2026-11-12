import json

import httpx
import pytest
from conftest import MEETING_DATE, transcript_from_lines

from voicescribe.extract.anthropic_adapter import TOOL, AnthropicExtractor
from voicescribe.extract.base import get_extractor
from voicescribe.extract.hosted import to_results
from voicescribe.extract.local import LocalExtractor, parse_label
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


# ---- local verifier -----------------------------------------------------------------------------


def test_parse_label_is_lenient_and_enforces_closed_owner_list():
    people = ["Ben", "Maya"]
    assert parse_label('{"label": "action", "task": "send it", "owner": "ben"}', people) == {
        "label": "action", "task": "send it", "owner": "Ben",
    }  # fmt: skip
    assert parse_label('{"label": "action", "task": "x", "owner": "Zed"}', people)["owner"] is None
    assert parse_label('{"label": "decision", "task": "go", "owner": null', people)["label"] == "decision"  # truncated
    assert parse_label('{"label": "maybe"}', people) is None
    assert parse_label("no json here", people) is None


def test_local_extractor_uses_model_labels_and_falls_back_on_garbage():
    calls = []

    def fake(messages, prefill):
        calls.append(messages[-1]["content"])
        user = messages[-1]["content"]
        if "free plan" in user:
            return prefill + 'decision", "task": "keep the free plan", "owner": null}'
        if "invite flow" in user:
            return prefill + 'action", "task": "ship invite flow", "owner": "Rohan"}'
        return "garbage"

    actions, decisions, summary = LocalExtractor(generate=fake).extract(transcript(), MEETING_DATE)
    assert summary is None
    assert [a.owner for a in actions] == ["Rohan"] and actions[0].method == "local"
    assert actions[0].deadline_date.isoformat() == "2026-11-20"  # always from the date resolver
    assert [d.text for d in decisions] == ["Keep the free plan"] or decisions[0].text.lower().startswith("keep the free plan")
    assert all("People:" in c for c in calls)


def test_local_extractor_drops_candidates_the_model_calls_none_and_keeps_rule_answer_on_error():
    def none_model(messages, prefill):
        return prefill + 'none", "task": "", "owner": null}'

    actions, decisions, _ = LocalExtractor(generate=none_model).extract(transcript(), MEETING_DATE)
    assert actions == [] and decisions == []

    def broken(messages, prefill):
        raise RuntimeError("model missing")

    actions, decisions, _ = LocalExtractor(generate=broken).extract(transcript(), MEETING_DATE)
    assert len(actions) == 1 and actions[0].method == "rules" and len(decisions) == 1


def test_local_extractor_ignores_invented_owner_and_task():
    def liar(messages, prefill):
        return prefill + 'action", "task": "launch the rocket to mars", "owner": "Zed"}'

    actions, _, _ = LocalExtractor(generate=liar).extract(transcript(), MEETING_DATE)
    assert all(a.owner != "Zed" for a in actions)
    assert all("rocket" not in a.task for a in actions)


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

    ex = OpenAIExtractor(client=make_client(handler), api_key="sk-test", model="m")
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
    ex = OpenAIExtractor(client=make_client(lambda r: response), api_key="k")
    actions, decisions, summary = ex.extract(transcript(), MEETING_DATE)
    assert actions and actions[0].method == "rules" and summary is None


def test_openai_adapter_without_key_or_with_fallback_off_raises():
    ex = OpenAIExtractor(client=make_client(lambda r: httpx.Response(200)), api_key="", fallback=False)
    with pytest.raises(AdapterError):
        ex.extract(transcript(), MEETING_DATE)


def test_openai_adapter_network_error_falls_back():
    def handler(request):
        raise httpx.ConnectError("offline")

    actions, _, _ = OpenAIExtractor(client=make_client(handler), api_key="k").extract(transcript(), MEETING_DATE)
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

    ex = AnthropicExtractor(client=make_client(handler), api_key="ak-test", model="m")
    actions, decisions, summary = ex.extract(transcript(), MEETING_DATE)
    assert seen["headers"]["x-api-key"] == "ak-test" and seen["headers"]["anthropic-version"]
    assert seen["body"]["tool_choice"] == {"type": "tool", "name": TOOL}
    assert actions[0].method == "anthropic" and decisions and summary


def test_anthropic_adapter_fallback_when_no_tool_block():
    ex = AnthropicExtractor(client=make_client(lambda r: httpx.Response(200, json={"content": []})), api_key="k")
    actions, _, _ = ex.extract(transcript(), MEETING_DATE)
    assert actions[0].method == "rules"


def test_registry(monkeypatch):
    assert get_extractor("rules").method == "rules"
    monkeypatch.setenv("VOICESCRIBE_EXTRACTOR", "openai")
    assert get_extractor().method == "openai"
    with pytest.raises(ValueError):
        get_extractor("nope")
