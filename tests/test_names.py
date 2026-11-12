from pathlib import Path

from conftest import MEETING_DATE, ROOT, transcript_from_lines, words_from_text

from voicescribe.eval import extract_eval
from voicescribe.extract.base import RulesExtractor
from voicescribe.names import apply_map, apply_to_owner, canonical_map, edit_distance, parse_people, similar
from voicescribe.pipeline import build_meeting

VARIANT_MEETING = [
    (None, "Great. Rohan, can you ship the shorter invite flow by next Friday?"),
    (None, "Yes, I will have it ready."),
    (None, "Rohan, please write up the sync bug by Wednesday."),
    (None, "Rohen, thanks for that."),
    (None, "Rohen will own the sync bug."),
    (None, "Lena, write a short announcement and send it to the beta users."),
    (None, "Thanks, Lina."),
]


def test_similar_merges_vowel_slips_but_not_different_names():
    assert similar("Rohan", "Rohen") and similar("Lena", "Lina") and similar("Tomas", "Tomos")
    assert not similar("Ben", "Ken") and not similar("Ana", "Anna")  # too short to judge
    assert not similar("Priya", "Maria") and not similar("Hugo", "Hana")
    assert edit_distance("kitten", "sitting") == 3


def test_most_frequent_spelling_wins():
    t = transcript_from_lines(VARIANT_MEETING)
    m = canonical_map(t.sentences)
    assert m == {"Rohen": "Rohan", "Lina": "Lena"} or m == {"Rohen": "Rohan", "Lena": "Lina"}
    assert m["Rohen"] == "Rohan"  # Rohan x2 as an addressed name, Rohen x2: tie broken by evidence/order, not by chance


def test_participants_are_a_closed_set():
    t = transcript_from_lines(VARIANT_MEETING)
    m = canonical_map(t.sentences, ["Rohan", "Lena", "Maya"])
    assert m == {"Rohen": "Rohan", "Lina": "Lena"}
    assert apply_to_owner("Rohan", ["Rohan", "Lena"]) == "Rohan"
    assert apply_to_owner("Dmitri", ["Rohan", "Lena"]) is None
    assert apply_to_owner("Dmitri", None) == "Dmitri"


def test_apply_map_rewrites_word_text_only_on_word_boundaries():
    ws = words_from_text("Rohen, thanks. Rohenson stays.")
    out = apply_map(ws, {"Rohen": "Rohan"})
    assert [w.text for w in out] == ["Rohan,", "thanks.", "Rohenson", "stays."]


def test_build_meeting_uses_one_spelling_for_owners_and_transcript():
    text = (
        "Okay. Rohan, can you ship the invite flow by Friday? Sure, I will. "
        "Later. Rohen, please write up the sync bug by Wednesday. Understood, I will do that. "
        "Rohan will own the cleanup. Thanks, Rohan."
    )
    m = build_meeting(words_from_text(text), 30.0, MEETING_DATE, RulesExtractor())
    owners = {a.owner for a in m.action_items}
    assert owners == {"Rohan"}
    assert "Rohen" not in m.transcript.text() and m.name_map == {"Rohen": "Rohan"}


def test_participants_drop_owners_outside_the_list():
    text = "Okay. Dmitri, can you book the room by Friday? Sure, I will."
    m = build_meeting(words_from_text(text), 15.0, MEETING_DATE, RulesExtractor(), people=["Ana", "Ben"])
    assert [a.owner for a in m.action_items] == [None]


def test_parse_people():
    assert parse_people("ana, BEN; Chloe\nana") == ["Ana", "Ben", "Chloe"]
    assert parse_people("") is None and parse_people("a1, x") is None


def test_labelled_meetings_are_left_untouched():
    for m in extract_eval.load_all(Path(ROOT / "data" / "eval" / "extract")):
        assert canonical_map(m.transcript.sentences) == {}, m.name


def test_meeting_records_the_participants_it_was_given():
    text = "Okay. Lena, can you ship the invite flow by Friday? Sure, I will. Thanks, Lina."
    m = build_meeting(words_from_text(text), 20.0, MEETING_DATE, RulesExtractor(), people=["Maya", "Lena"])
    assert m.participants == ["Maya", "Lena"] and "Lina" not in m.transcript.text() and m.name_map == {"Lina": "Lena"}
    assert build_meeting(words_from_text(text), 20.0, MEETING_DATE, RulesExtractor()).participants == []
