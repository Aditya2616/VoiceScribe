from voicescribe.eval.diar_metrics import der, summarise
from voicescribe.extract.rules import RuleExtractor, infer_speaker_names
from voicescribe.schema import Sentence
from voicescribe.synth import VOICE_POOL, assign_voices

REF = [(0.0, 2.0, "A"), (2.5, 5.0, "B"), (5.5, 8.0, "A")]


def test_perfect_hypothesis_under_any_label_names_has_zero_error():
    hyp = [(0.0, 2.0, "x"), (2.5, 5.0, "y"), (5.5, 8.0, "x")]
    r = der(REF, hyp)
    assert r["der"] == 0 and r["frame_accuracy"] == 1 and r["predicted_speakers"] == 2


def test_confusion_miss_and_false_alarm_are_counted():
    merged = [(0.0, 8.0, "x")]  # one speaker for everything: B's speech is confused
    r = der(REF, merged, collar=0.0)
    assert 0.3 < r["confusion"] < 0.4 and r["miss"] == 0 and r["predicted_speakers"] == 1
    r = der(REF, [(0.0, 2.0, "x")], collar=0.0)  # only the first turn found
    assert r["miss"] > 0.6 and r["confusion"] == 0
    r = der([(0.0, 1.0, "A")], [(0.0, 1.0, "x"), (3.0, 4.0, "x")], collar=0.0)
    assert r["false_alarm"] > 0.9


def test_collar_forgives_boundary_jitter():
    hyp = [(0.0, 2.1, "x"), (2.4, 5.0, "y"), (5.5, 8.0, "x")]
    assert der(REF, hyp, collar=0.25)["der"] == 0 < der(REF, hyp, collar=0.0)["der"]


def test_summary_weights_by_speech_and_counts_speakers():
    a = {"der": 0.0, "frame_accuracy": 1.0, "miss": 0, "false_alarm": 0, "confusion": 0, "speech": 10, "true_speakers": 3, "predicted_speakers": 3}
    b = {"der": 0.3, "frame_accuracy": 0.7, "miss": 0, "false_alarm": 0, "confusion": 0.3, "speech": 30, "true_speakers": 4, "predicted_speakers": 3}
    s = summarise([a, b])
    assert abs(s["der"] - 0.225) < 1e-9 and s["speaker_count_exact"] == 0.5 and s["speaker_count_mean_abs_error"] == 0.5


def test_voices_are_distinct_within_a_meeting():
    for offset in range(len(VOICE_POOL)):
        v = assign_voices(["a", "b", "c", "d", "e"], offset)
        assert len(set(v.values())) == 5


def sents(rows):
    return [Sentence(id=i, start=i, end=i + 1, text=t, speaker=s) for i, (s, t) in enumerate(rows)]


def test_names_are_bound_from_requests_thanks_and_introductions():
    ss = sents([
        ("S1", "Hi, I'm Maya."), ("S2", "Rohan, can you ship the invite flow by Friday?"), ("S3", "Yes, I'll have it ready."),
        ("S3", "Thanks, Rohan."), ("S1", "Great."),
    ])  # fmt: skip
    # S2 asks (so S2 is not Rohan), S3 accepts: Rohan; "Thanks, Rohan." said by S3 names the previous speaker S3 again
    names = infer_speaker_names(ss)
    assert names.get("S1") == "Maya" and names.get("S3") == "Rohan" and "S2" not in names


def test_conflicting_evidence_leaves_a_speaker_unnamed():
    ss = sents([
        ("S1", "Rohan, can you ship the flow?"), ("S2", "Sure, I'll do it."),
        ("S1", "Theo, can you send the copy?"), ("S2", "Yes, I will."),
        ("S1", "Theo, please check the vendor."), ("S2", "Okay, I'll email them."),
    ])  # fmt: skip
    # S2 is "Rohan" once and "Theo" twice: 2/3 >= 0.6 -> Theo; with a stricter bar it is left unnamed
    assert infer_speaker_names(ss)["S2"] == "Theo"
    assert "S2" not in infer_speaker_names(ss, min_confidence=0.8)


def test_a_name_goes_to_one_speaker_only():
    ss = sents([
        ("S1", "Rohan, can you ship the flow?"), ("S2", "Sure, I'll do it."),
        ("S1", "Rohan, please check the vendor."), ("S3", "Okay, I'll email them."),
    ])  # fmt: skip
    assert infer_speaker_names(ss) == {}  # S2 and S3 each claim Rohan once: a tie names neither


def test_first_person_commitment_takes_the_bound_speaker_name():
    ss = sents([
        ("S1", "Hi, this is Maya."), ("S1", "I'll book the launch retro for next week."), ("S2", "Good idea."),
    ])  # fmt: skip
    from conftest import MEETING_DATE

    from voicescribe.schema import Transcript

    actions, _ = RuleExtractor().extract(Transcript(duration=3, sentences=ss), MEETING_DATE)
    assert [a.owner for a in actions] == ["Maya"]


def test_build_meeting_names_bound_speakers_and_resolves_first_person_owners():
    from conftest import MEETING_DATE, words_from_text

    from voicescribe.extract.base import RulesExtractor
    from voicescribe.pipeline import build_meeting

    sentences = [
        ("Speaker 1", "Hi everyone, I'm Maya and this is the sync."),
        ("Speaker 1", "Rohan, can you ship the invite flow by Friday?"),
        ("Speaker 2", "Yes, I'll have it ready for review."),
        ("Speaker 1", "I'll book the launch retro for next week."),
        ("Speaker 2", "Thanks, Maya."),
    ]
    words = words_from_text(" ".join(t for _, t in sentences), per_word=0.4)
    turns, i = [], 0
    for spk, t in sentences:
        n = len(t.split())
        turns.append((words[i].start, words[i + n - 1].end, spk))
        i += n
    m = build_meeting(words, 30.0, MEETING_DATE, RulesExtractor(), turns=turns, people=["Maya", "Rohan"])
    assert m.speaker_names == {"Speaker 1": "Maya", "Speaker 2": "Rohan"}
    assert {a.task: a.owner for a in m.action_items} == {"Ship the invite flow": "Rohan", "Book the launch retro": "Maya"}
    assert "Speaker" not in m.transcript.text() and {s.speaker for s in m.transcript.sentences} == {"Maya", "Rohan"}
