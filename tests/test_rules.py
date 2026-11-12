from conftest import MEETING_DATE, transcript_from_lines

from voicescribe.extract.rules import RuleExtractor, infer_speaker_names, is_ack, is_reply


def run(lines, **kw):
    return RuleExtractor().extract(transcript_from_lines(lines), MEETING_DATE, **kw)


def test_addressed_request_with_acceptance_gets_owner_and_deadline():
    actions, decisions = run(
        [("Maya", "Rohan, can you ship the invite flow by next Friday?"), ("Rohan", "Yes, I'll have it ready by then.")]
    )
    assert decisions == []
    (a,) = actions
    assert a.owner == "Rohan" and a.task == "Ship the invite flow"
    assert a.deadline_date.isoformat() == "2026-11-20" and a.deadline_text == "by next Friday"
    assert a.source.sentence_ids == [0, 1]  # the acceptance belongs to the request


def test_self_commitment_uses_the_speaker_when_known_and_is_unassigned_otherwise():
    (known,), _ = run([("Lena", "I'll update the onboarding screens tomorrow.")])
    assert known.owner == "Lena" and known.deadline_date.isoformat() == "2026-11-13"
    (unknown,), _ = run([(None, "I'll update the onboarding screens tomorrow.")])
    assert unknown.owner is None


def test_third_person_assignment_and_imperative_address():
    (a,), _ = run([(None, "Sofia will own the migration runbook, due next Friday.")])
    assert a.owner == "Sofia" and "runbook" in a.task
    (b,), _ = run([(None, "Okay, Jo, take a look at the query plans next week.")])
    assert b.owner == "Jo" and b.deadline_date.isoformat() == "2026-11-20"


def test_decisions_are_separated_from_actions():
    actions, decisions = run(
        [
            (None, "We decided to keep the free plan with a limit of three calendars."),
            (None, "Let's push the launch to December."),
            (None, "Yes, that is approved."),
        ]
    )
    assert actions == []
    assert len(decisions) == 2


def test_hard_negatives_are_not_action_items():
    lines = [
        (None, "I can see why people are annoyed."),
        (None, "We should probably look at pricing at some point."),
        (None, "I can't make it on Friday."),
        (None, "Should we ship it on Friday?"),
        (None, "The release went well."),
    ]
    actions, decisions = run(lines)
    assert actions == [] and decisions == []


def test_is_ack_and_is_reply():
    assert is_ack("Sure, I'll do that tomorrow.")
    assert is_ack("Will do.")
    assert is_ack("I can take that, yes.")
    assert not is_ack("I'll write the rollback plan by Friday.")  # a new commitment, not an acceptance
    assert not is_ack("Sure, do you want it in Slack or email?")
    assert is_reply("Rohan, can you ping legal?", "Sure, I'll ping them today.")
    assert not is_reply("The release went well.", "Sure, I'll ping them today.")


def test_infer_speaker_names_from_the_reply():
    t = transcript_from_lines(
        [("SPEAKER_0", "Rohan, can you ship it?"), ("SPEAKER_1", "Sure, I'll do that."), ("SPEAKER_0", "Thanks.")]
    )
    assert infer_speaker_names(t.sentences) == {"SPEAKER_1": "Rohan"}
    actions, _ = RuleExtractor().extract(
        transcript_from_lines(
            [("SPEAKER_0", "Rohan, can you ship it?"), ("SPEAKER_1", "Sure, I'll do that."),
             ("SPEAKER_1", "I'll update the docs by Friday.")]
        ),
        MEETING_DATE,
    )  # fmt: skip
    assert [a.owner for a in actions] == ["Rohan", "Rohan"]


def test_deadline_in_the_acceptance_is_used_when_the_request_has_none():
    (a,), _ = run([(None, "Chloe, can you check the lag before the cutover?"), ("Chloe", "Sure, I can do that tomorrow.")])
    assert a.deadline_date.isoformat() == "2026-11-13"
