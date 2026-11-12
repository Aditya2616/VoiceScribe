import datetime as dt

import pytest

from voicescribe.dates import find_deadline

TODAY = dt.date(2026, 11, 12)  # Thursday

CASES = [
    ("by next Friday", "2026-11-20"),
    ("by Friday", "2026-11-13"),
    ("this Friday", "2026-11-13"),
    ("on Thursday", "2026-11-19"),  # same weekday means next week
    ("this Thursday", "2026-11-12"),
    ("by Monday", "2026-11-16"),
    ("next Monday", "2026-11-16"),
    ("next Wednesday", "2026-11-18"),
    ("tomorrow", "2026-11-13"),
    ("the day after tomorrow", "2026-11-14"),
    ("today", "2026-11-12"),
    ("by the end of the day", "2026-11-12"),
    ("before the 25th", "2026-11-25"),
    ("by the 5th", "2026-12-05"),  # the 5th already passed this month
    ("on the 12th", "2026-11-12"),
    ("end of the month", "2026-11-30"),
    ("next month", "2026-12-31"),
    ("in two weeks", "2026-11-26"),
    ("in 3 days", "2026-11-15"),
    ("in a few days", "2026-11-15"),
    ("in a couple of weeks", "2026-11-26"),
    ("in three months", "2027-02-12"),
    ("next week", "2026-11-20"),
    ("end of the week", "2026-11-13"),
    ("by March 3rd", "2027-03-03"),
    ("on 20th of December", "2026-12-20"),
    ("in January", "2027-01-31"),
    ("end of the quarter", "2026-12-31"),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_resolves(text, expected):
    found = find_deadline(f"We will ship it {text}.", TODAY)
    assert found is not None, text
    assert found[1].isoformat() == expected


def test_phrase_keeps_the_leading_preposition():
    phrase, _ = find_deadline("Send it by next Friday please", TODAY)
    assert phrase == "by next Friday"


def test_earliest_phrase_wins():
    phrase, date = find_deadline("Do it tomorrow, or at the latest on Friday", TODAY)
    assert phrase == "tomorrow"
    assert date == dt.date(2026, 11, 13)


@pytest.mark.parametrize("text", ["we may need more time", "that sat badly with me", "I will mark this", "no dates here"])
def test_no_false_positives(text):
    assert find_deadline(text, TODAY) is None


def test_month_end_and_leap_year():
    assert find_deadline("end of the month", dt.date(2028, 2, 10))[1] == dt.date(2028, 2, 29)
    assert find_deadline("in a month", dt.date(2026, 1, 31))[1] == dt.date(2026, 2, 28)
    assert find_deadline("on February 29", dt.date(2026, 11, 12)) is None  # 2027 has no 29 Feb


def test_year_rollover():
    assert find_deadline("next Friday", dt.date(2026, 12, 30))[1] == dt.date(2027, 1, 8)
    assert find_deadline("in two weeks", dt.date(2026, 12, 25))[1] == dt.date(2027, 1, 8)
