"""Resolve relative deadline phrases against the meeting date.

Conventions (deterministic and documented):

* A named period resolves to its *end*, because it is a deadline: "next week" is that week's
  Friday; "next month" and "end of the month" are the last day of the month.
* "Friday" / "by Friday" is the next Friday strictly after the meeting date; "this Friday" may
  be the meeting day itself.
* "next Friday" is the Friday of the following Monday-to-Sunday week.
* A bare day of the month ("the 20th") is its next occurrence on or after the meeting date.
* "March 20" with no year is the next such date on or after the meeting date.
"""

from __future__ import annotations

import calendar
import datetime as dt
import re

_WD_MAP = {
    "mon": 0, "monday": 0, "tue": 1, "tues": 1, "tuesday": 1, "wed": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3, "fri": 4, "friday": 4, "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}  # fmt: skip
_MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9,
    "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}  # fmt: skip
_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "couple of": 2, "a couple of": 2, "few": 3, "a few": 3,
}  # fmt: skip

_WD = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tues|tue|wed|thurs|thur|thu|fri)"
_MON = (
    r"(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)"
)
_ORD = r"(\d{1,2})(?:st|nd|rd|th)?"
_NUM = r"(\d+|a few|a couple of|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple of|few)"


def _num(word: str) -> int:
    w = word.lower()
    return int(w) if w.isdigit() else _NUMBER_WORDS[w]


def _end_of_month(y: int, m: int) -> dt.date:
    return dt.date(y, m, calendar.monthrange(y, m)[1])


def _add_months(y: int, m: int, n: int) -> tuple[int, int]:
    idx = y * 12 + (m - 1) + n
    return idx // 12, idx % 12 + 1


def _next_weekday(today: dt.date, wd: int, allow_today: bool = False) -> dt.date:
    delta = (wd - today.weekday()) % 7
    if delta == 0 and not allow_today:
        delta = 7
    return today + dt.timedelta(days=delta)


def _next_dom(today: dt.date, day: int) -> dt.date | None:
    y, m = today.year, today.month
    for _ in range(14):
        if day <= calendar.monthrange(y, m)[1]:
            cand = dt.date(y, m, day)
            if cand >= today:
                return cand
        y, m = _add_months(y, m, 1)
    return None


def _explicit(today: dt.date, month: int, day: int) -> dt.date | None:
    for year in (today.year, today.year + 1):
        try:
            cand = dt.date(year, month, day)
        except ValueError:
            return None
        if cand >= today:
            return cand
    return None


def _in_n(today: dt.date, n: int, unit: str) -> dt.date:
    if unit == "day":
        return today + dt.timedelta(days=n)
    if unit == "week":
        return today + dt.timedelta(weeks=n)
    y, m = _add_months(today.year, today.month, n)
    return dt.date(y, m, min(today.day, calendar.monthrange(y, m)[1]))


def _rules(today: dt.date):
    monday = today - dt.timedelta(days=today.weekday())
    fri_this_week = monday + dt.timedelta(days=4)
    q_end_month = ((today.month - 1) // 3 + 1) * 3
    in_month_year = lambda mon: (today.year if mon >= today.month else today.year + 1, mon)  # noqa: E731
    return [
        (r"\bthe day after tomorrow\b", lambda m: today + dt.timedelta(days=2)),
        (r"\btomorrow\b", lambda m: today + dt.timedelta(days=1)),
        (
            r"\b(?:today|tonight|this morning|this afternoon|this evening|end of (?:the )?day|eod|close of business)\b",
            lambda m: today,
        ),
        (rf"\bnext {_WD}\b", lambda m: monday + dt.timedelta(days=7 + _WD_MAP[m.group(1)])),
        (rf"\bthis {_WD}\b", lambda m: _next_weekday(today, _WD_MAP[m.group(1)], allow_today=True)),
        (rf"\b{_WD}\b", lambda m: _next_weekday(today, _WD_MAP[m.group(1)])),
        (r"\b(?:end of (?:the |this )?week|eow)\b", lambda m: fri_this_week),
        (r"\bnext week\b", lambda m: fri_this_week + dt.timedelta(days=7)),
        (r"\bthis week\b", lambda m: fri_this_week),
        (r"\b(?:end of (?:the |this )?month|eom|this month)\b", lambda m: _end_of_month(today.year, today.month)),
        (r"\bnext month\b", lambda m: _end_of_month(*_add_months(today.year, today.month, 1))),
        (r"\bend of (?:the |this )?quarter\b", lambda m: _end_of_month(today.year, q_end_month)),
        (rf"\bin {_NUM} (day|week|month)s?\b", lambda m: _in_n(today, _num(m.group(1)), m.group(2))),
        (
            rf"\b{_NUM} (day|week|month)s? from (?:now|today)\b",
            lambda m: _in_n(today, _num(m.group(1)), m.group(2)),
        ),
        (rf"\b{_MON}\.? {_ORD}\b", lambda m: _explicit(today, _MONTH_MAP[m.group(1)], int(m.group(2)))),
        (rf"\b{_ORD} of {_MON}\b", lambda m: _explicit(today, _MONTH_MAP[m.group(2)], int(m.group(1)))),
        (rf"\bthe {_ORD}\b(?! of)", lambda m: _next_dom(today, int(m.group(1)))),
        (rf"\bin {_MON}\b", lambda m: _end_of_month(*in_month_year(_MONTH_MAP[m.group(1)]))),
    ]


def find_deadline(text: str, today: dt.date) -> tuple[str, dt.date] | None:
    """Find the earliest deadline phrase in text: (phrase as written, resolved date)."""
    low = text.lower()
    best: tuple[int, int, dt.date] | None = None
    for pattern, resolve in _rules(today):
        for m in re.finditer(pattern, low):
            try:
                d = resolve(m)
            except (ValueError, KeyError):
                d = None
            if d is None:
                continue
            if best is None or m.start() < best[0] or (m.start() == best[0] and m.end() > best[1]):
                best = (m.start(), m.end(), d)
            break
    if best is None:
        return None
    start, end, d = best
    lead = re.search(r"\b(by|before|until|on|due)\s+(?:the\s+)?$", low[:start])
    if lead:
        start = lead.start()
    return text[start:end].strip(" ,.;"), d
