"""Vietnamese dates, times and numeric ranges.

Month and day words come from ``data/date/*.tsv`` (see ``data/NOTICE``), so a
different reading for month 4 (``tư``) is a data edit. The *rules* are here:

* Each pattern is anchored so it cannot swallow a neighbouring number.
* The replacement re-adds the words it consumed (``ngày``, ``tháng``, ``năm``)
  so a caller that already wrote ``ngày 20/10`` does not end up with
  ``ngày ngày``.
* A range (``2024-2025``) is matched before the bare integer pass, or the
  hyphen survives and the two years are read as separate numbers.
"""

from __future__ import annotations

import re

from . import data
from .numbers import int_to_words, ordinal_to_words

#: ``ngày 20/10/2026``, ``20-10-2026``. The optional leading ``ngày`` is
#: consumed and re-emitted once.
_DATE_FULL = re.compile(
    r"(?:\bngày\s+)?\b([0-3]?\d)[/-]([01]?\d)[/-](\d{4})\b",
    re.IGNORECASE,
)
#: ``20/10`` — day and month, no year.
_DATE_SHORT = re.compile(r"(?:\bngày\s+)?\b([0-3]?\d)[/-]([01]?\d)\b(?![/-]\d)")
#: ``15:30``, ``15h30``, ``15g30``.
_TIME = re.compile(r"\b([0-2]?\d)\s*[:hg]\s*([0-5]\d)\b", re.IGNORECASE)
#: ``15 giờ`` — an hour with no minutes.
_HOUR_ONLY = re.compile(r"\b([0-2]?\d)\s*(?:giờ|h)\b(?!\s*\d)", re.IGNORECASE)
#: ``2024-2025``, ``1873-1907`` — a year range.
_YEAR_RANGE = re.compile(r"\b(1\d{3}|20\d{2})\s*[-–—]\s*(1\d{3}|20\d{2})\b")  # noqa: RUF001
#: ``3-5``, ``10 - 20`` — a small numeric range.
_NUMBER_RANGE = re.compile(r"\b(\d{1,3})\s*[-–—]\s*(\d{1,3})\b")  # noqa: RUF001
#: ``thứ 2`` .. ``thứ 7``.
_WEEKDAY = re.compile(r"\bthứ\s+([1-7])\b", re.IGNORECASE)

def _day_word(day: int) -> str:
    """``1`` -> ``mùng một``; other days use the table."""
    word = data.days().get(str(day), int_to_words(day))
    return f"mùng {word}" if 1 <= day <= 9 else word

def _month_word(month: int) -> str:
    """Month 4 is ``tư``, not ``bốn`` — see ``date/months.tsv``."""
    return data.months().get(str(month), int_to_words(month))

def _full_date(m: re.Match[str]) -> str:
    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return m.group(0)
    return (
        f"ngày {_day_word(day)} tháng {_month_word(month)} "
        f"năm {int_to_words(year)}"
    )

def _short_date(m: re.Match[str]) -> str:
    day, month = int(m.group(1)), int(m.group(2))
    if not (1 <= day <= 31 and 1 <= month <= 12):
        return m.group(0)
    return f"ngày {_day_word(day)} tháng {_month_word(month)}"

def _time(m: re.Match[str]) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23:
        return m.group(0)
    if minute == 0:
        return f"{int_to_words(hour)} giờ"
    return f"{int_to_words(hour)} giờ {int_to_words(minute)} phút"

def _hour_only(m: re.Match[str]) -> str:
    hour = int(m.group(1))
    return f"{int_to_words(hour)} giờ" if hour <= 23 else m.group(0)

def _year_range(m: re.Match[str]) -> str:
    return f"{int_to_words(int(m.group(1)))} đến {int_to_words(int(m.group(2)))}"

def _number_range(m: re.Match[str]) -> str:
    return f"{int_to_words(int(m.group(1)))} đến {int_to_words(int(m.group(2)))}"

def _weekday(m: re.Match[str]) -> str:
    return f"thứ {ordinal_to_words(int(m.group(1)))}"

def convert_dates(text: str) -> str:
    """Expand dates, times, weekdays and ranges into words.

    Runs before :func:`~app.text.numbers.convert_numbers` so ``2024-2025``
    becomes a range rather than two numbers joined by a hyphen.
    """
    if not text:
        return ""
    text = _DATE_FULL.sub(_full_date, text)
    text = _DATE_SHORT.sub(_short_date, text)
    text = _TIME.sub(_time, text)
    text = _HOUR_ONLY.sub(_hour_only, text)
    text = _YEAR_RANGE.sub(_year_range, text)
    text = _NUMBER_RANGE.sub(_number_range, text)
    return _WEEKDAY.sub(_weekday, text)
