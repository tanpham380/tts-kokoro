"""Vietnamese cardinal, ordinal and decimal numbers.

The rules encoded here are the ones a Vietnamese reader applies without
thinking, and the ones the previous single-function implementation got wrong:

* ``.`` groups thousands and ``,`` marks the decimal point. ``1.234.567`` is
  one million two hundred thirty-four thousand five hundred sixty-seven, not
  a decimal. A dot followed by exactly three digits is a group separator.
* A non-leading group below one hundred keeps its filler words, so
  ``1000001`` is ``một triệu không trăm lẻ một`` and not ``một triệu một``.
* ``mốt``/``tư``/``lăm`` replace ``một``/``bốn``/``năm`` after ``mươi``.
* A leading-zero run is read digit by digit (phone numbers, codes).

Nothing here imports anything outside the standard library.
"""

from __future__ import annotations

import re

#: Digits 0-9 as words.
UNITS: tuple[str, ...] = (
    "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín",
)

#: Scale words for each group of three digits, least significant first.
_SCALES: tuple[str, ...] = ("", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ")

#: Ordinals that are irregular; everything else uses the cardinal.
_ORDINALS: dict[int, str] = {1: "nhất", 2: "hai", 4: "tư"}

#: ``1.234`` / ``1.234.567`` — a dot followed by exactly three digits.
_THOUSANDS_DOT = re.compile(r"(?<![\d.,])\d{1,3}(?:\.\d{3})+(?![\d.,])")
#: ``1,234,567`` — the English grouping, seen in imported text.
_THOUSANDS_COMMA = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d.,])")
#: ``3,14`` — a comma is always the Vietnamese decimal point.
_DECIMAL_COMMA = re.compile(r"(?<![\d.,])(\d+),(\d+)(?![\d.,])")
#: ``3.14`` — a dot that is *not* a group separator.
_DECIMAL_DOT = re.compile(r"(?<![\d.,])(\d+)\.(\d+)(?![\d.,])")
#: A bare integer.
_INTEGER = re.compile(r"(?<![\d.,])\d+(?![\d.,])")

def _read_tens(n: int) -> str:
    """Read 10..99. ``n`` must already be in range."""
    if n < 10:
        return UNITS[n]
    if n == 10:
        return "mười"
    if n < 20:
        return "mười lăm" if n == 15 else "mười " + UNITS[n % 10]

    tens, unit = divmod(n, 10)
    head = UNITS[tens] + " mươi"
    if unit == 0:
        return head
    if unit == 1:
        return head + " mốt"
    if unit == 4:
        return head + " tư"
    if unit == 5:
        return head + " lăm"
    return head + " " + UNITS[unit]

def _read_group(n: int, *, full: bool) -> str:
    """Read 0..999.

    ``full`` forces the hundreds and the ``lẻ`` filler. It is set for every
    group that is not the leading one, which is what keeps
    ``1000001`` -> ``một triệu không trăm lẻ một``.
    """
    hundreds, rest = divmod(n, 100)
    parts: list[str] = []
    if hundreds or full:
        parts.append(UNITS[hundreds] + " trăm")
    if rest == 0:
        return " ".join(parts)
    if rest < 10:
        if hundreds or full:
            parts.append("lẻ")
        parts.append(UNITS[rest])
        return " ".join(parts)
    parts.append(_read_tens(rest))
    return " ".join(parts)

def int_to_words(n: int) -> str:
    """Read an integer as natural Vietnamese words.

    Handles negatives and anything up to ``triệu tỷ`` (10^18). Groups of three
    digits are read most-significant first; a zero group is skipped but forces
    the filler words on the next non-zero group.
    """
    if n == 0:
        return UNITS[0]
    if n < 0:
        return "âm " + int_to_words(-n)

    groups: list[int] = []
    while n:
        n, group = divmod(n, 1000)
        groups.append(group)

    parts: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if group == 0:
            continue
        scale = _SCALES[index] if index < len(_SCALES) else ""
        # Only the leading group may drop its hundreds and the "lẻ" filler.
        full = index != len(groups) - 1
        words = _read_group(group, full=full)
        parts.append(f"{words} {scale}".strip())
    return " ".join(parts)

def digits_to_words(s: str) -> str:
    """Read a digit string one digit at a time (``"01"`` -> ``"không một"``)."""
    return " ".join(UNITS[int(c)] for c in s if c.isdigit())

def ordinal_to_words(n: int) -> str:
    """Read an ordinal (``4`` -> ``tư``, for ``thứ tư``)."""
    return _ORDINALS.get(n, int_to_words(n))

def _read_decimal(whole: str, fraction: str) -> str:
    """Read ``whole.fraction`` as ``... phẩy ...``.

    One or two fraction digits without a leading zero are read as a number
    (``3.14`` -> ``ba phẩy mười bốn``); anything else is read digit by digit
    (``0.05`` -> ``không phẩy không năm``, ``3.14159`` -> ``ba phẩy một bốn...``).
    """
    head = int_to_words(int(whole))
    if len(fraction) <= 2 and not fraction.startswith("0"):
        tail = int_to_words(int(fraction))
    else:
        tail = digits_to_words(fraction)
    return f"{head} phẩy {tail}"

def _strip_grouping(s: str) -> str:
    return s.replace(".", "").replace(",", "")

def read_number_token(token: str) -> str:
    """Read one numeric token that may contain grouping or a decimal point.

    This is the single place the ``.``-is-thousands / ``,``-is-decimal rule is
    applied, so callers that pull a number out of a larger pattern (``50.000đ``,
    ``VND 1.500.000``) get the same answer as :func:`convert_numbers`.

    ``50.000`` -> ``năm mươi nghìn``; ``3,14`` -> ``ba phẩy mười bốn``;
    ``3.14`` -> ``ba phẩy mười bốn``; ``0901234567`` -> digit by digit.
    """
    token = token.strip()
    if not token:
        return ""

    # A dot followed by exactly three digits, repeated, is grouping.
    if _THOUSANDS_DOT.fullmatch(token) or _THOUSANDS_COMMA.fullmatch(token):
        return int_to_words(int(_strip_grouping(token)))

    # A comma is always the decimal point in Vietnamese.
    if "," in token:
        whole, _, fraction = token.partition(",")
        return _read_decimal(whole, fraction)

    # A dot is a decimal point only when it is not a group separator.
    if "." in token:
        whole, _, fraction = token.partition(".")
        return _read_decimal(whole, fraction)

    if len(token) > 1 and token.startswith("0"):
        return digits_to_words(token)
    return int_to_words(int(token))

def convert_numbers(text: str) -> str:
    """Expand every numeric expression in ``text`` into words.

    Order matters and is fixed: grouped thousands, then decimals, then bare
    integers. Running the integer pass first would shred ``1.234`` into two
    numbers.
    """
    if not text:
        return ""

    def _thousands(m: re.Match[str]) -> str:
        return int_to_words(int(_strip_grouping(m.group(0))))

    def _decimal(m: re.Match[str]) -> str:
        return _read_decimal(m.group(1), m.group(2))

    def _integer(m: re.Match[str]) -> str:
        s = m.group(0)
        # A leading zero means a code or a phone number, not a quantity.
        if len(s) > 1 and s.startswith("0"):
            return digits_to_words(s)
        return int_to_words(int(s))

    text = _THOUSANDS_DOT.sub(_thousands, text)
    text = _THOUSANDS_COMMA.sub(_thousands, text)
    text = _DECIMAL_COMMA.sub(_decimal, text)
    text = _DECIMAL_DOT.sub(_decimal, text)
    return _INTEGER.sub(_integer, text)
