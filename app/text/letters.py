"""Standalone Latin letters and acronym-with-number rules.

Both are *rules*, not data tables, which is why they live here rather than in
an abbreviation dictionary:

* Every Latin letter has exactly one Vietnamese name, so the mapping is
  complete by construction and never needs extending.
* An uppercase run followed by a hyphen and digits (``COVID-19``) is an
  acronym with a suffix, not a range — a structural fact, not a lookup.

``Y`` is the exception that matters. It is a vowel in Vietnamese, so a lone
``Y`` reads /i/ rather than the letter name "quai" — the fix for issue #4 on
``iamdinhthuan/Kokoro-Vietnamese``. The rest use the alphabet name.

The letter pass must run **after** number expansion, so a ``B`` in
``nhóm máu A và B`` is still a letter and not a digit.
"""

from __future__ import annotations

import re

#: Vietnamese alphabet names. ``Y`` is the vowel /i/, not the letter name.
LETTER_NAMES: dict[str, str] = {
    "A": "a",
    "B": "bê",
    "C": "xê",
    "D": "dê",
    "E": "e",
    "F": "ép",
    "G": "gờ",
    "H": "hát",
    "I": "i",
    "J": "giây",
    "K": "ca",
    "L": "e-lờ",
    "M": "em",
    "N": "en",
    "O": "o",
    "P": "pê",
    "Q": "quy",
    "R": "e-rờ",
    "S": "ét",
    "T": "tê",
    "U": "u",
    "V": "vê",
    "W": "vê-kép",
    "X": "ích",
    "Y": "i",
    "Z": "dét",
}

#: One pattern per letter, upper and lower case, built once at import.
_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(rf"\b{letter}\b"), name) for letter, name in LETTER_NAMES.items()
) + tuple(
    (re.compile(rf"\b{letter.lower()}\b"), name)
    for letter, name in LETTER_NAMES.items()
)

#: ``COVID-19`` -> ``COVID 19``. An uppercase run followed by a hyphen and a
#: number is an acronym with a suffix, not a range, so the hyphen becomes a
#: space and the number is read normally.
_ACRONYM_HYPHEN = re.compile(r"\b([A-Z]{2,})\s*[-–—]\s*(\d+)\b")  # noqa: RUF001

def _acronym_hyphen(m: re.Match[str]) -> str:
    return f"{m.group(1)} {m.group(2)}"

def split_acronym_numbers(text: str) -> str:
    """Turn ``COVID-19`` into ``COVID 19`` so the number is read, not the hyphen."""
    if not text:
        return ""
    return _ACRONYM_HYPHEN.sub(_acronym_hyphen, text)

def fix_standalone_letters(text: str) -> str:
    """Read every lone Latin letter as its Vietnamese name.

    ``khoa Y`` -> ``khoa i``; ``vitamin C`` -> ``vitamin xê``;
    ``nhóm máu A và B`` -> ``nhóm máu a và bê``. A letter inside a word is
    untouched, so ``yêu quý`` stays ``yêu quý``.
    """
    if not text:
        return ""
    for pattern, name in _PATTERNS:
        text = pattern.sub(name, text)
    return text

