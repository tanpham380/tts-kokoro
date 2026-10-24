"""Vietnamese text normalization for TTS.

``normalize(text, language)`` is the only entry point the engine needs. The
pipeline is ordered, and the order is load-bearing:

1. Unicode NFC — so ``ế`` is one code point, not ``ê`` + a combining accent.
2. Units — currency, measurements, percentages. Before dates so ``15km/h`` is
   not read as a time.
3. Dates — dates, times, weekdays, ranges. Before numbers so ``2024-2025`` is
   a range and not two numbers joined by a hyphen.
4. Acronym + digits — before numbers, so ``COVID-19`` still has digits for the
   pattern to match.
5. Numbers — grouped thousands, decimals, bare integers.
6. Standalone letters — after numbers, so ``B`` in ``nhóm máu A và B`` is
   still a letter.

Reordering any of these silently breaks cases that currently pass. See
``.agents/rules/text-normalization.md`` for the measured defect table.

There is deliberately **no abbreviation dictionary**. Vietnamese abbreviations
are an open set, so a hand-maintained table is always incomplete and always
needs a rebuild to extend. Only closed sets are handled, and by rule: the Latin
alphabet (``letters.py``) and the number/date/unit grammar.

The word tables in ``data/`` are copied from NVIDIA NeMo-text-processing
(Apache-2.0 — see ``data/NOTICE``). Only the **data** was copied, not the
grammar's Python: NeMo's code is 1368 lines of which 291 are pynini/FST, and
the rest reads these same TSVs into dicts, which ``data.py`` does in ~200 lines
of standard-library Python. NeMo needs finite-state transducers to emit *every*
valid reading of an ambiguous input; a TTS frontend needs exactly one, so the
FST machinery and its 147 MB ``pynini`` dependency buy nothing here.
"""

from __future__ import annotations

import re
import unicodedata

from . import data
from .dates import convert_dates
from .letters import (
    LETTER_NAMES,
    fix_standalone_letters,
    split_acronym_numbers,
)
from .numbers import (
    UNITS,
    convert_numbers,
    digits_to_words,
    int_to_words,
    ordinal_to_words,
    read_number_token,
)
from .units import convert_units

__all__ = [
    "LETTER_NAMES",
    "UNITS",
    "convert_dates",
    "convert_numbers",
    "convert_units",
    "data",
    "digits_to_words",
    "fix_standalone_letters",
    "int_to_words",
    "normalize",
    "ordinal_to_words",
    "read_number_token",
    "split_acronym_numbers",
]

#: Languages the Vietnamese pipeline applies to.
_VIETNAMESE = frozenset({"vi", "vn", "vietnamese", "auto", ""})

def normalize(text: str, language: str | None = "vi") -> str:
    """Normalize ``text`` into the spoken form the G2P expects.

    ``language`` other than Vietnamese only gets NFC normalization and
    whitespace collapsing, so English input is not rewritten with Vietnamese
    number words.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    if (language or "").strip().lower() not in _VIETNAMESE:
        return re.sub(r"[ \t]+", " ", text).strip()

    text = convert_units(text)
    text = convert_dates(text)
    # Before numbers: ``COVID-19`` must become ``COVID 19`` while the digits are
    # still digits, or the pattern no longer matches.
    text = split_acronym_numbers(text)
    text = convert_numbers(text)
    text = fix_standalone_letters(text)
    return re.sub(r"[ \t]+", " ", text).strip()
