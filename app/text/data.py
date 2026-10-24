"""Loader for the Vietnamese normalization tables in ``data/``.

The tables are plain TSV, copied from NVIDIA NeMo-text-processing's Vietnamese
TN grammar (Apache-2.0 — see ``NOTICE``). They are data, not code: a unit, a
currency or a month name can be added without touching Python.

Why copy the tables instead of depending on ``nemo-text-processing``:

* The grammar's *code* is 1368 lines, of which only 291 are pynini/FST. The
  rest reads these TSVs into dicts — which is what this module does.
* ``pynini`` is a **147 MB** wheel with **manylinux-only** builds, so the
  dependency cannot be installed on Windows at all.
* The tables themselves are **4 KB**.

NeMo uses finite-state transducers because it must emit *every* valid reading
of an ambiguous input (audio-based normalization). A TTS frontend needs exactly
one reading, so the FST machinery buys nothing here.

Format: two columns (``key<TAB>value``) or three (``key<TAB>std<TAB>alt``).
Blank lines and ``#`` comments are skipped.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

#: Directory holding the TSV tables.
DATA_DIR = Path(__file__).parent / "data"

def _read_rows(relative: str) -> list[list[str]]:
    """Read one TSV into rows of columns, skipping blanks and comments."""
    path = DATA_DIR / relative
    if not path.is_file():
        return []
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        columns = [c.strip() for c in line.split("\t")]
        if len(columns) >= 2 and columns[0]:
            rows.append(columns)
    return rows

@lru_cache(maxsize=None)
def load_pairs(relative: str) -> dict[str, str]:
    """Read a two-column TSV into a ``{key: value}`` dict.

    Later rows win, which matches how NeMo's ``pynini.string_map`` resolves a
    duplicate key.
    """
    return {row[0]: row[1] for row in _read_rows(relative)}

@lru_cache(maxsize=None)
def load_triples(relative: str) -> dict[str, tuple[str, str]]:
    """Read a three-column TSV into ``{key: (standard, alternative)}``.

    Used by ``numbers/digit_special.tsv``, where ``1`` is ``một`` alone but
    ``mốt`` after ``mươi``.
    """
    return {row[0]: (row[1], row[2]) for row in _read_rows(relative) if len(row) >= 3}

@lru_cache(maxsize=None)
def load_keys(relative: str) -> tuple[str, ...]:
    """Read only the first column, preserving file order."""
    return tuple(row[0] for row in _read_rows(relative))

# --------------------------------------------------------------------------- #
# named accessors — one per table, so callers do not repeat the path
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def digits() -> dict[str, str]:
    """``1`` -> ``một`` .. ``9`` -> ``chín``."""
    return load_pairs("numbers/digit.tsv")

@lru_cache(maxsize=1)
def zero() -> str:
    """The word for ``0``."""
    return load_pairs("numbers/zero.tsv").get("0", "không")

@lru_cache(maxsize=1)
def teens() -> dict[str, str]:
    """``10`` -> ``mười`` .. ``19`` -> ``mười chín``."""
    return load_pairs("numbers/teen.tsv")

@lru_cache(maxsize=1)
def ties() -> dict[str, str]:
    """``2`` -> ``hai mươi`` .. ``9`` -> ``chín mươi``."""
    return load_pairs("numbers/ties.tsv")

@lru_cache(maxsize=1)
def magnitudes() -> dict[str, str]:
    """``thousand`` -> ``nghìn``, ``hundred`` -> ``trăm``, ``linh`` -> ``linh``."""
    return load_pairs("numbers/magnitudes.tsv")

@lru_cache(maxsize=1)
def special_digits() -> dict[str, tuple[str, str]]:
    """``1`` -> ``("một", "mốt")``, ``4`` -> ``("bốn", "tư")``, ``5`` -> ``("năm", "lăm")``."""
    return load_triples("numbers/digit_special.tsv")

@lru_cache(maxsize=1)
def quantity_abbreviations() -> dict[str, str]:
    """``k`` -> ``nghìn``, ``tr`` -> ``triệu``, ``t`` -> ``tỷ``."""
    return load_pairs("numbers/quantity_abbr.tsv")

@lru_cache(maxsize=1)
def months() -> dict[str, str]:
    """``1`` -> ``một`` .. ``12`` -> ``mười hai``. Note ``4`` is ``tư``."""
    return load_pairs("date/months.tsv")

@lru_cache(maxsize=1)
def days() -> dict[str, str]:
    """``1`` -> ``một`` .. ``31`` -> ``ba mươi mốt``. Note ``4`` is ``bốn``."""
    return load_pairs("date/days.tsv")

@lru_cache(maxsize=1)
def year_suffixes() -> dict[str, str]:
    """``tcn`` -> ``trước công nguyên``, ``scn`` -> ``sau công nguyên``."""
    return load_pairs("date/year_suffix.tsv")

@lru_cache(maxsize=1)
def measurements() -> dict[str, str]:
    """Unit symbol -> spoken form. ``km`` is not here; ``m``, ``g``, ``°C`` are."""
    merged = load_pairs("measure/base_units.tsv")
    merged.update(load_pairs("measure/measurements_minimal.tsv"))
    return merged

@lru_cache(maxsize=1)
def measure_prefixes() -> dict[str, str]:
    """SI prefixes: ``k`` -> ``ki lô``, ``M`` -> ``mê ga``."""
    return load_pairs("measure/prefixes.tsv")

@lru_cache(maxsize=1)
def currencies() -> dict[str, str]:
    """Currency symbol or code -> spoken form."""
    return load_pairs("money/currency.tsv")

@lru_cache(maxsize=1)
def per_unit_bases() -> dict[str, str]:
    """Denominator units for a rate: ``g`` -> ``gam``, ``m`` -> ``mét``."""
    return load_pairs("money/per_unit_bases.tsv")

@lru_cache(maxsize=1)
def per_unit_non_metric() -> dict[str, str]:
    """``/giờ`` -> ``trên giờ``, ``/ngày`` -> ``trên ngày``."""
    return load_pairs("money/per_unit_non_metric.tsv")

@lru_cache(maxsize=1)
def per_unit_prefixes() -> dict[str, str]:
    """SI prefixes allowed in a rate denominator."""
    return load_pairs("money/per_unit_prefixes.tsv")

@lru_cache(maxsize=1)
def ordinal_exceptions() -> dict[str, str]:
    """``1`` -> ``nhất``, ``4`` -> ``tư``."""
    return load_pairs("ordinal/ordinal_exceptions.tsv")

@lru_cache(maxsize=1)
def roman_numerals() -> dict[str, str]:
    """``I`` -> ``1``, ``IV`` -> ``4``, ``M`` -> ``1000``."""
    return load_pairs("roman/roman_numerals.tsv")

@lru_cache(maxsize=1)
def roman_key_words() -> tuple[str, ...]:
    """Words that introduce a roman numeral: ``thế kỉ``, ``chương``, ``thứ``."""
    return load_keys("roman/key_word.tsv")

@lru_cache(maxsize=1)
def time_units() -> dict[str, str]:
    """``h`` -> ``giờ``, ``p`` -> ``phút``, ``s`` -> ``giây``."""
    return load_pairs("time/time_units.tsv")

@lru_cache(maxsize=1)
def fraction_exceptions() -> dict[str, str]:
    """``4`` -> ``tư``, for ``một phần tư``."""
    return load_pairs("fraction/denominator_exceptions.tsv")
