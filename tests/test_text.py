"""Golden-file tests for Vietnamese text normalization.

Every case here is a real defect that was measured against the previous
hand-rolled implementation, or a case that must not regress. The expected
strings are the *spoken* form, so a change that alters pronunciation fails
loudly instead of silently.

Run with ``python -m pytest tests/test_text.py -q``.
"""

from __future__ import annotations

import pytest

from app.text import (
    convert_dates,
    convert_numbers,
    convert_units,
    digits_to_words,
    int_to_words,
    normalize,
    ordinal_to_words,
    read_number_token,
)

# --------------------------------------------------------------------------- #
# cardinal numbers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "không"),
        (5, "năm"),
        (10, "mười"),
        (11, "mười một"),
        (15, "mười lăm"),
        (20, "hai mươi"),
        (21, "hai mươi mốt"),
        (24, "hai mươi tư"),
        (25, "hai mươi lăm"),
        (100, "một trăm"),
        (101, "một trăm lẻ một"),
        (105, "một trăm lẻ năm"),
        (110, "một trăm mười"),
        (115, "một trăm mười lăm"),
        (1000, "một nghìn"),
        (1001, "một nghìn không trăm lẻ một"),
        (1010, "một nghìn không trăm mười"),
        (1100, "một nghìn một trăm"),
        (100_000, "một trăm nghìn"),
        (1_001_000, "một triệu không trăm lẻ một nghìn"),
        (1_000_001, "một triệu không trăm lẻ một"),
        (1_000_000_000, "một tỷ"),
        (1_000_000_001, "một tỷ không trăm lẻ một"),
        (
            1_234_567_890,
            "một tỷ hai trăm ba mươi tư triệu năm trăm sáu mươi bảy "
            "nghìn tám trăm chín mươi",
        ),
    ],
)
def test_int_to_words(value: int, expected: str) -> None:
    assert int_to_words(value) == expected

def test_int_to_words_negative() -> None:
    assert int_to_words(-5) == "âm năm"

def test_digits_to_words() -> None:
    assert digits_to_words("01") == "không một"
    assert digits_to_words("0901234567") == (
        "không chín không một hai ba bốn năm sáu bảy"
    )

def test_ordinal_to_words() -> None:
    assert ordinal_to_words(4) == "tư"
    assert ordinal_to_words(2) == "hai"
    assert ordinal_to_words(7) == "bảy"

# --------------------------------------------------------------------------- #
# the dot-is-thousands rule — the worst defect in the old implementation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("1.234", "một nghìn hai trăm ba mươi tư"),
        ("1.234.567", "một triệu hai trăm ba mươi tư nghìn năm trăm sáu mươi bảy"),
        ("50.000", "năm mươi nghìn"),
        ("1,234,567", "một triệu hai trăm ba mươi tư nghìn năm trăm sáu mươi bảy"),
        ("3.14", "ba phẩy mười bốn"),
        ("3,14", "ba phẩy mười bốn"),
        ("0.05", "không phẩy không năm"),
        ("6.5", "sáu phẩy năm"),
        ("0901234567", "không chín không một hai ba bốn năm sáu bảy"),
    ],
)
def test_read_number_token(token: str, expected: str) -> None:
    assert read_number_token(token) == expected

def test_grouped_thousands_is_not_a_decimal() -> None:
    """The regression that motivated the rewrite."""
    assert convert_numbers("1.234.567") == (
        "một triệu hai trăm ba mươi tư nghìn năm trăm sáu mươi bảy"
    )
    assert "phẩy" not in convert_numbers("1.234.567")

def test_decimal_still_reads_as_decimal() -> None:
    assert convert_numbers("3.14") == "ba phẩy mười bốn"
    assert convert_numbers("0.05") == "không phẩy không năm"

def test_leading_zero_is_digit_by_digit() -> None:
    assert convert_numbers("01") == "không một"
    assert convert_numbers("0901234567") == (
        "không chín không một hai ba bốn năm sáu bảy"
    )

# --------------------------------------------------------------------------- #
# dates, times, ranges
# --------------------------------------------------------------------------- #

def test_full_date() -> None:
    assert convert_dates("01/05/2001") == (
        "ngày mùng một tháng năm năm hai nghìn không trăm lẻ một"
    )

def test_date_does_not_duplicate_ngay() -> None:
    assert convert_dates("ngày 20/10/2026") == (
        "ngày hai mươi tháng mười năm hai nghìn không trăm hai mươi sáu"
    )

def test_invalid_date_is_left_alone() -> None:
    assert convert_dates("25/13/2020") == "25/13/2020"

def test_time() -> None:
    assert convert_dates("15:30") == "mười lăm giờ ba mươi phút"
    assert convert_dates("12h30") == "mười hai giờ ba mươi phút"
    assert convert_dates("7g05") == "bảy giờ năm phút"

def test_year_range() -> None:
    assert convert_dates("2024-2025") == (
        "hai nghìn không trăm hai mươi tư đến "
        "hai nghìn không trăm hai mươi lăm"
    )

def test_weekday() -> None:
    assert convert_dates("thứ 4") == "thứ tư"
    assert convert_dates("thứ 2") == "thứ hai"

# --------------------------------------------------------------------------- #
# currency, units, percentages
# --------------------------------------------------------------------------- #

def test_vnd_suffix() -> None:
    assert convert_units("50.000đ") == "năm mươi nghìn đồng"

def test_vnd_prefix_and_trailing() -> None:
    assert convert_units("VND 100") == "một trăm đồng"
    assert convert_units("1.500.000 VND") == "một triệu năm trăm nghìn đồng"

def test_usd() -> None:
    assert convert_units("USD 100") == "một trăm đô la"
    assert convert_units("100 USD") == "một trăm đô la"

def test_percent_range_beats_bare_percent() -> None:
    """The range must be matched first or the leading number is orphaned."""
    assert convert_units("3-5%") == "ba đến năm phần trăm"
    assert convert_units("100%") == "một trăm phần trăm"

def test_units() -> None:
    assert convert_units("15km") == "mười lăm ki lô mét"
    assert convert_units("10km/h") == "mười ki lô mét một giờ"
    assert convert_units("3kg") == "ba ki lô gam"
    assert convert_units("5cm") == "năm xen ti mét"

def test_k_suffix() -> None:
    assert convert_units("100k") == "một trăm nghìn"

# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.234.567", "một triệu hai trăm ba mươi tư nghìn năm trăm sáu mươi bảy"),
        ("50.000đ", "năm mươi nghìn đồng"),
        ("Giá 1.500.000 VND", "Giá một triệu năm trăm nghìn đồng"),
        ("15:30", "mười lăm giờ ba mươi phút"),
        ("2024-2025", "hai nghìn không trăm hai mươi tư đến hai nghìn không trăm hai mươi lăm"),
        ("3-5%", "ba đến năm phần trăm"),
        ("1000001", "một triệu không trăm lẻ một"),
        ("thứ 4", "thứ tư"),
        ("10km/h", "mười ki lô mét một giờ"),
        ("VND 100", "một trăm đồng"),
        ("01/05/2001", "ngày mùng một tháng năm năm hai nghìn không trăm lẻ một"),
        ("3.14", "ba phẩy mười bốn"),
        ("0.05", "không phẩy không năm"),
        ("6.5", "sáu phẩy năm"),
        ("100k", "một trăm nghìn"),
        ("năm 2024", "năm hai nghìn không trăm hai mươi tư"),
        ("tăng trưởng 100%", "tăng trưởng một trăm phần trăm"),
        ("khoảng cách 15km", "khoảng cách mười lăm ki lô mét"),
        ("Tôi có 2 con mèo và 3 con chó.", "Tôi có hai con mèo và ba con chó."),
    ],
)
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text) == expected

# --------------------------------------------------------------------------- #
# project-specific decisions that must survive a library swap
# --------------------------------------------------------------------------- #

def test_abbreviations_are_not_a_dictionary() -> None:
    """Vietnamese abbreviations are an open set, so there is no lookup table.

    ``TP.HCM`` is no longer expanded by this package. It is left for NeMo TN or
    the caller, which is the trade-off of dropping the hand-maintained table.
    """
    from app.text import LETTER_NAMES

    assert "TP.HCM" not in LETTER_NAMES
    assert normalize("TP.HCM") == "TP.HCM"

def test_unknown_acronym_is_left_alone() -> None:
    """A multi-letter run is not a standalone letter, so it is not spelled out."""
    assert normalize("ABC") == "ABC"

def test_standalone_letters() -> None:
    """Upstream issue #4: a lone Y must read /i/, not the letter name."""
    assert normalize("khoa Y") == "khoa i"
    assert normalize("bác sĩ Y") == "bác sĩ i"
    assert normalize("nhóm máu A và B") == "nhóm máu a và bê"
    assert normalize("vitamin C") == "vitamin xê"
    assert normalize("vitamin D") == "vitamin dê"

def test_letter_names_are_complete() -> None:
    """Every Latin letter has a name, so the rule never needs extending."""
    from app.text import LETTER_NAMES

    assert set(LETTER_NAMES) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert LETTER_NAMES["Y"] == "i"

def test_letters_inside_words_are_untouched() -> None:
    assert normalize("yêu quý ngày mai") == "yêu quý ngày mai"

def test_case_is_preserved() -> None:
    """Lowercasing the whole string destroys proper-noun cues."""
    assert normalize("Xin chào") == "Xin chào"

def test_acronym_with_number() -> None:
    assert normalize("COVID-19") == "COVID mười chín"

def test_nemo_stage_is_optional() -> None:
    """The suite must pass with or without pynini installed."""
    from app.text import nemo

    if nemo.is_available():
        assert nemo.unavailable_reason() is None
    else:
        assert nemo.unavailable_reason() is not None
        # The fallback must be a no-op, not an exception.
        assert nemo.normalize("1.234") == "1.234"

def test_non_vietnamese_language_skips_expansion() -> None:
    assert normalize("1.234.567", language="en") == "1.234.567"
    assert normalize("khoa Y", language="en") == "khoa Y"

def test_empty_input() -> None:
    assert normalize("") == ""
    assert normalize("   ") == ""
