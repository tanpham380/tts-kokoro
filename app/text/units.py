"""Vietnamese currency, units and percentages.

Money, measurements and percentages are read as a number followed by a scale
word: ``đồng`` for VND, ``ki lô mét`` for ``km``, ``phần trăm`` for ``%``.

Two ordering rules are load-bearing:

* A range (``3-5%``) must be matched **before** the bare percentage, or the
  trailing ``5%`` wins and the leading ``3-`` is left behind.
* The number inside a unit is read by
  :func:`~app.text.numbers.read_number_token`, so ``50.000đ`` is fifty
  thousand đồng and not fifty point zero-zero-zero.
"""

from __future__ import annotations

import re

from .numbers import read_number_token

#: A number that may carry grouping dots or a decimal comma.
_NUM = r"\d+(?:[.,]\d+)*"

#: ``VND 100`` -> ``một trăm đồng``
_VND_PREFIX = re.compile(rf"\bVND\s+({_NUM})\b", re.IGNORECASE)
#: ``1.500.000 VND`` -> ``một triệu năm trăm nghìn đồng``
_VND_TRAILING = re.compile(rf"\b({_NUM})\s+VND\b", re.IGNORECASE)
#: ``USD 100`` -> ``một trăm đô la``
_USD_PREFIX = re.compile(rf"\bUSD\s+({_NUM})\b", re.IGNORECASE)
#: ``100 USD`` -> ``một trăm đô la``
_USD_TRAILING = re.compile(rf"\b({_NUM})\s+USD\b", re.IGNORECASE)
#: ``50.000đ`` -> ``năm mươi nghìn đồng``
_VND_SUFFIX = re.compile(rf"\b({_NUM})\s*đ\b", re.IGNORECASE)
#: ``100k`` -> ``một trăm nghìn``
_K_SUFFIX = re.compile(rf"\b({_NUM})k\b(?![\w])", re.IGNORECASE)
#: ``15-20%`` -> ``mười lăm đến hai mươi phần trăm``
_PERCENT_RANGE = re.compile(rf"\b({_NUM})\s*[-–—]\s*({_NUM})\s*%")  # noqa: RUF001
#: ``15%`` -> ``mười lăm phần trăm``
_PERCENT = re.compile(rf"\b({_NUM})\s*%")
#: ``120km/h`` -> ``một trăm hai mươi ki lô mét một giờ``
_KM_PER_HOUR = re.compile(rf"\b({_NUM})\s*km/h\b", re.IGNORECASE)
#: ``15km`` -> ``mười lăm ki lô mét``
_KM = re.compile(rf"\b({_NUM})\s*km\b(?![\s./])", re.IGNORECASE)
#: ``15kg`` -> ``mười lăm ki lô gam``
_KG = re.compile(rf"\b({_NUM})\s*kg\b(?![\s./])", re.IGNORECASE)
#: ``15cm`` -> ``mười lăm xen ti mét``
_CM = re.compile(rf"\b({_NUM})\s*cm\b(?![\s./])", re.IGNORECASE)
#: ``15mm`` -> ``mười lăm mi li mét``
_MM = re.compile(rf"\b({_NUM})\s*mm\b(?![\s./])", re.IGNORECASE)
#: ``15m2`` -> ``mười lăm mét hai``
_M2 = re.compile(rf"\b({_NUM})\s*m2\b(?![\s./])", re.IGNORECASE)
#: ``15m3`` -> ``mười lăm mét khối``
_M3 = re.compile(rf"\b({_NUM})\s*m3\b(?![\s./])", re.IGNORECASE)

def _with_suffix(suffix: str):
    """Build a replacement that reads the captured number and appends ``suffix``."""

    def _replace(m: re.Match[str]) -> str:
        return f"{read_number_token(m.group(1))} {suffix}"

    return _replace

def _vnd_prefix(m: re.Match[str]) -> str:
    return f"{read_number_token(m.group(1))} đồng"

def _usd_prefix(m: re.Match[str]) -> str:
    return f"{read_number_token(m.group(1))} đô la"

def _vnd_trailing(m: re.Match[str]) -> str:
    return f"{read_number_token(m.group(1))} đồng"

def _usd_trailing(m: re.Match[str]) -> str:
    return f"{read_number_token(m.group(1))} đô la"

def _percent_range(m: re.Match[str]) -> str:
    return (
        f"{read_number_token(m.group(1))} đến "
        f"{read_number_token(m.group(2))} phần trăm"
    )

def convert_units(text: str) -> str:
    """Expand currency, measurements and percentages into words.

    Runs before :func:`~app.text.dates.convert_dates` so ``15km/h`` is not
    mistaken for a time and ``3-5%`` is not mistaken for a date range.
    """
    if not text:
        return ""
    text = _VND_PREFIX.sub(_vnd_prefix, text)
    text = _USD_PREFIX.sub(_usd_prefix, text)
    text = _VND_TRAILING.sub(_vnd_trailing, text)
    text = _USD_TRAILING.sub(_usd_trailing, text)
    text = _VND_SUFFIX.sub(_with_suffix("đồng"), text)
    text = _K_SUFFIX.sub(_with_suffix("nghìn"), text)
    text = _PERCENT_RANGE.sub(_percent_range, text)
    text = _PERCENT.sub(_with_suffix("phần trăm"), text)
    text = _KM_PER_HOUR.sub(_with_suffix("ki lô mét một giờ"), text)
    text = _KM.sub(_with_suffix("ki lô mét"), text)
    text = _KG.sub(_with_suffix("ki lô gam"), text)
    text = _CM.sub(_with_suffix("xen ti mét"), text)
    text = _MM.sub(_with_suffix("mi li mét"), text)
    text = _M2.sub(_with_suffix("mét vuông"), text)
    return _M3.sub(_with_suffix("mét khối"), text)
