---
trigger: model_decision
description: "Apply when changing Vietnamese text normalization, number/date/currency expansion, abbreviation handling, standalone-letter fixes, or sentence splitting before phonemization."
---

# Vietnamese Text Normalization Rules

The normalizer turns written Vietnamese into the spoken form the G2P expects.
It lives in `app/engine.py` (`normalize_vietnamese_text`,
`convert_numbers_to_vietnamese_words`, `int_to_vietnamese_words`,
`digits_to_vietnamese_words`) and runs inside `VoiceRuntime.sentences()`.

## Pipeline order is load-bearing

`normalize_vietnamese_text` applies steps in this exact order. Reordering
silently breaks cases that currently pass:

1. `unicodedata.normalize("NFC", text)`
2. `VI_ABBREVIATIONS` — must run **before** number expansion so `15km`
   becomes `15 ki lô mét` and then `mười lăm ki lô mét`.
3. `convert_numbers_to_vietnamese_words` — dates, then decimals, then integers.
4. `STANDALONE_LETTER_FIXES` — must run **after** number expansion so `B` in
   `nhóm máu A và B` is still a letter and not a digit.
5. Whitespace collapse.

## Known defects — do not "fix" one without a test

These are real, reproducible outputs of the current code. Treat them as the
backlog, not as intended behaviour:

| Input | Current output | Expected |
|---|---|---|
| `1.234.567` | `một phẩy hai ba bốn.năm trăm sáu mươi bảy` | `một triệu hai trăm ba mươi tư nghìn năm trăm sáu mươi bảy` |
| `50.000đ` | `năm mươi.000đ` | `năm mươi nghìn đồng` |
| `Giá 1.500.000 VND` | `Giá một phẩy năm không không.không không không đồng` | `giá một triệu năm trăm nghìn đồng` |
| `15:30` | `mười lăm:ba mươi` | `mười lăm giờ ba mươi phút` |
| `12h30` | `12h30` (untouched) | `mười hai giờ ba mươi phút` |
| `2024-2025` | `…-…` (hyphen kept) | `… đến …` |
| `3-5%` | `ba-năm phần trăm` | `ba đến năm phần trăm` |
| `COVID-19` | `COVID-mười chín` | `cô vít mười chín` |
| `1000001` | `một triệu một` | `một triệu không trăm lẻ một` |
| `1000000001` | `một tỷ một` | `một tỷ không trăm lẻ một` |
| `thứ 4` | `thứ bốn` | `thứ tư` |
| `10km/h` | `mười ki lô mét một giờ` | `mười ki-lô-mét trên giờ` |
| `VND 100` | `đồng một trăm` | `một trăm đồng` |

The dot-as-thousands-separator bug is the worst one: `1.234.567` is read as a
decimal, so any price, population or ID in a document is mispronounced.

## Rules for changes

- **Add a test in `tests/test_engine.py` for every case you touch.** The
  existing tests pin `01/05/2001`, `3.14`, `0.05`, `6.5`, `100k`, `2024`,
  `100%`, `15km`, `TP.HCM`, `khoa Y`, `vitamin C` — do not regress them.
- **Never widen a regex to "catch more" without checking the negative cases.**
  `\b\d+\b` currently swallows phone numbers digit-by-digit, which is correct
  for `0901234567` but wrong for `1000001`.
- **`VI_ABBREVIATIONS` is an ordered dict and `re.sub` runs in insertion
  order.** `TP.HCM` must stay above `TP`. Adding a key at the end can shadow
  nothing, but moving one can.
- **Do not lowercase the whole string.** `Xin chào` must stay `Xin chào`; the
  G2P handles case, and lowercasing destroys proper-noun cues.
- **Keep the function pure.** No I/O, no logging, no settings lookups — it is
  called per sentence inside the synthesis hot path.

## Preferred direction

Prefer a dedicated `app/text/` package over growing `engine.py`:

```
app/text/
├── __init__.py        # normalize(text, language) -> str
├── numbers.py         # cardinal, ordinal, decimal, digit-by-digit
├── dates.py           # dd/mm/yyyy, ranges, times
├── units.py           # currency, measurement, percent
└── abbreviations.py   # acronym + standalone-letter tables
```

`engine.py` should only call `normalize(text, language)`. See the
`vietnamese-text-normalization` skill for the evaluated third-party options.
