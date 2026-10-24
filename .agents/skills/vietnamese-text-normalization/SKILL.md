---
name: vietnamese-text-normalization
description: "Use when improving Vietnamese text normalization for TTS: numbers, dates, times, currency, units, percentages, ranges, acronyms, or standalone letters. Covers the current hand-rolled implementation, its measured defects, and evaluated third-party alternatives with licenses."
---

# Vietnamese Text Normalization for TTS

## When to use

- A number, date, price, time or unit is read incorrectly.
- Adding a new abbreviation, currency or measurement.
- Deciding whether to keep the hand-rolled normalizer or adopt a library.
- Reviewing a change to `normalize_vietnamese_text` in `app/engine.py`.

## Current implementation

`app/engine.py` holds four functions and two tables:

| Symbol | Purpose |
|---|---|
| `normalize_vietnamese_text` | Orchestrator, 5 ordered steps |
| `convert_numbers_to_vietnamese_words` | Dates → decimals → integers |
| `int_to_vietnamese_words` | Cardinal 0..999,999,999,999 |
| `digits_to_vietnamese_words` | Digit-by-digit for leading-zero strings |
| `VI_ABBREVIATIONS` | Ordered regex → replacement table |
| `STANDALONE_LETTER_FIXES` | `\bY\b → i`, `\bB\b → bê`, … |

Read `.agents/rules/text-normalization.md` for the pipeline order and the
defect table before changing anything.

## Measured defects

Run this to reproduce the baseline:

```bash
python -c "
from app.engine import normalize_vietnamese_text as n
for s in ['1.234.567','50.000đ','15:30','2024-2025','3-5%','COVID-19','1000001','thứ 4','10km/h','VND 100']:
    print(repr(s), '->', repr(n(s)))
"
```

The dominant failure is that `.` is treated as a decimal separator, so
`1.234.567` becomes `một phẩy hai ba bốn.năm trăm sáu mươi bảy`. Vietnamese
uses `.` for thousands and `,` for decimals — the regex
`\b(\d+)[.,](\d+)\b` cannot tell them apart without a lookahead for
`\d{3}\b` groups.

Second: `int_to_vietnamese_words` drops the `không trăm lẻ` filler above the
thousands group, so `1000001` → `một triệu một` instead of
`một triệu không trăm lẻ một`.

## Evaluated alternatives

| Option | License | Verdict |
|---|---|---|
| [`vietnormalizer`](https://pypi.org/project/vietnormalizer/) | MIT | **Best fit.** Purpose-built for TTS. Handles `50.000đ`, `1.500.000`, `15:30`, `2024-2025`, `3-5%`, `120km/h`, `thứ 2`, phone numbers, acronyms, transliteration. ~0.6 ms/call, 17K dictionary entries. |
| [`num2words`](https://pypi.org/project/num2words/) | LGPL-2.1 | Has `lang_VI.py` but it is **not registered** in `CONVERTER_CLASSES` — `num2words(123, to='vi')` raises `NotImplementedError`. LGPL is also awkward for a bundled server. Skip. |
| [`vinorm`](https://pypi.org/project/vinorm/) | "Free for non-commercial" | Non-commercial only. Cannot ship in an Apache-2.0 product. Skip. |
| [`viphoneme`](https://pypi.org/project/viphoneme/) | AILAB, non-commercial | Same problem. Skip. |
| [NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing) `vi` grammar | Apache-2.0 | Has a `vi` TN grammar, but needs `pynini` (OpenFst) — a heavy native build for a container that currently has no C++ toolchain beyond `build-essential`. Only worth it if you also want the `en` grammar for mixed text. |
| [`underthesea`](https://pypi.org/project/underthesea/) | Apache-2.0 | Full Vietnamese NLP toolkit; word segmentation and `text_normalize`. Much larger dependency than needed for TTS normalization. |

### What `vietnormalizer` gets right that we do not

```
50.000đ            -> năm mươi nghìn đồng
Giá 1.500.000 VND  -> giá một triệu năm trăm nghìn đồng
15:30              -> mười lăm giờ ba mươi phút
2024-2025          -> hai nghìn không trăm hai mươi tư đến hai nghìn không trăm hai mươi lăm
3-5%               -> ba đến năm phần trăm
10km/h             -> mười ki-lô-mét trên giờ
```

### What it gets wrong for us

```
khoa Y        -> khoa y      (we need "i" — upstream issue #4)
vitamin C     -> vi-ta-min c (we need "xê")
3.14          -> ba.mười bốn (broken decimal — it assumes dot = thousands)
100k          -> 100k        (we expand to "một trăm nghìn")
TP.HCM        -> tê pê.hát xê em (we expand to "Thành phố Hồ Chí Minh")
```

It also lowercases the whole string, which destroys proper-noun cues.

## Recommended approach

Do **not** swap wholesale. Adopt in layers:

1. **Extract** the normalizer into `app/text/` (see the rules file for the
   layout) so it can be tested and replaced per-concern.
2. **Borrow the number/date/currency/time logic** from `vietnormalizer` — that
   is where it is strongest and where we are weakest. Either depend on it
   behind a `Settings` flag, or port the rules with attribution (MIT permits
   this; keep the notice).
3. **Keep our own** `VI_ABBREVIATIONS` and `STANDALONE_LETTER_FIXES` — they
   encode project-specific decisions (`TP.HCM`, `Y → i`, `C → xê`) that no
   general library makes.
4. **Add a golden-file test** with a corpus of real sentences and their
   expected spoken form, so a library swap is measurable rather than vibes.

## Verification

```bash
python -m pytest tests/test_engine.py -q -k normalize
python -c "
from app.engine import normalize_vietnamese_text as n
print(n('Giá 1.500.000 VND, giảm 3-5% từ 15:30 ngày 25/12/2024'))
"
```

Expected after a correct fix:
`Giá một triệu năm trăm nghìn đồng, giảm ba đến năm phần trăm từ mười lăm giờ ba mươi phút ngày hai mươi lăm tháng mười hai năm hai nghìn không trăm hai mươi tư`
