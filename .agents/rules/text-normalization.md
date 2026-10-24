---
trigger: model_decision
description: "Apply when changing Vietnamese text normalization, number/date/currency expansion, standalone-letter rules, or sentence splitting before phonemization."
---

# Vietnamese Text Normalization Rules

The normalizer turns written Vietnamese into the spoken form the G2P expects.
It lives in `app/text/` and runs inside `VoiceRuntime.sentences()`.

```
app/text/
├── __init__.py     normalize(text, language) — the ordered pipeline
├── data.py         lru_cached TSV loader + named accessors
├── data/           word tables copied from NeMo-text-processing (Apache-2.0)
├── numbers.py      cardinal, ordinal, decimal, read_number_token
├── dates.py        dates, times, weekdays, ranges
├── units.py        currency, measurement, percentage
└── letters.py      standalone letters + acronym-with-number
```

## Pipeline order is load-bearing

`normalize()` applies these passes in this exact order. Reordering silently
breaks cases that currently pass:

1. `unicodedata.normalize("NFC", text)`
2. `convert_units` — before dates, so `15km/h` is not read as a time
3. `convert_dates` — before numbers, so `2024-2025` is a range
4. `split_acronym_numbers` — **before** numbers, so `COVID-19` still has digits
   for the pattern to match
5. `convert_numbers` — grouped thousands, decimals, bare integers
6. `fix_standalone_letters` — **after** numbers, so `B` in `nhóm máu A và B`
   is still a letter

## There is no abbreviation dictionary

This is deliberate. Vietnamese abbreviations are an **open set**, so a
hand-maintained table is always incomplete and always needs a rebuild to
extend. Only closed sets are handled, and they are handled by rule:

| Closed set | Where | Why it is complete |
|---|---|---|
| The Latin alphabet | `letters.py::LETTER_NAMES` | 26 letters, one name each |
| Number/date/unit grammar | `numbers.py`, `dates.py`, `units.py` | Finite patterns |
| Acronym + digits | `letters.py::split_acronym_numbers` | Structural, not a lookup |

`TP.HCM` is therefore **not** expanded by this package. It is left to the
caller. If you need it, add it at the call site — do not reintroduce a
dictionary here.

## The dot-is-thousands rule

The single most important rule, and the one the previous implementation got
wrong. Vietnamese uses `.` for thousands and `,` for decimals:

```
1.234.567  -> một triệu hai trăm ba mươi tư nghìn năm trăm sáu mươi bảy
3.14       -> ba phẩy mười bốn
3,14       -> ba phẩy mười bốn
```

`read_number_token()` in `numbers.py` is the one place this is decided. Any
caller that pulls a number out of a larger pattern (`50.000đ`, `VND 1.500.000`)
must go through it.

## Word tables are data, not code

`data/` holds 25 TSV files copied verbatim from
[NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing)'s
Vietnamese grammar (Apache-2.0 — see `data/NOTICE`). **Only the data was
copied, not the Python.**

Why not depend on `nemo-text-processing`:

| | NeMo dependency | Vendored tables |
|---|---|---|
| Install size | `pynini` wheel is **147.6 MB** | **4.1 KB** of TSV |
| Windows | No wheel; source build fails (`cl : D8021 invalid numeric argument '/Wno-register'`) | Pure stdlib |
| Code needed | 1368 lines, of which 291 are pynini/FST | `data.py` is ~200 lines |

NeMo uses finite-state transducers because it must emit *every* valid reading
of an ambiguous input. A TTS frontend needs exactly **one** reading, so the FST
machinery buys nothing here. The remaining 1077 lines of NeMo's grammar just
read these same TSVs into dicts — which is what `data.py` does.

Two deliberate divergences from upstream, both documented in code:

- `numbers/magnitudes.tsv` says `linh` (southern); the project pins `lẻ`
  (northern) via `_FILLER` in `numbers.py`. The TSV stays byte-identical.
- `km` is not in `measure/base_units.tsv` (it is `k` + `m`), so `_KM` is
  spelled out in `units.py`.

To add a unit, month, or currency: **edit the TSV**, do not add a branch.

## Rules for changes

- **Add a test in `tests/test_text.py` for every case you touch.** The suite
  pins `01/05/2001`, `3.14`, `0.05`, `6.5`, `100k`, `2024`, `100%`, `15km`,
  `khoa Y`, `vitamin C`, `nhóm máu A và B`, `yêu quý ngày mai`.
- **Never widen a regex without checking the negative cases it now matches.**
- **Do not lowercase the whole string.** `Xin chào` must stay `Xin chào`.
- **Keep the functions pure.** No I/O, no logging, no settings lookups — this
  runs per sentence inside the synthesis hot path.
- **Keep the TSVs byte-identical to upstream.** A local edit makes the next
  upstream sync a manual merge. Put project choices in code, as `_FILLER` and
  `_KM` do.
