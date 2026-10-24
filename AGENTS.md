# Kokoro-Vietnamese TTS Server — Agent Guide

OpenAI-compatible Vietnamese TTS API. FastAPI wraps the `kokoro-vietnamese`
package (Kokoro-82M / StyleTTS 2) and is deployed as a GPUStack custom backend.

## Layout

| Path | Role |
|---|---|
| `app/engine.py` | Model load, voice runtimes, phoneme fitting, scheduler, timeout, metrics |
| `app/text/` | Vietnamese normalization: numbers, dates, units, letters, optional NeMo |
| `app/cache.py` | Bounded LRU audio cache |
| `app/audio.py` | float32 → mp3/wav/flac/opus/aac/pcm, crossfade, concat |
| `app/server.py` | FastAPI app, `/v1/audio/speech`, streaming, probes, request id |
| `app/schemas.py` | Pydantic request models + the `APIError` envelope |
| `app/voices.py` | 14 voicepacks, OpenAI aliases, `resolve()` |
| `app/config.py` | Env-driven frozen `Settings` dataclass |
| `app/cli.py` | argparse entrypoint (`kokoro-vi-server`) |
| `app/ui.py` | Single-file Web Studio served at `/` |
| `tests/` | pytest; never imports torch or `kokoro_vietnamese` |

## Commands

```bash
pip install -e ".[test]"                    # install
python -m pytest -q                         # tests (no GPU, no model download)
ruff check app tests                        # lint
ruff format app tests                       # format
./scripts/smoke.sh http://127.0.0.1:8880    # live smoke test against a running server
```

## Non-negotiables

- **Never import torch at module level.** `tests/` must run without the model
  stack installed; import torch inside the function that needs it.
- **Errors go through `APIError`**, never FastAPI's default `{"detail": ...}`.
  OpenAI SDKs and GPUStack read `error.message` / `error.type` / `error.code`.
- **`Settings` is frozen.** Add a field, then wire it in `Settings.from_env()`
  and `app/cli.py`; never mutate an instance.
- **One model, many voicepacks.** `VoiceRuntime` shares `self._model`. Never
  construct a second `KokoroVietnamese` per voice — that is 14 copies of the
  network in memory.
- **Text normalization is Vietnamese-specific and order-sensitive.** Read
  `.agents/rules/text-normalization.md` before touching it.
- **No abbreviation dictionary.** Vietnamese abbreviations are an open set;
  only closed sets (the alphabet, the number grammar) are handled, and by rule.
- Line length 88; ruff `E,F,W,I,UP,B,C4,SIM,RUF`. Public functions get type
  hints and every module starts with `from __future__ import annotations`.

## Where to look first

- Changing synthesis behaviour → `.agents/rules/engine-and-audio.md`
- Changing request/response shape → `.agents/rules/api-contract.md`
- Adding or fixing tests → `.agents/rules/testing.md`
- Text, numbers, dates, units → `.agents/rules/text-normalization.md`
- Images, CI, GPUStack → `.agents/rules/deploy.md`
