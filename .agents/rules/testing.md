---
trigger: glob
globs: "tests/**"
description: "Test conventions for the TTS server: fakes instead of torch, what to pin, and how to run the suite."
---

# Testing Rules

## The suite must run without the model stack

`tests/` never imports `torch` or `kokoro_vietnamese`. `test_engine.py` defines
`FakeVoicepack` (a `__len__`-only object) and `FakeModel` (returns half a second
of ones) and injects them through `_patched_engine()`. Keep it that way — CI
runs on a bare runner with no GPU and no model download.

If you need a new upstream symbol, add it to the `ModuleType` mock in
`test_gpustack_local_directory_mount_resolution` rather than importing it.

## What to pin

- **Normalization**: exact string equality. `tests/test_engine.py` already pins
  `01/05/2001`, `3.14`, `0.05`, `6.5`, `100k`, `2024`, `100%`, `15km`,
  `TP.HCM`, `khoa Y`, `vitamin C`, `nhóm máu A và B`, `yêu quý ngày mai`.
  When you fix a defect from `.agents/rules/text-normalization.md`, change the
  expectation **and** add the neighbouring negative case.
- **Scheduler**: `_acquire()` / `_release()` counters, 503 on a full queue,
  503 when `max_queue=0` and the slot is taken.
- **HTTP**: `TestClient(app)` with the `_patch_engine_for_tests` fixture, which
  fakes `engine._model` so `/health` reports loaded. Assert on the OpenAI error
  envelope, not on `{"detail": ...}`.
- **Voices**: every alias and prefix form in `resolve()`.

## Commands

```bash
python -m pytest -q                       # whole suite
python -m pytest tests/test_engine.py -q  # one file
python -m pytest -q -k normalize          # one area
ruff check app tests && ruff format --check app tests
```

`pyproject.toml` sets `testpaths = ["tests"]` and ignores
`DeprecationWarning`. `pytest-asyncio` is available for async endpoints.

## Anti-patterns

- Asserting on audio *content*. Assert on shape, dtype, length and the
  crossfade seam instead.
- Downloading a voicepack in a test. Use `tmp_path` and the local-directory
  resolution path.
- Adding a test that needs `ffmpeg`. Guard it with `ffmpeg_available()` or
  skip it.
