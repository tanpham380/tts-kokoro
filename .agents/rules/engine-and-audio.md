---
trigger: glob
globs: "app/engine.py, app/audio.py"
description: "Synthesis engine, voice runtime, phoneme fitting, scheduler, crossfade and audio encoding rules."
---

# Engine & Audio Rules

## Model and voice lifecycle

- `TTSEngine._model` is built **once**. `VoiceRuntime` holds a voicepack plus a
  reference to that shared model. `_build_runtime()` must keep the
  `if self._model is None:` branch — that is what makes voice #2..#14 cheap.
- `_load_lock` guards model construction, `_voice_lock` guards `self._voices`.
  Do not merge them; a voice load must not block behind a model load.
- `load()` is idempotent and safe to call from the startup hook and a request
  handler concurrently. Preserve the double-checked `is_loaded` pattern.
- `_detect_device()` falls back `cuda → mps → cpu`. A pinned `cuda` on a host
  without CUDA logs a warning and returns `cpu` rather than raising.

## Phoneme budget

- `MAX_PHONEMES = 510` is the decoder limit. `_fit_phonemes()` recursively
  splits a chunk at `_CLAUSE_BREAKS` (`; , : — - space`), longest pause first,
  and gives up at `depth >= 8`.
- `_fit_phonemes` calls `phonemize()` on every recursion level, so a long
  paragraph is phonemized O(n log n) times. If you touch it, measure before and
  after — this is the main CPU cost on the streaming path.
- A chunk that still exceeds the limit after splitting raises `EngineError`,
  which surfaces as HTTP 503. Do not silently truncate audio.

## Streaming contract

`TTSEngine.stream()` holds back the trailing `fade_samples` of each chunk so
the next chunk can be crossfaded into it, then flushes the remainder after the
loop. Consequences:

- The first byte is delayed by one sentence. That is intentional.
- `synthesize()` is `concat(list(stream(...)))` plus the optional peak limiter.
  The limiter **cannot** run on the streaming path because the peak is not
  known until the end — do not try to add it there.
- `_acquire()` / `_release()` wrap the whole generator in a `try/finally`.
  `_release` must run even when the client disconnects mid-stream.

## Audio encoding

- `wav` and `flac` go through `soundfile`; `mp3`, `aac`, `opus` shell out to
  `ffmpeg` reading raw S16LE from a pipe; `pcm` is passed through untouched.
- `to_pcm16le` **clamps** rather than wraps. Wrapping turns a slightly loud
  sample into an audible click.
- Streaming forces `wav`/`pcm` because per-chunk `ffmpeg` invocations reset the
  encoder and produce seams. Keep that guard in `app/server.py`.
- `crossfade()` is a linear fade over `fade_samples`. `crossfade_ms=0` means
  concatenate with an audible seam — that is a supported configuration, not a
  bug.

## Style

- Module-level torch imports are forbidden; import inside the function.
- Counters live in `_Counters` behind `threading.Lock`. Read them through
  `metrics()`, never by reaching into the dataclass from outside.
