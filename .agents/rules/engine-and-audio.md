---
trigger: glob
globs: "app/engine.py, app/audio.py, app/cache.py"
description: "Synthesis engine, voice runtime, phoneme fitting, scheduler, timeout, audio cache, crossfade and audio encoding rules."
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

## Scheduler, timeout and draining

- `_slots` is a `BoundedSemaphore(max_concurrency)`. `_acquire()` / `_release()`
  wrap the whole generator in a `try/finally`; `_release` must run even when the
  client disconnects mid-stream.
- `max_queue=0` means "no queueing": a taken slot returns 503 immediately.
- `synthesis_timeout` is a **deadline on the whole request**, checked between
  chunks. It exists because `max_text_chars` bounds the input but not the decode
  time — without it one long input holds the only GPU slot indefinitely.
- `begin_shutdown()` sets `_draining`; `_require_loaded()` then refuses new work
  with `server_shutting_down`. In-flight requests are left to finish.

## Audio cache

- `AudioCache` is bounded on **both** entry count and total bytes. An entry cap
  alone lets a few long inputs blow the memory budget.
- `CacheKey` must include everything that changes the bytes: text, voice, speed,
  format, sample rate. Omitting any of them serves the wrong audio.
- Only the **batch** path is cached. A streaming response is consumed as it is
  produced, so there is nothing to store until the last chunk arrives.
- The cache is process-local. A multi-worker deployment would need Redis; the
  hit rate does not justify that for a single-GPU server.

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

## Metrics

- `metrics()` returns JSON for humans and the Web Studio.
- `prometheus()` returns the scrapeable text exposition. Keep both in step: a
  counter added to one must be added to the other.
- Only counters and gauges are exposed. A latency histogram would need
  per-request buckets, which the engine does not keep.

## Style

- Module-level torch imports are forbidden; import inside the function.
- Counters live in `_Counters` behind `threading.Lock`. Read them through
  `metrics()`, never by reaching into the dataclass from outside.
