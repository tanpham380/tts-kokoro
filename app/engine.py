"""The synthesis engine.

Two things about ``kokoro-vietnamese`` shape this module:

* A ``KokoroVietnamese`` instance owns a full ``KModel`` **and** one voicepack,
  and the voicepack is loaded in ``__init__``. Building one instance per voice
  would put a separate copy of the network in memory per voice — fourteen
  copies of a model that only needs one.
* ``synthesize()`` always processes the whole input and returns one array, so
  it cannot drive a streaming response.

So the model is built once and every voice becomes a :class:`VoiceRuntime`
sharing that one model with its own voicepack. Sentence splitting,
phonemization and the decoder call are the same operations upstream's
``synthesize()`` performs, which is what lets :meth:`TTSEngine.stream` emit
audio sentence by sentence instead of all at the end.

Nothing here imports torch at module level: the test suite replaces the engine
with a scripted fake and must run without the model stack installed.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .audio import concat, crossfade
from .cache import AudioCache, CacheKey
from .config import DEFAULT_SAMPLE_RATE, Settings
from .schemas import APIError
from .text import (
    convert_numbers,
    digits_to_words,
    int_to_words,
    normalize,
)
from .voices import VOICES, VoiceError, resolve

logger = logging.getLogger("tts_kokoro.engine")

#: Longest phoneme sequence the decoder accepts. Upstream raises above this, so
#: a sentence longer than the limit is split further rather than rejected.
MAX_PHONEMES = 510

#: Boundaries a too-long chunk is cut at, longest pause first.
_CLAUSE_BREAKS = (";", ",", ":", "—", "-", " ")

#: Backwards-compatible aliases. The implementation moved to :mod:`app.text`;
#: these names are kept because tests and downstream callers import them.
int_to_vietnamese_words = int_to_words
digits_to_vietnamese_words = digits_to_words
convert_numbers_to_vietnamese_words = convert_numbers
normalize_vietnamese_text = normalize

class EngineError(RuntimeError):
    """Raised when the model or a voicepack cannot be prepared."""


@dataclass
class VoiceRuntime:
    """A voicepack plus the shared model it runs on."""

    id: str
    voicepack: Any
    model: Any
    device: str

    def decode(self, phonemes: str, speed: float) -> np.ndarray:
        """Run one phoneme sequence through the decoder."""
        import torch

        with torch.no_grad():
            ref_s = self.voicepack[len(phonemes) - 1]
            audio = self.model(phonemes, ref_s, float(speed))
        return np.asarray(audio.detach().cpu().numpy(), dtype=np.float32)

    def sentences(
        self,
        text: str,
        speed: float,
        language: str = "vi",
    ) -> Iterator[np.ndarray]:
        """Yield one array per sentence, in order."""
        from kokoro_vietnamese import phonemize, split_text

        # ``normalize`` already handles the non-Vietnamese case (NFC + whitespace).
        clean_text = normalize(text, language)

        for text_chunk in split_text(clean_text):
            for piece in _fit_phonemes(text_chunk):
                phonemes = phonemize(piece)
                if not phonemes:
                    continue
                if len(phonemes) > MAX_PHONEMES:
                    raise EngineError(
                        f"phoneme sequence of {len(phonemes)} exceeds the "
                        f"decoder limit of {MAX_PHONEMES}"
                    )
                yield self.decode(phonemes, speed)


def _fit_phonemes(text_chunk: str, depth: int = 0) -> list[str]:
    """Split ``text_chunk`` until each piece is short enough to phonemize."""
    from kokoro_vietnamese import phonemize

    if len(phonemize(text_chunk)) <= MAX_PHONEMES or depth >= 8:
        return [text_chunk]

    cutoff = len(text_chunk) // 2
    split_at = 0
    for token in _CLAUSE_BREAKS:
        position = text_chunk.rfind(token, 0, cutoff)
        if position > 0:
            split_at = max(split_at, position + len(token))
    if not split_at:
        split_at = cutoff

    head, tail = text_chunk[:split_at].strip(), text_chunk[split_at:].strip()
    if not head or not tail:
        return [text_chunk]
    return [*_fit_phonemes(head, depth + 1), *_fit_phonemes(tail, depth + 1)]

def _detect_device(requested: str | None) -> str:
    """Resolve the ``device`` setting, falling back as far as ``cpu``."""
    try:
        import torch
    except Exception:  # pragma: no cover - torch is in any real image
        return requested or "cpu"

    if requested and requested.lower() not in ("auto", ""):
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("device=%s requested but CUDA is unavailable", requested)
            return "cpu"
        return requested

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"

@dataclass
class _Counters:
    requests: int = 0
    succeeded: int = 0
    failed: int = 0
    audio_seconds: float = 0.0
    synth_seconds: float = 0.0
    queued: int = 0
    max_queued: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

class TTSEngine:
    """Owns the model, the voice cache and the concurrency gate."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sample_rate = DEFAULT_SAMPLE_RATE
        #: A voice name, not a voicepack: resolution happens per request.
        self.default_voice = settings.default_voice

        self._model: Any = None
        self._voices: dict[str, VoiceRuntime] = {}
        self._default_runtime: VoiceRuntime | None = None
        self._device: str | None = None
        self._load_lock = threading.Lock()
        self._voice_lock = threading.Lock()
        self._is_loading = False
        self._loaded_at: float | None = None
        self._load_seconds: float | None = None
        self._load_error: str | None = None
        self._draining = False

        # A semaphore rather than an executor. Synthesis is a single blocking
        # call and the HTTP server already runs it in a worker thread, so all
        # that is missing is a limit on how many run at once.
        self._slots = threading.BoundedSemaphore(max(1, settings.max_concurrency))
        self._counters = _Counters()
        self._cache = AudioCache(
            max_entries=settings.cache_max_entries,
            max_bytes=settings.cache_max_bytes,
        )

    # -- state ---------------------------------------------------------------

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._is_loading

    @property
    def is_ready(self) -> bool:
        return self.is_loaded and self._load_error is None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def is_draining(self) -> bool:
        """True once shutdown has started; new requests should be refused."""
        return self._draining

    def begin_shutdown(self) -> None:
        """Stop accepting new work. In-flight requests are left to finish."""
        self._draining = True

    # -- loading -------------------------------------------------------------

    def load(self) -> None:
        """Download the weights (first run only) and build the model.

        Idempotent and safe to call from the startup hook and a request handler
        at the same time: a second caller waits for the first to finish instead
        of building a second model.
        """
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            self._is_loading = True
            started = time.perf_counter()
            try:
                self._device = _detect_device(self.settings.device)
                logger.info(
                    "loading Kokoro-Vietnamese from %s on %s",
                    self.settings.model_path,
                    self._device,
                )
                self._build_runtime(self.settings.default_voice)
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.exception("failed to load the Kokoro-Vietnamese model")
                raise EngineError(self._load_error) from exc
            else:
                self._load_error = None
                self._loaded_at = time.time()
                if self.settings.warmup:
                    self._warmup()
            finally:
                self._is_loading = False
                self._load_seconds = round(time.perf_counter() - started, 3)

    def load_in_background(self) -> threading.Thread:
        """Kick off :meth:`load` on a daemon thread and return it."""
        thread = threading.Thread(
            target=self._load_quietly, name="kokoro-load", daemon=True
        )
        thread.start()
        return thread

    def _load_quietly(self) -> None:
        # Already recorded in _load_error; reported by /health and /ready.
        with contextlib.suppress(EngineError):
            self.load()

    def _build_runtime(self, voice: str) -> VoiceRuntime:
        """Build the model, or just a voicepack when the model already exists."""
        from kokoro_vietnamese import KokoroVietnamese

        if self._model is None:
            # When GPUStack mounts models to a local directory, resolve files locally
            # to guarantee zero network calls / zero redundant downloads.
            model_dir = Path(self.settings.model_path)
            model_file = self.settings.model_file
            config_file = self.settings.config_file
            repo_id = self.settings.model_path
            voicepack_file: str | None = None

            if model_dir.is_dir():
                # 1. Resolve model checkpoint (supports both contextboxai & dinhthuan variants)
                model_candidates = [
                    model_dir / model_file,
                    model_dir / "kokoro_vi.pth",
                    model_dir / "kokoro_vi_ngoc_huyen.pth",
                    model_dir / "model" / "kokoro_vi_ngoc_huyen.pth",
                ]
                for cand in model_candidates:
                    if cand.is_file():
                        model_file = str(cand)
                        break
                else:
                    pth_files = sorted(list(model_dir.glob("*.pth")) + list(model_dir.glob("*/*.pth")))
                    if pth_files:
                        model_file = str(pth_files[0])

                # 2. Resolve config.json
                cfg_candidates = [
                    model_dir / config_file,
                    model_dir / "config.json",
                ]
                for cand in cfg_candidates:
                    if cand.is_file():
                        config_file = str(cand)
                        break

                # 3. Resolve initial voicepack locally
                vp_candidates = [
                    model_dir / "voicepacks" / f"{voice}.pt",
                    model_dir / "voices" / f"{voice}.pt",
                    model_dir / f"{voice}.pt",
                ]
                for cand in vp_candidates:
                    if cand.is_file():
                        voicepack_file = str(cand)
                        break

            first = KokoroVietnamese(
                repo_id=repo_id,
                voice=voice,
                model_path=model_file,
                config_path=config_file,
                voicepack_path=voicepack_file,
                device=self._device or "cpu",
            )
            self._model = first.model
            runtime = VoiceRuntime(
                id=voice,
                voicepack=first.voicepack,
                model=first.model,
                device=str(first.device),
            )
        else:
            runtime = VoiceRuntime(
                id=voice,
                voicepack=self._load_voicepack(voice),
                model=self._model,
                device=self._device or "cpu",
            )

        with self._voice_lock:
            self._voices[voice] = runtime
            if self._default_runtime is None or voice == self.settings.default_voice:
                self._default_runtime = runtime
        logger.info("voice %s ready (%d voice(s) loaded)", voice, len(self._voices))
        return runtime

    def _load_voicepack(self, voice: str) -> Any:
        import torch
        from kokoro_vietnamese import resolve_voicepack_filename

        # 1. Custom voicepack cache directory
        if self.settings.voicepack_dir:
            local = Path(self.settings.voicepack_dir) / f"{voice}.pt"
            if local.exists():
                return torch.load(local, map_location="cpu", weights_only=True)

        # 2. Local directory in model_path (when GPUStack mounts model locally)
        model_dir = Path(self.settings.model_path)
        if model_dir.is_dir():
            candidates = [
                model_dir / "voicepacks" / f"{voice}.pt",
                model_dir / "voices" / f"{voice}.pt",
                model_dir / f"{voice}.pt",
                model_dir / resolve_voicepack_filename(voice, None),
            ]
            for cand in candidates:
                if cand.exists():
                    return torch.load(cand, map_location="cpu", weights_only=True)

        # 3. Fallback to downloading via Hugging Face Hub
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError

        hf_candidates = [
            resolve_voicepack_filename(voice, None),
            f"voices/{voice}.pt",
            f"voicepacks/{voice}.pt",
            f"{voice}.pt",
        ]
        last_err = None
        for fn in hf_candidates:
            try:
                path = hf_hub_download(
                    repo_id=self.settings.model_path,
                    filename=fn,
                    token=self.settings.hf_token,
                )
                return torch.load(path, map_location="cpu", weights_only=True)
            except EntryNotFoundError as exc:
                last_err = exc
                continue
            except Exception as exc:
                last_err = exc
                continue
        if last_err:
            raise last_err

    def _runtime_for(self, voice: str) -> VoiceRuntime:
        with self._voice_lock:
            runtime = self._voices.get(voice)
        return runtime if runtime is not None else self._build_runtime(voice)

    def _warmup(self) -> None:
        """One short sentence so the first real request is not the slow one.

        CUDA allocates and compiles kernels on the first use of each shape,
        which can take seconds. A failure here is not fatal: the model loaded,
        so requests still work even if warmup did not.
        """
        runtime = self._default_runtime
        if runtime is None:
            return
        try:
            next(iter(runtime.sentences("Xin chào.", 1.0)))
        except Exception:  # pragma: no cover - depends on the local accelerator
            logger.debug("warmup failed", exc_info=True)

    def _require_loaded(self) -> None:
        if self._draining:
            raise APIError(
                "the server is shutting down and is not accepting new requests",
                status_code=503,
                code="server_shutting_down",
                retry_after=5.0,
            )
        if self.is_loaded:
            return
        if self._load_error:
            raise APIError(
                f"the model failed to load: {self._load_error}",
                status_code=503,
                code="model_unavailable",
            )
        raise APIError(
            "the model is still loading; retry shortly",
            status_code=503,
            code="model_loading",
            retry_after=2.0,
        )

    # -- synthesis -----------------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        crossfade_ms: int | None = None,
        volume_multiplier: float = 1.0,
        language: str | None = "vi",
        task_type: str | None = None,
        instructions: str | None = None,
        reference_audio: str | None = None,
        use_speaker_embedding_only: bool | None = None,
    ) -> np.ndarray:
        """Synthesize ``text`` into one float32 mono array."""
        audio = concat(
            list(
                self.stream(
                    text,
                    voice=voice,
                    speed=speed,
                    crossfade_ms=crossfade_ms,
                    volume_multiplier=volume_multiplier,
                    language=language,
                    task_type=task_type,
                    instructions=instructions,
                    reference_audio=reference_audio,
                    use_speaker_embedding_only=use_speaker_embedding_only,
                )
            )
        )

        # Only possible here: a stream has to start before its peak is known.
        peak = self.settings.normalize_peak
        if peak and peak > 0 and audio.size:
            loudest = float(np.max(np.abs(audio)))
            if loudest > peak:
                audio = (audio * (peak / loudest)).astype(np.float32, copy=False)
        return audio

    def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        crossfade_ms: int | None = None,
        volume_multiplier: float = 1.0,
        language: str | None = "vi",
        task_type: str | None = None,
        instructions: str | None = None,
        reference_audio: str | None = None,
        use_speaker_embedding_only: bool | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield audio sentence by sentence, crossfading the joins.

        The trailing ``fade`` samples of each chunk are held back so the
        crossfade can be applied once the next chunk exists. That is what makes
        streamed audio join as seamlessly as the batched path.
        """
        self._require_loaded()
        try:
            resolved = resolve(voice, self.default_voice)
        except VoiceError as exc:
            raise APIError(
                str(exc), status_code=400, code="invalid_voice", param="voice"
            ) from exc

        if instructions:
            logger.debug("GPUStack instructions received: %s", instructions)
        if reference_audio:
            logger.debug(
                "GPUStack reference_audio: %s (speaker_embedding_only: %s)",
                reference_audio,
                use_speaker_embedding_only,
            )

        runtime = self._runtime_for(resolved)
        pace = float(speed if speed is not None else self.settings.speed)
        fade = self.settings.crossfade_ms if crossfade_ms is None else crossfade_ms
        fade_samples = max(0, round(self.sample_rate * fade / 1000))

        self._acquire()
        started = time.perf_counter()
        produced = 0
        failed = True
        # A deadline rather than a per-chunk timeout: the limit is on the whole
        # request, and checking between chunks is enough because one chunk is
        # one sentence.
        timeout = self.settings.synthesis_timeout
        deadline = started + timeout if timeout and timeout > 0 else None
        try:
            held: np.ndarray | None = None
            lang = language or "vi"
            for chunk in runtime.sentences(text, pace, language=lang):
                if deadline is not None and time.perf_counter() > deadline:
                    raise APIError(
                        f"synthesis exceeded the {timeout:.0f}s limit",
                        status_code=503,
                        code="synthesis_timeout",
                        retry_after=self._retry_hint(),
                    )
                if volume_multiplier != 1.0:
                    chunk = (chunk * float(volume_multiplier)).astype(np.float32)
                if held is not None:
                    chunk = crossfade(held, chunk, fade_samples)
                produced += chunk.size

                if fade_samples and chunk.size > fade_samples:
                    yield chunk[:-fade_samples]
                    held = chunk[-fade_samples:]
                else:
                    yield chunk
                    held = None
            if held is not None and held.size:
                yield held
            failed = False
        finally:
            self._release(produced, time.perf_counter() - started, failed=failed)

    def cache_get(self, key: CacheKey) -> bytes | None:
        """Return cached encoded audio, or ``None``."""
        return self._cache.get(key)

    def cache_put(self, key: CacheKey, value: bytes) -> None:
        """Store encoded audio for reuse."""
        self._cache.put(key, value)

    def cache_clear(self) -> None:
        self._cache.clear()

    def _acquire(self) -> None:
        limit = self.settings.max_queue
        # Computed before the lock is taken: ``_retry_hint`` reads the counters
        # under the same lock, so calling it inside the block would deadlock.
        hint = self._retry_hint()
        with self._counters.lock:
            if limit == 0:
                if not self._slots.acquire(blocking=False):
                    raise APIError(
                        "the server is busy: maximum concurrency reached and queueing is disabled",
                        status_code=503,
                        code="server_busy",
                        retry_after=hint,
                    )
                return
            if limit and self._counters.queued >= limit:
                raise APIError(
                    f"the server is busy: {self._counters.queued} requests are "
                    f"already waiting (queue limit {limit})",
                    status_code=503,
                    code="server_busy",
                    retry_after=hint,
                )
            self._counters.queued += 1
            self._counters.max_queued = max(
                self._counters.max_queued, self._counters.queued
            )

        timeout = self.settings.queue_timeout
        acquired = self._slots.acquire(timeout=timeout)
        with self._counters.lock:
            self._counters.queued -= 1

        if not acquired:
            raise APIError(
                f"the server is busy: request timed out waiting for a synthesis slot ({timeout}s)",
                status_code=503,
                code="server_busy",
                retry_after=hint,
            )

    def _retry_hint(self) -> float:
        """Seconds a client should wait before retrying a 503.

        Derived from the observed decode time per request: if a request that
        produced 10 s of audio took 20 s, a waiting client should expect a
        similar wait. Falls back to 1 s before any request has completed, and is
        clamped so a pathological value cannot ask a client to sleep for hours.

        Must not be called while ``_counters.lock`` is held.
        """
        counters = self._counters
        with counters.lock:
            synth = counters.synth_seconds
            requests = counters.requests
        if synth <= 0 or requests <= 0:
            return 1.0
        average = synth / requests
        return float(min(60.0, max(1.0, average)))

    def _release(self, samples: int, seconds: float, *, failed: bool) -> None:
        self._slots.release()
        with self._counters.lock:
            self._counters.requests += 1
            self._counters.synth_seconds += seconds
            self._counters.audio_seconds += samples / float(self.sample_rate)
            if failed:
                self._counters.failed += 1
            else:
                self._counters.succeeded += 1

    # -- introspection -------------------------------------------------------

    def info(self) -> dict[str, Any]:
        return {
            "engine": "kokoro-vietnamese",
            "model": self.settings.served_model_name,
            "model_path": self.settings.model_path,
            "loaded": self.is_loaded,
            "loading": self.is_loading,
            "ready": self.is_ready,
            "device": self._device,
            "profile": self.settings.profile,
            "language": "vi",
            "sample_rate": self.sample_rate,
            "default_voice": self.default_voice,
            "voices_loaded": sorted(self._voices),
            "voices_available": len(VOICES),
            "crossfade_ms": self.settings.crossfade_ms,
            "normalize_peak": self.settings.normalize_peak,
            "max_text_chars": self.settings.max_text_chars,
            "max_concurrency": self.settings.max_concurrency,
            "max_queue": self.settings.max_queue,
            "load_error": self._load_error,
            "load_seconds": self._load_seconds,
            "loaded_at": self._loaded_at,
        }

    def metrics(self) -> dict[str, Any]:
        counters = self._counters
        with counters.lock:
            audio, synth = counters.audio_seconds, counters.synth_seconds
            snapshot = {
                "requests_total": counters.requests,
                "requests_success": counters.succeeded,
                "requests_failed": counters.failed,
                "queued": counters.queued,
                "max_queued": counters.max_queued,
            }
        # Synthesis speed relative to playback: below 1.0 is faster than real
        # time, above 1.0 means a client would underrun.
        rtf = (synth / audio) if audio else None
        cache = self._cache.stats()
        return {
            **snapshot,
            "audio_seconds_total": round(audio, 3),
            "synth_seconds_total": round(synth, 3),
            "realtime_factor": round(rtf, 4) if rtf is not None else None,
            "rtf": round(rtf, 4) if rtf is not None else None,
            "cache_enabled": self._cache.enabled,
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "cache_evictions": cache.evictions,
            "cache_entries": cache.entries,
            "cache_bytes": cache.bytes,
            "loaded": self.is_loaded,
            "loading": self.is_loading,
            "device": self._device,
        }

    def prometheus(self) -> str:
        """Render the counters in Prometheus text exposition format.

        ``/metrics`` returns JSON for humans and the Web Studio; this is the
        scrapeable form. Only counters and gauges are exposed — a latency
        histogram would need per-request buckets, which the engine does not
        keep.
        """
        counters = self._counters
        with counters.lock:
            audio, synth = counters.audio_seconds, counters.synth_seconds
            snapshot = {
                "requests_total": counters.requests,
                "requests_success": counters.succeeded,
                "requests_failed": counters.failed,
                "queued": counters.queued,
                "max_queued": counters.max_queued,
            }
        cache = self._cache.stats()
        rtf = (synth / audio) if audio else 0.0

        lines: list[str] = []

        def _metric(name: str, kind: str, help_text: str, value: float) -> None:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {kind}")
            lines.append(f"{name} {value}")

        _metric(
            "kokoro_requests_total",
            "counter",
            "Synthesis requests accepted.",
            snapshot["requests_total"],
        )
        _metric(
            "kokoro_requests_success_total",
            "counter",
            "Synthesis requests that completed.",
            snapshot["requests_success"],
        )
        _metric(
            "kokoro_requests_failed_total",
            "counter",
            "Synthesis requests that raised.",
            snapshot["requests_failed"],
        )
        _metric(
            "kokoro_queue_depth",
            "gauge",
            "Requests currently waiting for a synthesis slot.",
            snapshot["queued"],
        )
        _metric(
            "kokoro_queue_max_depth",
            "gauge",
            "High-water mark of the queue depth.",
            snapshot["max_queued"],
        )
        _metric(
            "kokoro_audio_seconds_total",
            "counter",
            "Audio produced, in seconds.",
            round(audio, 3),
        )
        _metric(
            "kokoro_synth_seconds_total",
            "counter",
            "Time spent decoding, in seconds.",
            round(synth, 3),
        )
        _metric(
            "kokoro_realtime_factor",
            "gauge",
            "Decode seconds per audio second; below 1.0 is faster than real time.",
            round(rtf, 4),
        )
        _metric(
            "kokoro_model_loaded",
            "gauge",
            "1 when the model is in memory.",
            1 if self.is_loaded else 0,
        )
        _metric(
            "kokoro_voices_loaded",
            "gauge",
            "Voicepacks currently resident.",
            len(self._voices),
        )
        _metric(
            "kokoro_cache_hits_total",
            "counter",
            "Audio cache hits.",
            cache.hits,
        )
        _metric(
            "kokoro_cache_misses_total",
            "counter",
            "Audio cache misses.",
            cache.misses,
        )
        _metric(
            "kokoro_cache_entries",
            "gauge",
            "Entries currently in the audio cache.",
            cache.entries,
        )
        _metric(
            "kokoro_cache_bytes",
            "gauge",
            "Bytes currently held by the audio cache.",
            cache.bytes,
        )
        return "\n".join(lines) + "\n"
