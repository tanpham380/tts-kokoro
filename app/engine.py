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

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .audio import concat, crossfade
from .config import DEFAULT_SAMPLE_RATE, Settings
from .schemas import APIError
from .voices import VOICES, VoiceError, resolve

logger = logging.getLogger("tts_kokoro.engine")

import re
import unicodedata

#: Longest phoneme sequence the decoder accepts. Upstream raises above this, so
#: a sentence longer than the limit is split further rather than rejected.
MAX_PHONEMES = 510

#: Boundaries a too-long chunk is cut at, longest pause first.
_CLAUSE_BREAKS = (";", ",", ":", "—", "-", " ")

#: Fix for Issue #4 on iamdinhthuan/Kokoro-Vietnamese:
#: Single isolated letters mispronounced by English fallback (e.g. "Y" -> "quai").
STANDALONE_LETTER_FIXES: dict[str, str] = {
    r"\bY\b": "i",
    r"\by\b": "i",
    r"\bA\b": "a",
    r"\bE\b": "e",
    r"\bI\b": "i",
    r"\bO\b": "o",
    r"\bU\b": "u",
    r"\bB\b": "bê",
    r"\bC\b": "xê",
    r"\bD\b": "dê",
    r"\bF\b": "ép",
    r"\bG\b": "gờ",
    r"\bH\b": "hát",
    r"\bJ\b": "giây",
    r"\bK\b": "ca",
    r"\bL\b": "e-lờ",
    r"\bM\b": "em",
    r"\bN\b": "en",
    r"\bP\b": "pê",
    r"\bQ\b": "quy",
    r"\bR\b": "e-rờ",
    r"\bS\b": "ét",
    r"\bT\b": "tê",
    r"\bV\b": "vê",
    r"\bW\b": "vê-kép",
    r"\bX\b": "ích",
    r"\bZ\b": "dét",
}

#: Common Vietnamese acronyms and abbreviations for natural speech synthesis.
VI_ABBREVIATIONS: dict[str, str] = {
    r"\bTP\.?\s*HCM\b": "Thành phố Hồ Chí Minh",
    r"\bTP\.?\s*HN\b": "Thành phố Hà Nội",
    r"\bTP\b": "Thành phố",
    r"\bVN\b": "Việt Nam",
    r"\bVND\b": "đồng",
    r"\bUSD\b": "đô la",
    r"\bBS\.\b": "Bác sĩ",
    r"\bThS\.\b": "Thạc sĩ",
    r"\bTS\.\b": "Tiến sĩ",
    r"\bPGS\.\b": "Phó giáo sư",
    r"\bGS\.\b": "Giáo sư",
    r"(?<=\d|\s)km/h\b": " ki lô mét một giờ",
    r"(?<=\d|\s)km\b": " ki lô mét",
    r"(?<=\d|\s)kg\b": " ki lô gam",
    r"(?<=\d|\s)cm\b": " xen ti mét",
    r"(?<=\d|\s)mm\b": " mi li mét",
    r"(?<=\d|\s)m2\b": " mét vuông",
    r"(?<=\d|\s)m3\b": " mét khối",
    r"(?<=\d)k\b": " nghìn",
    r"%": " phần trăm",
    r"&": " và ",
}


def normalize_vietnamese_text(text: str) -> str:
    """Preprocess Vietnamese text: normalize abbreviations and fix standalone letters (Issue #4)."""
    if not text:
        return ""
    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Abbreviations
    for pattern, rep in VI_ABBREVIATIONS.items():
        text = re.sub(pattern, rep, text, flags=re.IGNORECASE)

    # 3. Standalone letter fixes (Issue #4: Y -> i, avoiding 'quai' pronunciation)
    for pattern, rep in STANDALONE_LETTER_FIXES.items():
        text = re.sub(pattern, rep, text)

    # 4. Clean extra spaces
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


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

        # Only apply Vietnamese acronyms & standalone letter fix if language is vi or auto
        if language in ("vi", "auto"):
            clean_text = normalize_vietnamese_text(text)
        else:
            clean_text = unicodedata.normalize("NFC", text).strip()

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

        # A semaphore rather than an executor. Synthesis is a single blocking
        # call and the HTTP server already runs it in a worker thread, so all
        # that is missing is a limit on how many run at once.
        self._slots = threading.BoundedSemaphore(max(1, settings.max_concurrency))
        self._counters = _Counters()

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
        try:
            self.load()
        except EngineError:
            # Already recorded in _load_error; reported by /health and /ready.
            pass

    def _build_runtime(self, voice: str) -> VoiceRuntime:
        """Build the model, or just a voicepack when the model already exists."""
        from kokoro_vietnamese import KokoroVietnamese

        if self._model is None:
            # Only the first instance pays for the network and the download.
            model_dir = Path(self.settings.model_path)
            model_file = self.settings.model_file
            config_file = self.settings.config_file
            repo_id = self.settings.model_path

            if model_dir.is_dir():
                # GPUStack mounts models to a local directory
                candidate_model = model_dir / model_file
                if candidate_model.exists():
                    model_file = str(candidate_model)
                candidate_cfg = model_dir / config_file
                if candidate_cfg.exists():
                    config_file = str(candidate_cfg)

            first = KokoroVietnamese(
                repo_id=repo_id,
                voice=voice,
                model_path=model_file,
                config_path=config_file,
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
        try:
            held: np.ndarray | None = None
            lang = language or "vi"
            for chunk in runtime.sentences(text, pace, language=lang):
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

    def _acquire(self) -> None:
        limit = self.settings.max_queue
        with self._counters.lock:
            if limit == 0:
                if not self._slots.acquire(blocking=False):
                    raise APIError(
                        "the server is busy: maximum concurrency reached and queueing is disabled",
                        status_code=503,
                        code="server_busy",
                    )
                return
            if limit and self._counters.queued >= limit:
                raise APIError(
                    f"the server is busy: {self._counters.queued} requests are "
                    f"already waiting (queue limit {limit})",
                    status_code=503,
                    code="server_busy",
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
            )

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
        return {
            **snapshot,
            "audio_seconds_total": round(audio, 3),
            "synth_seconds_total": round(synth, 3),
            "realtime_factor": round(rtf, 4) if rtf is not None else None,
            "rtf": round(rtf, 4) if rtf is not None else None,
            "loaded": self.is_loaded,
            "loading": self.is_loading,
            "device": self._device,
        }
