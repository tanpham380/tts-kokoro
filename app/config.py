"""Runtime configuration.

Everything is driven by environment variables so the same image works as a
GPUStack custom backend (which injects env vars) or as a plain ``docker run``.
Command line flags, when present, win over the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import os

DEFAULT_REPO_ID = "contextboxai/Kokoro-Vietnamese"
DEFAULT_SERVED_NAME = "Kokoro-Vietnamese"
DEFAULT_VOICE = "diem_trinh"
DEFAULT_SAMPLE_RATE = 24000

#: Output formats ``/v1/audio/speech`` accepts. ``pcm`` is headerless signed
#: 16-bit little-endian, which is what OpenAI returns for that value.
RESPONSE_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
DEFAULT_RESPONSE_FORMAT = "mp3"

#: Streaming modes. ``sse`` mirrors OpenAI's ``speech.audio.delta`` events;
#: ``audio`` is a plain chunked byte stream, which browsers can play directly.
STREAM_FORMATS = ("sse", "audio")

def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()

def _flag(name: str, default: bool) -> bool:
    raw = _get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "y", "on")

def _maybe_int(name: str, default: int | None = None) -> int | None:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

def _maybe_float(name: str, default: float | None = None) -> float | None:
    raw = _get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

def _maybe_str(name: str) -> str | None:
    return _get(name)

@dataclass(frozen=True)
class Settings:
    """Immutable server settings."""

    # --- model ---------------------------------------------------------------
    #: A Hugging Face repo id holding ``kokoro_vi.pth``, ``config.json`` and a
    #: ``voicepacks/`` folder, or a local directory laid out the same way.
    model_path: str = DEFAULT_REPO_ID
    served_model_name: str = DEFAULT_SERVED_NAME
    #: Specific filenames inside ``model_path``. Only change these if you point
    #: ``model_path`` at a re-exported checkpoint.
    model_file: str = "kokoro_vi.pth"
    config_file: str = "config.json"
    #: Where voicepacks are cached. Empty means "next to the downloaded model".
    voicepack_dir: str | None = None
    hf_token: str | None = None
    #: ``cuda``, ``cuda:1``, ``mps``, ``cpu``, or ``auto`` to let torch decide.
    device: str | None = None
    #: Load the model during startup instead of on the first request.
    preload: bool = True
    #: Synthesize a short sentence after loading so the first real request does
    #: not pay for lazy CUDA kernel compilation.
    warmup: bool = True

    # --- synthesis -----------------------------------------------------------
    default_voice: str = DEFAULT_VOICE
    #: Default pace. A request's ``speed`` wins over this.
    speed: float = 1.0
    #: Crossfade applied where two sentence chunks are joined, in milliseconds.
    #: 0 concatenates them with an audible seam.
    crossfade_ms: int = 50
    #: Optional peak limiter applied to batched output. ``None`` disables it;
    #: streaming responses never apply it because the peak is not yet known.
    normalize_peak: float | None = None
    #: Reject longer inputs instead of holding the GPU for minutes.
    max_text_chars: int = 20000

    # --- scheduler -----------------------------------------------------------
    #: Decodes running at once. Kokoro is a small model, so 1 is right for a
    #: single GPU; raise it on a multi-GPU box with one worker per device.
    max_concurrency: int = 1
    #: Requests allowed to wait for a slot; a full queue returns HTTP 503
    #: instead of growing latency without bound. 0 disables the cap.
    max_queue: int = 256
    #: Maximum seconds a request can wait in the queue before timing out with 503.
    queue_timeout: float = 60.0
    #: Maximum seconds one synthesis may run before it is abandoned with 503.
    #: Without this a single long input holds the only GPU slot indefinitely,
    #: because ``max_text_chars`` bounds the input but not the decode time.
    #: 0 disables the limit.
    synthesis_timeout: float = 300.0

    # --- cache ---------------------------------------------------------------
    #: Cache synthesized audio keyed by (text, voice, speed, format). Repeated
    #: text is common in production (prompts, IVR lines), and a hit skips the
    #: decode entirely. 0 disables the cache.
    cache_max_entries: int = 256
    #: Maximum bytes the audio cache may hold. 0 disables the cache.
    cache_max_bytes: int = 256 * 1024 * 1024

    # --- http ----------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8880
    api_key: str | None = None
    cors_origins: tuple[str, ...] = ("*",)
    log_level: str = "info"
    #: Emit one JSON object per log line so Loki/ELK can parse it.
    log_json: bool = False

    @property
    def profile(self) -> str:
        """The hardware profile this configuration looks tuned for.

        Derived rather than declared: ``cpu``/``mps`` when a device was pinned
        to one, ``cuda`` when a CUDA device was pinned, and ``auto`` when the
        device is left to torch's own detection.
        """
        device = (self.device or "auto").lower()
        if device.startswith(("cuda", "rocm")):
            return "cuda"
        if device.startswith(("cpu", "mps")):
            return device.split(":")[0]
        return "auto"

    @classmethod
    def from_env(cls) -> Settings:
        origins = _get("CORS_ORIGINS", "*")
        return cls(
            model_path=(
                _get("MODEL_PATH") or _get("KOKORO_MODEL") or DEFAULT_REPO_ID
            ),
            served_model_name=(
                _get("SERVED_MODEL_NAME") or _get("MODEL_NAME") or DEFAULT_SERVED_NAME
            ),
            model_file=_get("MODEL_FILE", "kokoro_vi.pth"),
            config_file=_get("CONFIG_FILE", "config.json"),
            voicepack_dir=_maybe_str("VOICEPACK_DIR"),
            hf_token=_get("HF_TOKEN"),
            device=_maybe_str("DEVICE"),
            preload=_flag("PRELOAD", True),
            warmup=_flag("WARMUP", True),
            default_voice=_get("DEFAULT_VOICE", DEFAULT_VOICE),
            speed=_maybe_float("SPEED", 1.0) or 1.0,
            crossfade_ms=_maybe_int("CROSSFADE_MS", 50) or 0,
            normalize_peak=_maybe_float("NORMALIZE_PEAK"),
            max_text_chars=_maybe_int("MAX_TEXT_CHARS", 20000) or 0,
            max_concurrency=_maybe_int("MAX_CONCURRENCY", 1) or 1,
            max_queue=_maybe_int("MAX_QUEUE", 256) or 0,
            queue_timeout=_maybe_float("QUEUE_TIMEOUT", 60.0) or 60.0,
            synthesis_timeout=_maybe_float("SYNTHESIS_TIMEOUT", 300.0) or 0.0,
            cache_max_entries=_maybe_int("CACHE_MAX_ENTRIES", 256) or 0,
            cache_max_bytes=_maybe_int("CACHE_MAX_BYTES", 256 * 1024 * 1024) or 0,
            host=_get("HOST", "0.0.0.0"),
            port=_maybe_int("PORT", 8880) or 8880,
            api_key=_get("API_KEY"),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
            log_level=(_get("LOG_LEVEL", "info") or "info").lower(),
            log_json=_flag("LOG_JSON", False),
        )

    def merge(self, **overrides: Any) -> Settings:
        """Return a copy with the non-``None`` overrides applied."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        known = {f.name for f in fields(self)}
        unknown = set(clean) - known
        if unknown:
            raise TypeError(f"unknown setting(s): {sorted(unknown)}")
        return replace(self, **clean)
