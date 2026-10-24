"""Audio encoding.

The engine hands back float32 mono samples at 24 kHz
(:data:`after <kokoro_vietnamese.SAMPLE_RATE>`). Everything here turns those
into bytes: a container the client asked for, or the pieces a streaming
response is assembled from.

``wav`` and ``flac`` are written by libsndfile through ``soundfile``; the
compressed formats go through ``ffmpeg`` reading raw PCM from a pipe. ``pcm``
is passed through untouched, which is why it needs no encoder at all.
"""

from __future__ import annotations

import base64
import io
import shutil
import struct
import subprocess
from collections.abc import Iterable, Iterator

import numpy as np

#: ``Content-Type`` per format. ``pcm`` is deliberately ``audio/pcm``: OpenAI
#: returns the same headerless S16LE bytes for that value.
FORMAT_MIMES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

#: Extra ``ffmpeg`` output flags. ``mp3`` and ``flac`` need nothing beyond
#: ``-f`` because the defaults are already what we want; ``aac`` is muxed as
#: ADTS rather than MP4 so it can be played before the stream ends.
_FFMPEG_OUTPUT_ARGS: dict[str, list[str]] = {
    "mp3": ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
    "aac": ["-c:a", "aac", "-b:a", "128k", "-f", "adts"],
    "opus": ["-c:a", "libopus", "-b:a", "64k", "-f", "ogg"],
}

#: Containers ``ffmpeg`` writes; the rest are handled in-process.
_FFMPEG_FORMATS = frozenset(_FFMPEG_OUTPUT_ARGS)
_SOUNDFILE_FORMATS = frozenset({"wav", "flac"})

#: Full-scale value for the float32 -> int16 conversion below.
_INT16_MAX = 32767.0

class AudioError(RuntimeError):
    """Transcoding failed, usually because ``ffmpeg`` is missing."""

def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None

def to_pcm16le(samples: np.ndarray) -> bytes:
    """Convert float32 samples in ``[-1, 1]`` to signed 16-bit little-endian.

    Out-of-range values are clamped rather than wrapped: wrapping turns a
    slightly too-loud sample into a loud click.
    """
    if samples.size == 0:
        return b""
    clipped = np.clip(samples, -1.0, 1.0) * _INT16_MAX
    ints = np.rint(clipped).astype("<i2")
    return ints.tobytes()

def pcm16le_to_float32(data: bytes) -> np.ndarray:
    """Inverse of :func:`to_pcm16le`, used when an encoder hands audio back."""
    if not data:
        return np.zeros(0, dtype=np.float32)
    ints = np.frombuffer(data, dtype="<i2")
    return (ints.astype(np.float32) / _INT16_MAX).astype(np.float32)

def _write_container(fmt: str, samples: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    # soundfile infers the container from the format string, so the filename is
    # only there to make error messages readable.
    sf.write(buffer, samples, sample_rate, format=fmt.upper(), subtype="PCM_16")
    return buffer.getvalue()

def _encode_with_ffmpeg(
    fmt: str, samples: np.ndarray, sample_rate: int, *, timeout: float = 600.0
) -> bytes:
    if not ffmpeg_available():
        raise AudioError(
            f"response_format={fmt!r} needs ffmpeg, which is not on PATH; "
            "use wav, flac or pcm, or install ffmpeg"
        )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        *_FFMPEG_OUTPUT_ARGS[fmt],
        "pipe:1",
    ]
    try:
        finished = subprocess.run(
            command,
            input=to_pcm16le(samples),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - hostile input
        raise AudioError(f"ffmpeg timed out after {timeout:.0f}s") from exc
    if finished.returncode != 0:
        detail = finished.stderr.decode("utf-8", "replace").strip()
        raise AudioError(f"ffmpeg failed for {fmt}: {detail or 'unknown error'}")
    return finished.stdout

def encode(samples: np.ndarray, fmt: str, sample_rate: int) -> bytes:
    """Encode ``samples`` as a complete file in ``fmt``."""
    fmt = fmt.lower()
    if fmt == "pcm":
        return to_pcm16le(samples)
    if fmt in _SOUNDFILE_FORMATS:
        return _write_container(fmt, samples, sample_rate)
    if fmt in _FFMPEG_FORMATS:
        return _encode_with_ffmpeg(fmt, samples, sample_rate)
    raise AudioError(f"unsupported response_format {fmt!r}")

def streaming_mime(fmt: str) -> str:
    return FORMAT_MIMES.get(fmt.lower(), "application/octet-stream")

def wav_streaming_header(sample_rate: int, channels: int = 1) -> bytes:
    """A WAV header with ``0xFFFFFFFF`` sizes, for streams of unknown length.

    A player that trusts the length would stop immediately, so the sentinel
    tells it to keep reading until the connection closes.
    """
    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    unknown = 0xFFFFFFFF
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", unknown),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits),
            b"data",
            struct.pack("<I", unknown),
        )
    )

def stream_prefix(fmt: str, sample_rate: int, channels: int = 1) -> bytes:
    """Bytes that must precede the first delta for ``fmt``, if any.

    Raw ``pcm`` is deliberately headerless, and the ffmpeg containers are
    self-describing once a decoder sees enough of them, so only WAV needs one.
    """
    fmt = fmt.lower()
    if fmt == "wav":
        return wav_streaming_header(sample_rate, channels)
    return b""

def encode_stream_chunk(
    samples: np.ndarray, fmt: str, sample_rate: int, *, flush: bool = False
) -> bytes:
    """Encode one streaming delta.

    Only the headerless formats are handled here. Re-encoding MP3 or Opus per
    delta would reset the encoder and add an audible artefact at every chunk
    boundary, so callers should stream those as WAV or PCM instead — see
    :func:`stream_prefix` and the note in the README.
    """
    fmt = fmt.lower()
    if fmt == "pcm":
        return to_pcm16le(samples)
    if fmt == "wav":
        return to_pcm16le(samples)
    raise AudioError(
        f"streaming is not supported for response_format={fmt!r}; "
        "use wav or pcm, or set stream=false"
    )

def iter_frames(samples: np.ndarray, frame_samples: int) -> Iterator[np.ndarray]:
    """Yield consecutive slices of ``samples``, each at most ``frame_samples``."""
    if frame_samples <= 0:
        raise ValueError("frame_samples must be positive")
    for start in range(0, len(samples), frame_samples):
        yield samples[start : start + frame_samples]

def concat(chunks: Iterable[np.ndarray]) -> np.ndarray:
    """Join chunks into one array, tolerating an empty iterable."""
    parts = [c for c in chunks if c.size]
    if not parts:
        return np.zeros(0, dtype=np.float32)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)

def duration_seconds(samples: np.ndarray, sample_rate: int) -> float:
    if not sample_rate:
        return 0.0
    return float(len(samples)) / float(sample_rate)

def crossfade(a: np.ndarray, b: np.ndarray, fade_samples: int) -> np.ndarray:
    """Overlap the tail of ``a`` with the head of ``b`` over ``fade_samples``.

    A linear ramp is enough here: adjacent chunks are the same speaker reading
    consecutive sentences, so there is no pitch or phase to preserve and the
    ramp's mild comb filtering is inaudible at speech bandwidth.
    """
    if fade_samples <= 0 or a.size == 0 or b.size == 0:
        return np.concatenate((a, b))
    fade = min(fade_samples, a.size, b.size)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    overlap = a[-fade:] * (1.0 - ramp) + b[:fade] * ramp
    return np.concatenate((a[:-fade], overlap, b[fade:]))

def base64_chunk(data: bytes) -> str:
    """Base64 for an SSE ``speech.audio.delta`` payload."""
    return base64.b64encode(data).decode("ascii")
