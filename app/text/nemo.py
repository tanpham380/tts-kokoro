"""Optional NeMo-text-processing adapter for Vietnamese text normalization.

NeMo's Vietnamese TN grammar covers number, date, time, money, measure,
fraction, ordinal, roman and range in one compiled FST. It is more complete
than the regex passes in this package, so it is used as a second stage when it
is installed.

Two things make it optional rather than required:

* ``pynini`` (OpenFst) ships **manylinux-only** wheels. It installs in the
  Docker images (glibc 2.35/2.36) but not on Windows, so a developer on Windows
  must still be able to run the test suite.
* Compiling the grammar takes seconds on first use, so it is built lazily and
  cached to a ``.far`` file.

The regex passes stay in front of it because NeMo does not know this project's
decisions: it reads ``TP.HCM`` as ``tê pê hát xê em`` and ``khoa Y`` as
``khoa y``. Those are handled before NeMo ever sees the text.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger("tts_kokoro.text.nemo")

#: Where the compiled grammar is cached. Overridable so a container can point
#: it at a writable volume.
CACHE_ENV = "NEMO_TN_CACHE_DIR"
_DEFAULT_CACHE = Path(os.environ.get(CACHE_ENV, "/tmp/nemo_tn_cache"))

_lock = threading.Lock()
_normalizer: object | None = None
_unavailable: str | None = None

def _build() -> object | None:
    """Compile (or restore) the Vietnamese TN grammar. Returns ``None`` on failure."""
    global _unavailable
    try:
        from nemo_text_processing.text_normalization.normalize import Normalizer
    except Exception as exc:  # pragma: no cover - depends on the host
        _unavailable = f"{type(exc).__name__}: {exc}"
        logger.info("NeMo TN unavailable, using regex passes only (%s)", _unavailable)
        return None

    try:
        _DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
        return Normalizer(
            input_case="cased",
            lang="vi",
            cache_dir=str(_DEFAULT_CACHE),
            overwrite_cache=False,
            post_process=True,
        )
    except Exception as exc:  # pragma: no cover - depends on the host
        _unavailable = f"{type(exc).__name__}: {exc}"
        logger.warning("NeMo TN failed to build, using regex passes only (%s)", _unavailable)
        return None

def get_normalizer() -> object | None:
    """Return the shared normalizer, building it on first call.

    Thread-safe and idempotent. Returns ``None`` when NeMo is not installed or
    the grammar could not be compiled; callers must fall back to the regex
    passes in that case.
    """
    global _normalizer
    if _normalizer is not None or _unavailable is not None:
        return _normalizer
    with _lock:
        if _normalizer is None and _unavailable is None:
            _normalizer = _build()
    return _normalizer

def is_available() -> bool:
    """Whether NeMo TN can be used on this host."""
    return get_normalizer() is not None

def unavailable_reason() -> str | None:
    """Why NeMo TN is unavailable, or ``None`` when it works."""
    get_normalizer()
    return _unavailable

def normalize(text: str) -> str:
    """Normalize ``text`` with NeMo TN, or return it unchanged when unavailable."""
    if not text:
        return ""
    normalizer = get_normalizer()
    if normalizer is None:
        return text
    try:
        result = normalizer.normalize(text)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - hostile input
        logger.debug("NeMo TN failed on %r", text[:80], exc_info=True)
        return text
    return result if isinstance(result, str) and result.strip() else text
