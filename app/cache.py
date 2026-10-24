"""A bounded LRU cache for synthesized audio.

Repeated text is common in production — IVR lines, prompts, fixed
announcements — and a hit skips the decode entirely, which is the single
largest latency win available without changing the model.

Two properties matter:

* **Bounded on both axes.** An entry count alone lets a few long inputs blow
  the memory budget, so the byte total is capped too and the least-recently-used
  entry is evicted until both fit.
* **Keyed on everything that changes the audio.** Text, voice, speed, format
  and sample rate all go into the key; omitting any of them would serve the
  wrong bytes.

The cache is process-local and thread-safe. It is deliberately not shared
across workers: a multi-worker deployment would need Redis, and the hit rate
does not justify that for a single-GPU server.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheKey:
    """Everything that changes the synthesized bytes."""

    text: str
    voice: str
    speed: float
    fmt: str
    sample_rate: int

    def digest(self) -> str:
        """A stable short key. ``blake2b`` is faster than SHA-256 and enough here."""
        payload = "\x1f".join(
            (
                self.text,
                self.voice,
                f"{self.speed:.4f}",
                self.fmt,
                str(self.sample_rate),
            )
        )
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    entries: int = 0
    bytes: int = 0

class AudioCache:
    """A thread-safe LRU cache bounded by entry count and total bytes."""

    def __init__(self, max_entries: int = 256, max_bytes: int = 256 * 1024 * 1024):
        self.max_entries = max(0, max_entries)
        self.max_bytes = max(0, max_bytes)
        self._entries: OrderedDict[str, bytes] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0 and self.max_bytes > 0

    def get(self, key: CacheKey) -> bytes | None:
        """Return the cached bytes, or ``None`` on a miss."""
        if not self.enabled:
            return None
        digest = key.digest()
        with self._lock:
            value = self._entries.get(digest)
            if value is None:
                self._misses += 1
                return None
            self._entries.move_to_end(digest)
            self._hits += 1
            return value

    def put(self, key: CacheKey, value: bytes) -> None:
        """Store ``value``, evicting least-recently-used entries as needed."""
        if not self.enabled or not value:
            return
        # A single value larger than the whole budget can never be cached.
        if len(value) > self.max_bytes:
            return
        digest = key.digest()
        with self._lock:
            previous = self._entries.pop(digest, None)
            if previous is not None:
                self._bytes -= len(previous)
            self._entries[digest] = value
            self._bytes += len(value)
            while self._entries and (
                len(self._entries) > self.max_entries or self._bytes > self.max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= len(evicted)
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                entries=len(self._entries),
                bytes=self._bytes,
            )
