"""Tests for the production-hardening features: cache, timeout, draining,
Prometheus exposition and request ids.

No torch, no model, no network — the engine is driven through its public
surface with a fake runtime.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from starlette.testclient import TestClient

from app.cache import AudioCache, CacheKey
from app.config import Settings
from app.engine import TTSEngine, VoiceRuntime
from app.schemas import APIError
from app.server import app, engine

# --------------------------------------------------------------------------- #
# audio cache
# --------------------------------------------------------------------------- #

def _key(text: str = "xin chào", **overrides) -> CacheKey:
    base = {
        "text": text,
        "voice": "diem_trinh",
        "speed": 1.0,
        "fmt": "mp3",
        "sample_rate": 24000,
    }
    base.update(overrides)
    return CacheKey(**base)

def test_cache_roundtrip() -> None:
    cache = AudioCache(max_entries=4, max_bytes=1024)
    assert cache.get(_key()) is None
    cache.put(_key(), b"audio")
    assert cache.get(_key()) == b"audio"

def test_cache_key_covers_every_input() -> None:
    """A different voice, speed, format or rate must not hit the same entry."""
    cache = AudioCache(max_entries=8, max_bytes=4096)
    cache.put(_key(), b"a")
    assert cache.get(_key(voice="mai_linh")) is None
    assert cache.get(_key(speed=1.5)) is None
    assert cache.get(_key(fmt="wav")) is None
    assert cache.get(_key(sample_rate=16000)) is None
    assert cache.get(_key(text="khác")) is None

def test_cache_evicts_least_recently_used() -> None:
    cache = AudioCache(max_entries=2, max_bytes=4096)
    cache.put(_key("a"), b"1")
    cache.put(_key("b"), b"2")
    cache.get(_key("a"))  # "a" is now the most recent
    cache.put(_key("c"), b"3")
    assert cache.get(_key("b")) is None
    assert cache.get(_key("a")) == b"1"
    assert cache.get(_key("c")) == b"3"

def test_cache_respects_byte_budget() -> None:
    cache = AudioCache(max_entries=100, max_bytes=10)
    cache.put(_key("a"), b"12345")
    cache.put(_key("b"), b"67890")
    cache.put(_key("c"), b"abcde")  # pushes the total over 10 bytes
    stats = cache.stats()
    assert stats.bytes <= 10
    assert stats.evictions >= 1

def test_cache_rejects_oversized_value() -> None:
    cache = AudioCache(max_entries=10, max_bytes=4)
    cache.put(_key(), b"12345")
    assert cache.get(_key()) is None

def test_cache_disabled_when_budget_is_zero() -> None:
    cache = AudioCache(max_entries=0, max_bytes=0)
    assert cache.enabled is False
    cache.put(_key(), b"audio")
    assert cache.get(_key()) is None

def test_cache_stats_count_hits_and_misses() -> None:
    cache = AudioCache(max_entries=4, max_bytes=1024)
    cache.get(_key())
    cache.put(_key(), b"audio")
    cache.get(_key())
    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.entries == 1

# --------------------------------------------------------------------------- #
# engine: cache, timeout, draining
# --------------------------------------------------------------------------- #

class _FakeVoicepack:
    """Stands in for a torch voicepack: indexable and sized."""

    def __len__(self) -> int:
        return 600

    def __getitem__(self, index: int) -> np.ndarray:
        return np.zeros(256, dtype=np.float32)

class _FakeModel:
    def __call__(self, phonemes: str, ref_s: object, speed: float) -> _FakeAudio:
        return _FakeAudio()

class _FakeAudio:
    """Mimics the torch tensor the decoder returns."""

    def detach(self) -> _FakeAudio:
        return self

    def cpu(self) -> _FakeAudio:
        return self

    def numpy(self) -> np.ndarray:
        return np.ones(2400, dtype=np.float32)

@pytest.fixture(autouse=True)
def _fake_model_stack(monkeypatch):
    """``VoiceRuntime`` imports the model stack; stub it out.

    The suite must run without torch or a model download, so the three things
    the engine touches are replaced with trivial stand-ins.
    """
    import sys
    from types import ModuleType

    kokoro = ModuleType("kokoro_vietnamese")
    kokoro.phonemize = lambda text: text  # type: ignore[attr-defined]
    kokoro.split_text = lambda text: [text] if text.strip() else []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro_vietnamese", kokoro)

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *exc):
            return False

    torch = ModuleType("torch")
    torch.no_grad = _NoGrad  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    yield

def _engine(settings: Settings) -> TTSEngine:
    engine = TTSEngine(settings)
    engine._model = _FakeModel()
    engine._voices["diem_trinh"] = VoiceRuntime(
        id="diem_trinh",
        voicepack=_FakeVoicepack(),
        model=engine._model,
        device="cpu",
    )
    engine._default_runtime = engine._voices["diem_trinh"]
    engine._device = "cpu"
    return engine

def test_engine_cache_roundtrip() -> None:
    engine = _engine(Settings(cache_max_entries=4, cache_max_bytes=4096))
    key = _key()
    assert engine.cache_get(key) is None
    engine.cache_put(key, b"audio")
    assert engine.cache_get(key) == b"audio"

def test_engine_cache_disabled_by_default_budget() -> None:
    engine = _engine(Settings(cache_max_entries=0, cache_max_bytes=0))
    engine.cache_put(_key(), b"audio")
    assert engine.cache_get(_key()) is None

def test_synthesis_timeout_raises_503(monkeypatch) -> None:
    """A deadline of 0 seconds must trip on the first chunk."""
    engine = _engine(Settings(synthesis_timeout=0.0001, max_queue=0))
    # Make the clock jump past the deadline immediately.
    ticks = iter([0.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(
        "app.engine.time.perf_counter", lambda: next(ticks, 100.0)
    )
    with pytest.raises(APIError) as exc:
        list(engine.stream("xin chào"))
    assert exc.value.status_code == 503
    assert exc.value.code == "synthesis_timeout"

def test_synthesis_timeout_disabled_when_zero() -> None:
    engine = _engine(Settings(synthesis_timeout=0, max_queue=0))
    chunks = list(engine.stream("xin chào"))
    assert chunks

def test_draining_refuses_new_requests() -> None:
    engine = _engine(Settings())
    assert engine.is_draining is False
    engine.begin_shutdown()
    assert engine.is_draining is True
    with pytest.raises(APIError) as exc:
        list(engine.stream("xin chào"))
    assert exc.value.code == "server_shutting_down"

def test_metrics_include_cache_fields() -> None:
    engine = _engine(Settings(cache_max_entries=4, cache_max_bytes=4096))
    metrics = engine.metrics()
    for field in (
        "cache_enabled",
        "cache_hits",
        "cache_misses",
        "cache_evictions",
        "cache_entries",
        "cache_bytes",
    ):
        assert field in metrics

def test_prometheus_exposition_is_well_formed() -> None:
    engine = _engine(Settings())
    text = engine.prometheus()
    assert text.endswith("\n")
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        assert name.startswith("kokoro_"), line
        float(value)  # must parse
    assert "# TYPE kokoro_requests_total counter" in text
    assert "# TYPE kokoro_queue_depth gauge" in text

# --------------------------------------------------------------------------- #
# server: request id, prometheus route, cache header
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _patch_engine():
    original = (engine._model, engine._device, engine._load_error, engine._draining)
    engine._model = object()
    engine._device = "cpu"
    engine._load_error = None
    engine._draining = False
    yield
    engine._model, engine._device, engine._load_error, engine._draining = original

@pytest.fixture
def client() -> TestClient:
    return TestClient(app)

def test_request_id_is_generated(client: TestClient) -> None:
    response = client.get("/health")
    assert response.headers.get("X-Request-ID")

def test_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "trace-42"})
    assert response.headers["X-Request-ID"] == "trace-42"

def test_prometheus_route(client: TestClient) -> None:
    response = client.get("/metrics/prometheus")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "kokoro_requests_total" in response.text

def test_metrics_route_still_json(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["cache_enabled"] in (True, False)

def test_speech_reports_cache_miss_then_hit(client: TestClient, monkeypatch) -> None:
    """The second identical request must be served from the cache."""
    logging.disable(logging.CRITICAL)
    engine.cache_clear()
    monkeypatch.setattr(
        "app.server.encode_audio", lambda audio, fmt, rate: b"encoded-audio"
    )
    # The engine's own runtime is a fake so no model is needed.
    fake = _engine(Settings(cache_max_entries=8, cache_max_bytes=65536))
    monkeypatch.setattr(engine, "_model", fake._model)
    monkeypatch.setattr(engine, "_voices", fake._voices)
    monkeypatch.setattr(engine, "_default_runtime", fake._default_runtime)
    monkeypatch.setattr(engine, "_cache", fake._cache)
    body = {
        "model": "Kokoro-Vietnamese",
        "input": "xin chào",
        "voice": "diem_trinh",
        "response_format": "wav",
    }
    first = client.post("/v1/audio/speech", json=body)
    second = client.post("/v1/audio/speech", json=body)
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.headers.get("X-Cache") == "MISS"
    assert second.headers.get("X-Cache") == "HIT"
    assert second.content == first.content
