"""Engine tests. We never import torch or kokoro_vietnamese — a fake runtime
stands in for the real model so the queueing/voice-mix logic can be tested on a
bare laptop."""



from __future__ import annotations

import threading

import numpy as np

import pytest



from app.config import Settings

from app.engine import TTSEngine, VoiceRuntime, _fit_phonemes

from app.schemas import APIError





class FakeVoicepack:

    """Minimal stand-in for a torch.load() voicepack."""



    def __len__(self):

        return 600





class FakeModel:

    """Minimal stand-in for KokoroVietnamese.model."""



    def __call__(self, phonemes: str, ref_s: object, speed: float):

        # Return one half-second of audio so we can verify the duration.

        rate = 24_000

        n = int(rate * 0.5 / max(speed, 0.1))

        return np.ones(n, dtype=np.float32)





def _make_runtime(runtime_id: str) -> VoiceRuntime:

    return VoiceRuntime(

        id=runtime_id,

        voicepack=FakeVoicepack(),

        model=FakeModel(),        device="cpu",

    )





def _patched_engine(settings: Settings, monkeypatch):

    """Build a TTSEngine whose load path is replaced by our fake runtime."""

    engine = TTSEngine(settings)

    # Pretend the model and default voicepack are already in memory.

    engine._model = FakeModel()

    engine._voices["diem_trinh"] = _make_runtime("diem_trinh")

    engine._default_runtime = engine._voices["diem_trinh"]

    engine._device = "cpu"

    engine._is_loading = False

    engine._loaded_at = __import__("time").time()

    return engine





# --------------------------------------------------------------------------- #

# construction / state

# --------------------------------------------------------------------------- #





def test_engine_defaults():

    """Constructing does not preload anything."""

    s = Settings()

    e = TTSEngine(s)

    assert e.is_loaded is False

    assert e.is_ready is False

    assert e.default_voice == "diem_trinh"

    assert e.device is None





def test_voice_runtime_dataclass():

    v = _make_runtime("test")

    assert v.id == "test"

    assert v.device == "cpu"





# --------------------------------------------------------------------------- #

# streaming gate

# --------------------------------------------------------------------------- #





def test_acquire_blocked_when_full(monkeypatch):
    """max_concurrency=1 and max_queue=0 means two requests cannot overlap."""
    s = Settings(max_concurrency=1, max_queue=0)
    e = _patched_engine(s, monkeypatch)
    # Simulate one slot already taken by holding the semaphore.
    e._slots.acquire()
    with pytest.raises(APIError) as exc:
        e._acquire()
    assert exc.value.status_code == 503

def test_release_reopens_slot():
    e = TTSEngine(Settings(max_concurrency=1))
    e._acquire()
    e._release(0, 0.01, failed=False)
    assert e._counters.queued == 0
    assert e._counters.requests == 1

def test_acquire_records_queued():
    s = Settings(max_concurrency=1, max_queue=2)
    e = _patched_engine(s, None)
    e._acquire()
    assert e._counters.queued == 0
    e._release(100, 0.1, failed=False)
    assert e._counters.requests == 1





# --------------------------------------------------------------------------- #

# public surface

# --------------------------------------------------------------------------- #





def test_info_shape():

    e = TTSEngine(Settings())

    info = e.info()

    assert "device" in info

    assert "model" in info

    assert "sample_rate" in info





def test_metrics_shape():

    e = TTSEngine(Settings())

    m = e.metrics()

    assert "requests_total" in m

    assert "audio_seconds_total" in m





def test_synthesize_not_loaded():

    e = TTSEngine(Settings())

    try:

        e.synthesize("test")

    except APIError as exc:
        assert exc.status_code == 503
        assert exc.code in ("model_loading", "model_unavailable")
    else:
        raise AssertionError("should have raised APIError")


def test_require_loaded_empty_queue():
    """When the model is loading / failed, streams should refuse cleanly."""
    e = TTSEngine(Settings())
    for fn in (lambda: next(iter(e.stream("h"))), lambda: e.synthesize("h")):
        with pytest.raises(APIError):
            fn()


def test_engine_load_error_recorded():
    e = TTSEngine(Settings())
    e._load_error = "boom"
    try:
        e.synthesize("test")
    except APIError as exc:
        assert exc.code == "model_unavailable"
        assert "boom" in exc.message

    else:

        raise AssertionError("expected APIError")





# --------------------------------------------------------------------------- #

# counters

# --------------------------------------------------------------------------- #





def test_release_counts_succeeded():

    e = TTSEngine(Settings())

    e._acquire()

    e._release(24000, 1.0, failed=False)
    assert e._counters.succeeded == 1
    assert e._counters.audio_seconds == 1.0

    assert e._counters.synth_seconds == 1.0

    m = e.metrics()

    assert m["requests_total"] == 1

    assert m["audio_seconds_total"] == 1.0


def test_normalize_vietnamese_text_issue_4_fixes():
    from app.engine import normalize_vietnamese_text

    # Issue #4: Single isolated Y was pronounced as "quai"
    assert normalize_vietnamese_text("khoa Y") == "khoa i"
    assert normalize_vietnamese_text("bác sĩ Y") == "bác sĩ i"
    assert normalize_vietnamese_text("nhóm máu A và B") == "nhóm máu a và bê"
    assert normalize_vietnamese_text("vitamin C") == "vitamin xê"
    assert normalize_vietnamese_text("vitamin D") == "vitamin dê"
    # Words with 'y' or 'a' inside must remain intact
    assert normalize_vietnamese_text("yêu quý ngày mai") == "yêu quý ngày mai"


def test_normalize_vietnamese_abbreviations():
    from app.engine import normalize_vietnamese_text

    assert "Thành phố Hồ Chí Minh" in normalize_vietnamese_text("TP.HCM đẹp lắm")
    assert "phần trăm" in normalize_vietnamese_text("tăng trưởng 100%")
    assert "ki lô mét" in normalize_vietnamese_text("khoảng cách 15km")


def test_gpustack_local_directory_mount_resolution(tmp_path, monkeypatch):
    """Test that when GPUStack mounts a model directory, files are resolved locally without HF download."""
    import sys
    from types import ModuleType

    # Create fake model directory mimicking dinhthuan/kokoro-vi-ngoc-huyen
    model_dir = tmp_path / "kokoro-vi-ngoc-huyen"
    model_dir.mkdir()
    (model_dir / "kokoro_vi_ngoc_huyen.pth").write_text("fake-weights")
    (model_dir / "config.json").write_text("{}")
    voices_dir = model_dir / "voices"
    voices_dir.mkdir()
    (voices_dir / "diem_trinh.pt").write_text("fake-voicepack")

    captured_init = {}

    class MockKokoroVietnamese:
        def __init__(self, **kwargs):
            captured_init.update(kwargs)
            self.model = FakeModel()
            self.voicepack = FakeVoicepack()
            self.device = "cpu"

    # Mock module if not present in test environment
    if "kokoro_vietnamese" not in sys.modules:
        mock_mod = ModuleType("kokoro_vietnamese")
        mock_mod.KokoroVietnamese = MockKokoroVietnamese
        mock_mod.resolve_voicepack_filename = lambda v, _: f"{v}.pt"
        monkeypatch.setitem(sys.modules, "kokoro_vietnamese", mock_mod)
    else:
        monkeypatch.setattr("kokoro_vietnamese.KokoroVietnamese", MockKokoroVietnamese)

    engine = TTSEngine(Settings(model_path=str(model_dir), default_voice="diem_trinh"))
    runtime = engine._build_runtime("diem_trinh")

    assert captured_init["model_path"] == str(model_dir / "kokoro_vi_ngoc_huyen.pth")
    assert captured_init["config_path"] == str(model_dir / "config.json")
    assert captured_init["voicepack_path"] == str(voices_dir / "diem_trinh.pt")
    assert runtime.id == "diem_trinh"


def test_normalize_numbers_and_dates():
    from app.engine import normalize_vietnamese_text

    # User test case: "01 05 2001"
    assert normalize_vietnamese_text("01 05 2001") == "không một không năm hai nghìn không trăm lẻ một"
    # Date with slashes
    assert normalize_vietnamese_text("01/05/2001") == "ngày mùng một tháng năm năm hai nghìn không trăm lẻ một"
    # Decimals
    assert normalize_vietnamese_text("3.14") == "ba phẩy một bốn"
    # Regular numbers
    assert normalize_vietnamese_text("năm 2024") == "năm hai nghìn không trăm hai mươi tư"
    assert normalize_vietnamese_text("100k") == "một trăm nghìn"


