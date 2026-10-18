"""Tests for the configuration module."""



from __future__ import annotations

import importlib

import pytest



from app.config import Settings





# --------------------------------------------------------------------------- #

# defaults

# --------------------------------------------------------------------------- #





def test_settings_defaults():

    """Settings.from_env() reads variables from os.environ."""

    s = Settings.from_env()

    assert s.served_model_name == "Kokoro-Vietnamese"

    assert s.default_voice == "diem_trinh"

    assert s.speed == 1.0

    assert s.crossfade_ms == 50

    assert s.normalize_peak is None

    assert s.max_concurrency == 1

    assert s.max_queue == 256

    assert s.device is None

    assert s.preload is True

    assert s.warmup is True

    assert s.log_level == "info"

    assert s.port == 8880





def test_settings_seed(monkeypatch):

    """Values supplied by the caller always win over the default constructor."""

    s = Settings()

    assert s.port == 8880

    assert s.host == "0.0.0.0"

    assert s.api_key is None

    assert s.hf_token is None





def test_profile_auto():

    s = Settings()

    assert s.profile == "auto"





def test_profile_cpu():

    s = Settings(device="cpu")

    assert s.profile == "cpu"





def test_profile_cuda():

    s = Settings(device="cuda")

    assert s.profile == "cuda"





def test_profile_cuda_with_index():

    s = Settings(device="cuda:1")

    assert s.profile == "cuda"





def test_profile_mps():

    s = Settings(device="mps")

    assert s.profile == "mps"





# --------------------------------------------------------------------------- #

# environment

# --------------------------------------------------------------------------- #





def test_settings_from_env(monkeypatch):

    monkeypatch.setenv("DEVICE", "cuda:0")

    monkeypatch.setenv("PORT", "9000")

    monkeypatch.setenv("DEFAULT_VOICE", "mai_linh")

    monkeypatch.setenv("LOG_LEVEL", "debug")

    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")

    monkeypatch.setenv("MAX_CONCURRENCY", "4")

    monkeypatch.setenv("MAX_QUEUE", "128")

    monkeypatch.setenv("PRELOAD", "false")

    monkeypatch.setenv("WARMUP", "false")

    monkeypatch.setenv("NORMALIZE_PEAK", "0.9")

    monkeypatch.setenv("MAX_TEXT_CHARS", "5000")



    s = Settings.from_env()

    assert s.device == "cuda:0"

    assert s.port == 9000

    assert s.default_voice == "mai_linh"

    assert s.log_level == "debug"

    assert s.cors_origins == ("https://a.example", "https://b.example")

    assert s.max_concurrency == 4

    assert s.max_queue == 128

    assert s.preload is False

    assert s.warmup is False

    assert s.normalize_peak == pytest.approx(0.9)

    assert s.max_text_chars == 5000





def test_settings_from_env_preload_bool(monkeypatch):

    monkeypatch.setenv("PRELOAD", "0")

    s = Settings.from_env()

    assert s.preload is False





# --------------------------------------------------------------------------- #

# merge

# --------------------------------------------------------------------------- #





def test_merge_overrides_known():

    s = Settings().merge(device="cpu", port=1234)

    assert s.device == "cpu"

    assert s.port == 1234





def test_merge_rejects_unknown():

    with pytest.raises(TypeError):

        Settings().merge(unknown_field=1)

