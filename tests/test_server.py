"""Tests for FastAPI HTTP endpoints."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from app.config import Settings
from app.server import app, engine


@pytest.fixture(autouse=True)
def _patch_engine_for_tests():
    """Ensure engine reports loaded in HTTP tests without pulling real model."""
    orig_loaded = engine._model
    orig_device = engine._device
    orig_error = engine._load_error

    engine._model = object()
    engine._device = "cpu"
    engine._load_error = None

    yield

    engine._model = orig_loaded
    engine._device = orig_device
    engine._load_error = orig_error


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["loaded"] is True
    assert data["voices"] == 14


def test_ready_and_live_probes(client):
    res_ready = client.get("/ready")
    assert res_ready.status_code == 200
    assert res_ready.json()["ready"] is True

    res_live = client.get("/live")
    assert res_live.status_code == 200
    assert res_live.json()["status"] == "ok"


def test_list_models(client):
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "Kokoro-Vietnamese"


def test_list_voices_and_aliases(client):
    res1 = client.get("/v1/voices")
    assert res1.status_code == 200
    data1 = res1.json()
    assert len(data1["data"]) == 14

    # Compatible alias endpoint /v1/audio/voices
    res2 = client.get("/v1/audio/voices")
    assert res2.status_code == 200
    data2 = res2.json()
    assert len(data2["data"]) == 14


def test_speech_invalid_voice_error_format(client):
    res = client.post(
        "/v1/audio/speech",
        json={
            "model": "Kokoro-Vietnamese",
            "input": "Xin chào",
            "voice": "invalid_voice_name_12345",
        },
    )
    assert res.status_code == 400
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "invalid_voice"
    assert data["error"]["type"] == "invalid_request_error"


def test_speech_empty_input_validation(client):
    res = client.post(
        "/v1/audio/speech",
        json={
            "model": "Kokoro-Vietnamese",
            "input": "   ",
            "voice": "diem_trinh",
        },
    )
    assert res.status_code == 400
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "invalid_request_body"


def test_speech_model_not_found(client):
    res = client.post(
        "/v1/audio/speech",
        json={
            "model": "completely-wrong-model",
            "input": "Xin chào",
            "voice": "diem_trinh",
        },
    )
    assert res.status_code == 404
    data = res.json()
    assert "error" in data
    assert data["error"]["code"] == "model_not_found"


def test_index_playground(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Kokoro-Vietnamese TTS Studio" in res.text
