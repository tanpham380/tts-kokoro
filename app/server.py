"""OpenAI-compatible Vietnamese TTS server.

Endpoints
---------
``POST /v1/audio/speech``
    The OpenAI TTS endpoint. Returns binary audio by default; streams when
    ``stream_format`` is ``sse`` or ``audio``. Supports voice, speed and
    volume control.
``GET  /v1/models``
    Model listing in OpenAI format so clients that query models first succeed.
``GET  /v1/voices``
    The fourteen voicepacks, plus the OpenAI voice aliases.
``GET  /health``, ``GET  /ready``, ``GET  /live``
    Liveness / readiness probes.
``GET  /metrics``
    Counters and a real-time factor.
``GET  /``
    A small playground for manual synthesis.

The engine loads ``kokoro-vietnamese`` lazily by default (fast container boot
on CPU-only hosts), and sets ``PRELOAD=true`` to load eagerly (better for GPU
splits where the first request should not pay for the download). Override
``DEFAULT_VOICE`` to change the default among the 14 voicepacks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .audio import (
    base64_chunk,
    encode as encode_audio,
    encode_stream_chunk,
    stream_prefix,
    streaming_mime,
)
from .config import DEFAULT_RESPONSE_FORMAT, Settings
from .engine import TTSEngine
from .schemas import APIError, HealthResponse, ModelCard, ModelList, SpeechRequest, VoiceList, VoiceCard
from .voices import OPENAI_VOICE_ALIASES, VOICES, catalogue, resolve

# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #


_log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_str, logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("kokoro_vi_server")

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


settings = Settings.from_env()

# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #


engine = TTSEngine(settings)


# --------------------------------------------------------------------------- #
# FastAPI app setup
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the model in the background when ``PRELOAD=true`` so the app starts
    immediately but is ready shortly after."""
    if settings.preload:
        thread = engine.load_in_background()
        # Give the thread a tiny head-start so /ready reports "loading" instead
        # of falling through to a synchronous first-request load.
        thread.join(timeout=0.2)
    yield


app = FastAPI(
    title="Kokoro-Vietnamese TTS Server",
    description=(
        "OpenAI-compatible text-to-speech API for Vietnamese, powered by "
        "Kokoro-Vietnamese. Accepts /v1/audio/speech requests and streams audio "
        "in SSE or raw-byte format."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# error handling
# --------------------------------------------------------------------------- #


@app.exception_handler(APIError)
async def _api_error_handler(_: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.payload())


@app.exception_handler(StarletteHTTPException)
async def _http_error_handler(
    _: Request, exc: StarletteHTTPException
) -> JSONResponse:
    error = APIError(
        str(exc.detail),
        status_code=exc.status_code,
        code="not_found" if exc.status_code == 404 else None,
    )
    return JSONResponse(status_code=exc.status_code, content=error.payload())


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    error = APIError(f"invalid request: {exc.errors()}", code="invalid_request_body")
    return JSONResponse(status_code=400, content=error.payload())


def require_api_key(request: Request) -> None:
    """Enforce ``API_KEY`` when configured; open server otherwise."""
    if not settings.api_key:
        return
    header = request.headers.get("authorization", "")
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        token = request.headers.get("x-api-key", "").strip()
    if token != settings.api_key:
        raise APIError(
            "invalid API key",
            status_code=401,
            code="invalid_api_key",
            error_type="authentication_error",
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _model_payload() -> dict[str, Any]:
    return ModelCard(
        id=settings.served_model_name,
        created=int(engine._loaded_at or time.time()),
        voices=len(VOICES),
    ).model_dump()


def _check_model(requested: str | None) -> None:
    """Reject a ``model`` value that is not this deployment."""
    if not requested or not requested.strip():
        return
    accepted = {settings.served_model_name.lower(), settings.model_path.lower()}
    # Accept the basename of a local model directory.
    accepted.add(
        settings.model_path.replace("\\", "/").rstrip("/").split("/")[-1].lower()
    )
    # Common OpenAI TTS model names.
    accepted |= {"tts-1", "tts-1-hd"}
    if requested.strip().lower() not in accepted:
        raise APIError(
            f"the model '{requested}' does not exist; this server serves "
            f"'{settings.served_model_name}'",
            status_code=404,
            code="model_not_found",
            param="model",
        )


def _sse(payload: dict[str, Any]) -> str:
    """Serialize one SSE event."""
    return f"data: {json.dumps(payload)}\n\n"


def _audio_delta(data: bytes) -> str:
    """Build an OpenAI-style ``speech.audio.delta`` event."""
    return _sse(
        {
            "type": "speech.audio.delta",
            "audio": base64_chunk(data),
        }
    )


# --------------------------------------------------------------------------- #
# routes: health / ready / metrics
# --------------------------------------------------------------------------- #


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe: whether the model is loaded."""
    return HealthResponse(
        status="ok" if engine.is_ready else ("error" if engine.load_error else "loading"),
        model=settings.served_model_name,
        loaded=engine.is_loaded,
        loading=engine.is_loading,
        device=engine.device,
        voices=len(VOICES),
        version=__version__,
        error=engine.load_error,
    )


@app.get("/ready", include_in_schema=False)
async def ready_probe(response: Response) -> JSONResponse:
    """Readiness probe: 200 when synth is ready, 503 while loading."""
    if not engine.is_ready:
        status_code = 503
        body: dict[str, Any] = {
            "ready": False,
            "status": "error" if engine.load_error else "loading",
        }
        if engine.load_error:
            body["error"] = engine.load_error
    else:
        status_code = 200
        body = {
            "ready": True,
            "status": "ok",
            "model": settings.served_model_name,
            "device": engine.device,
        }
    return JSONResponse(status_code=status_code, content=body)


@app.get("/live", include_in_schema=False)
async def live_probe() -> dict[str, Any]:
    """Liveness probe: always 200 while the process is alive."""
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    base = {**engine.metrics(), **engine.info()}
    return base


# --------------------------------------------------------------------------- #
# routes: OpenAI-compatible
# --------------------------------------------------------------------------- #


@app.get("/v1/models", response_model=ModelList)
async def list_models(
    _auth: None = Depends(require_api_key),
) -> ModelList:
    """The active model, OpenAI format."""
    return ModelList(data=[_model_payload()])


@app.get("/v1/voices", response_model=VoiceList)
@app.get("/v1/audio/voices", response_model=VoiceList)
async def list_voices(
    _auth: None = Depends(require_api_key),
) -> VoiceList:
    """All loaded voicepacks, plus OpenAI voice aliases."""
    entries = catalogue()
    return VoiceList(data=[VoiceCard(**v) for v in entries])


@app.post("/v1/audio/speech")
async def create_speech(
    req: SpeechRequest,
    _auth: None = Depends(require_api_key),
) -> Response:
    """
    Drop-in replacement for OpenAI's ``POST /v1/audio/speech``.

    Parameters
    ----------
    model:
        Model identifier. This server accepts any of ``tts-v1``, ``kokoro``,
        ``Kokoro-Vietnamese``, or the served model name.
    input:
        The text to synthesize (required). Up to ``MAX_TEXT_CHARS`` characters.
    voice:
        One of the fourteen Vietnamese voicepacks, or an OpenAI voice alias
        (``alloy``, ``onyx``, ``nova``, ...). Overrides ``DEFAULT_VOICE``.
    response_format:
        ``mp3`` (default), ``wav``, ``flac``, ``aac``, ``opus``, or ``pcm``.
    speed:
        Speech speed, 0.25 to 4.0 (default 1.0).
    stream_format:
        ``sse`` (default when omitted) streams ``speech.audio.delta`` events.
        ``audio`` streams raw bytes. Returns the full response when omitted.
    stream:
        Standard OpenAI boolean flag. When true, defaults to raw audio streaming.
    volume_multiplier:
        Output gain, 0.1 to 2.0 (default 1.0).
    """
    _check_model(req.model)
    req.check_limits(settings)

    try:
        resolve(req.voice, settings.default_voice)  # validate early
    except Exception as exc:
        raise APIError(
            str(exc), status_code=400, code="invalid_voice", param="voice"
        ) from exc

    fmt = req.response_format
    stream_format = req.stream_format
    if not stream_format and req.stream is True:
        stream_format = "audio"
    streaming = stream_format in ("sse", "audio")

    logger.info(
        "synthesize %d chars | voice=%s speed=%.2f format=%s stream=%s",
        len(req.input),
        req.voice,
        req.speed,
        fmt,
        stream_format or "false",
    )

    # ------------------------------------------------------------- streaming
    if streaming:
        if fmt in ("mp3", "aac", "opus"):
            # Per-chunk ffmpeg encoders reset and add seams; streaming these
            # containers would produce garbled audio. We force WAV or PCM.
            logger.warning(
                "stream_format=%s with response_format=%s forced to wav",
                stream_format,
                fmt,
            )
            fmt = "wav"
        media = streaming_mime(fmt)
        suffix = "" if fmt == "pcm" else stream_prefix(fmt, engine.sample_rate)

        if stream_format == "sse":
            generator = _stream_sse(
                req.input,
                voice=req.voice,
                speed=req.speed,
                fmt=req.response_format,
                volume_multiplier=req.volume_multiplier,
                prefix=suffix,
            )
            return StreamingResponse(
                generator,
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "X-Content-Type-Options": "nosniff",
                },
            )
        else:  # stream_format == "audio"
            generator = _stream_audio(
                req.input,
                voice=req.voice,
                speed=req.speed,
                fmt=fmt,
                volume_multiplier=req.volume_multiplier,
                prefix=suffix,
            )
            return StreamingResponse(
                generator,
                media_type=media,
                headers={
                    "Cache-Control": "no-cache",
                    "Content-Type": media,
                },
            )

    # ------------------------------------------------------------ batch path
    def _synth() -> bytes:
        audio = engine.synthesize(
            req.input,
            voice=req.voice,
            speed=req.speed,
            volume_multiplier=req.volume_multiplier,
        )
        return encode_audio(audio, req.response_format, engine.sample_rate)

    try:
        audio_bytes = await run_in_threadpool(_synth)
    except APIError:
        raise
    except Exception as exc:
        logger.exception("synthesis failed")
        raise APIError(
            f"synthesis failed: {exc}", status_code=500
        ) from exc

    return Response(
        content=audio_bytes,
        media_type=streaming_mime(req.response_format),
    )


# --------------------------------------------------------------------------- #
# streaming helpers
# --------------------------------------------------------------------------- #


async def _stream_sse(
    text: str,
    *,
    voice: str | None,
    speed: float | None,
    fmt: str,
    volume_multiplier: float,
    prefix: bytes,
) -> AsyncIterator[str]:
    """Generator for ``stream_format=sse``.

    Synthesis runs in a thread-pool worker; the audio is encoded and emitted
    as soon as each sentence finishes so playback can start before the full
    text has been synthesized.
    """

    async for chunk, flush_flag in _threaded_stream(
        text, voice=voice, speed=speed, fmt=fmt,
        volume_multiplier=volume_multiplier,
    ):
        if prefix:
            delta = prefix + chunk
            prefix = b""
        else:
            delta = chunk
        yield _audio_delta(delta)
        if flush_flag:
            break

    yield _sse({"type": "speech.audio.done"})
    yield "data: [DONE]\n\n"


async def _stream_audio(
    text: str,
    *,
    voice: str | None,
    speed: float | None,
    fmt: str,
    volume_multiplier: float,
    prefix: bytes,
) -> AsyncIterator[bytes]:
    """Generator for ``stream_format=audio`` (raw-byte streaming)."""
    first = True
    async for chunk, flush_flag in _threaded_stream(
        text, voice=voice, speed=speed, fmt=fmt,
        volume_multiplier=volume_multiplier,
    ):
        data = (prefix + chunk) if first and prefix else chunk
        first = False
        yield data
        if flush_flag:
            break


async def _threaded_stream(
    text: str,
    *,
    voice: str | None,
    speed: float | None,
    fmt: str,
    volume_multiplier: float,
) -> AsyncIterator[tuple[bytes, bool]]:
    """Pull numpy chunks from the engine and encode them to bytes.

    The encoded bytes are yielded along with a completion flag that is True for
    the final chunk, so SSE and audio stream generators know when to close.
    """

    def _produce() -> Generator[tuple[bytes, bool], None, None]:
        try:
            chunks = list(
                engine.stream(
                    text,
                    voice=voice,
                    speed=speed,
                    volume_multiplier=volume_multiplier,
                )
            )
        except APIError:
            # Voice / quota errors surface on the first yielded frame in the
            # batched path, before the lock is acquired. Convert to 503.
            raise
        except Exception as exc:
            raise APIError(f"synthesis failed: {exc}", status_code=500) from exc

        for i, np_chunk in enumerate(chunks):
            try:
                data = encode_stream_chunk(np_chunk, fmt, engine.sample_rate)
            except Exception as exc:
                raise APIError(f"encoding failed: {exc}", status_code=500) from exc
            yield (data, i == len(chunks) - 1)

    loop = asyncio.get_running_loop()
    producer = _produce()
    while True:
        try:
            item = await loop.run_in_executor(None, next, producer)
        except StopIteration:
            return
        yield item  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# playground
# --------------------------------------------------------------------------- #


_HTML = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Kokoro-Vietnamese TTS Studio</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #090b10;
  --bg-gradient: radial-gradient(circle at 50% 0%, rgba(56, 114, 255, 0.12), transparent 50%), radial-gradient(circle at 100% 100%, rgba(138, 43, 226, 0.08), transparent 40%);
  --panel: rgba(18, 22, 33, 0.75);
  --panel-border: rgba(255, 255, 255, 0.08);
  --panel-hover: rgba(25, 31, 46, 0.85);
  --accent: #3b82f6;
  --accent-glow: rgba(59, 130, 246, 0.35);
  --accent-hover: #2563eb;
  --text: #f3f4f6;
  --text-muted: #94a3b8;
  --badge-bg: rgba(59, 130, 246, 0.12);
  --badge-text: #60a5fa;
  --success: #10b981;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  background-image: var(--bg-gradient);
  background-attachment: fixed;
  color: var(--text);
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
  min-height: 100vh;
  padding: 2.5rem 1.5rem;
  display: flex;
  justify-content: center;
}
.container {
  width: 100%;
  max-width: 900px;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}
header {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}
.header-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 1rem;
}
h1 {
  font-family: 'Outfit', sans-serif;
  font-size: 1.85rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #ffffff 40%, #93c5fd 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.35rem 0.85rem;
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 9999px;
  font-size: 0.8rem;
  font-weight: 500;
  backdrop-filter: blur(8px);
}
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 10px var(--success);
}
.sub {
  color: var(--text-muted);
  font-size: 0.95rem;
  line-height: 1.5;
}
.sub code {
  color: #93c5fd;
  background: rgba(59, 130, 246, 0.1);
  padding: 0.15rem 0.4rem;
  border-radius: 4px;
  font-size: 0.88em;
}
.card {
  background: var(--panel);
  border: 1px solid var(--panel-border);
  border-radius: 16px;
  padding: 1.5rem;
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.36);
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
}
.label-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
label {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
textarea {
  width: 100%;
  min-height: 110px;
  padding: 0.9rem;
  background: rgba(10, 13, 20, 0.7);
  color: var(--text);
  border: 1px solid var(--panel-border);
  border-radius: 12px;
  font-family: inherit;
  font-size: 1rem;
  line-height: 1.6;
  resize: vertical;
  transition: border-color 0.2s, box-shadow 0.2s;
}
textarea:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-glow);
}
.sample-chips {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.chip {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid var(--panel-border);
  color: var(--text-muted);
  padding: 0.3rem 0.75rem;
  border-radius: 8px;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s;
}
.chip:hover {
  background: rgba(59, 130, 246, 0.15);
  color: #93c5fd;
  border-color: rgba(59, 130, 246, 0.3);
}
.controls-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
}
.control-item {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}
select, input[type=range] {
  width: 100%;
  accent-color: var(--accent);
}
select {
  padding: 0.65rem 0.85rem;
  background: rgba(10, 13, 20, 0.7);
  color: var(--text);
  border: 1px solid var(--panel-border);
  border-radius: 10px;
  font-size: 0.9rem;
  cursor: pointer;
  outline: none;
  transition: border-color 0.2s;
}
select:focus {
  border-color: var(--accent);
}
.range-header {
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
}
.btn-generate {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 0.9rem 1.5rem;
  font-size: 1.05rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.6rem;
  box-shadow: 0 4px 20px var(--accent-glow);
  transition: transform 0.15s, opacity 0.15s, box-shadow 0.2s;
}
.btn-generate:hover:not(:disabled) {
  opacity: 0.95;
  transform: translateY(-1px);
  box-shadow: 0 6px 24px var(--accent-glow);
}
.btn-generate:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.player-card {
  display: flex;
  flex-direction: column;
  gap: 0.8rem;
}
audio {
  width: 100%;
  border-radius: 10px;
  outline: none;
}
.metrics-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.82rem;
  color: var(--text-muted);
  padding: 0 0.25rem;
}
.badge-tag {
  background: var(--badge-bg);
  color: var(--badge-text);
  padding: 0.2rem 0.5rem;
  border-radius: 6px;
  font-weight: 500;
}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="header-row">
      <h1>Kokoro-Vietnamese TTS</h1>
      <div class="status-pill">
        <span class="status-dot"></span>
        <span id="sys-status">Model Ready</span>
        <span class="badge-tag" id="sys-device">CPU</span>
      </div>
    </div>
    <p class="sub">OpenAI-compatible Vietnamese Text-to-Speech engine. Use the studio below or call <code>POST /v1/audio/speech</code>.</p>
  </header>

  <div class="card">
    <div class="label-row">
      <label for="text">Văn bản tiếng Việt</label>
      <span style="font-size:0.8rem; color:var(--text-muted);"><span id="char-count">0</span> ký tự</span>
    </div>
    <textarea id="text" placeholder="Nhập văn bản tiếng Việt để tổng hợp...">Xin chào! Tôi là trợ lý giọng nói tiếng Việt, được tạo ra từ mô hình Kokoro Vietnamese.</textarea>

    <div class="sample-chips">
      <span class="chip" onclick="setSample('Xin chào quý khách, cảm ơn quý khách đã liên hệ với trung tâm hỗ trợ khách hàng.')">Chào khách hàng</span>
      <span class="chip" onclick="setSample('Dự báo thời tiết hôm nay tại Hà Nội và Thành phố Hồ Chí Minh có mưa rào vài nơi, chiều tối trời mát.')">Thời tiết</span>
      <span class="chip" onclick="setSample('Trí tuệ nhân tạo đang giúp nâng cao hiệu suất làm việc và giải quyết những thách thức lớn của tương lai.')">Công nghệ AI</span>
    </div>

    <div class="controls-grid">
      <div class="control-item">
        <label for="voice">Giọng đọc (Voice)</label>
        <select id="voice"></select>
      </div>
      <div class="control-item">
        <label for="format">Định dạng (Format)</label>
        <select id="format">
          <option value="mp3" selected>mp3 (khuyên dùng)</option>
          <option value="wav">wav</option>
          <option value="opus">opus</option>
          <option value="aac">aac</option>
          <option value="flac">flac</option>
        </select>
      </div>
      <div class="control-item">
        <div class="range-header">
          <label for="speed">Tốc độ (Speed)</label>
          <span id="speed-val" style="color:#60a5fa; font-weight:600;">1.00x</span>
        </div>
        <input id="speed" type="range" min="0.5" max="2.0" step="0.05" value="1.0">
      </div>
      <div class="control-item">
        <label for="stream">Truyền phát (Stream)</label>
        <select id="stream">
          <option value="false" selected>Tải toàn bộ (Standard)</option>
          <option value="audio">Audio chunked</option>
          <option value="sse">OpenAI SSE</option>
        </select>
      </div>
    </div>

    <button class="btn-generate" id="generate" onclick="generateSpeech()">
      <span id="btn-icon">▶</span>
      <span id="btn-text">Tổng Hợp Giọng Nói</span>
    </button>
  </div>

  <div class="card player-card" id="playback-section" style="display:none;">
    <div class="label-row">
      <label>Kết quả âm thanh</label>
      <div id="stats-badge" class="badge-tag">Sẵn sàng</div>
    </div>
    <audio id="player" controls></audio>
    <div class="metrics-row">
      <span id="metric-latency">Thời gian xử lý: -</span>
      <span id="metric-format">Định dạng: 24kHz Mono</span>
    </div>
  </div>
</div>

<script>
const textEl = document.getElementById('text');
const charCountEl = document.getElementById('char-count');
const voiceEl = document.getElementById('voice');
const formatEl = document.getElementById('format');
const speedEl = document.getElementById('speed');
const speedValEl = document.getElementById('speed-val');
const streamEl = document.getElementById('stream');
const player = document.getElementById('player');
const generateBtn = document.getElementById('generate');
const btnIcon = document.getElementById('btn-icon');
const btnText = document.getElementById('btn-text');
const playbackSection = document.getElementById('playback-section');
const statsBadge = document.getElementById('stats-badge');
const metricLatency = document.getElementById('metric-latency');

function updateCharCount() {
  charCountEl.textContent = textEl.value.length;
}
textEl.addEventListener('input', updateCharCount);
updateCharCount();

speedEl.addEventListener('input', () => {
  speedValEl.textContent = parseFloat(speedEl.value).toFixed(2) + 'x';
});

function setSample(text) {
  textEl.value = text;
  updateCharCount();
}

async function loadHealthAndVoices() {
  try {
    const [hRes, vRes] = await Promise.all([
      fetch('/health'),
      fetch('/v1/voices')
    ]);
    if (hRes.ok) {
      const hData = await hRes.json();
      document.getElementById('sys-status').textContent = hData.loaded ? 'Model Ready' : (hData.loading ? 'Model Loading...' : 'Error');
      document.getElementById('sys-device').textContent = (hData.device || 'CPU').toUpperCase();
    }
    if (vRes.ok) {
      const { data } = await vRes.json();
      voiceEl.innerHTML = data.map(v => {
        const genderMark = v.gender === 'female' ? '♀ Nữ' : (v.gender === 'male' ? '♂ Nam' : 'Trực tiếp');
        return `<option value="${v.id}" ${v.id === 'diem_trinh' ? 'selected' : ''}>${v.name} (${genderMark})</option>`;
      }).join('');
    }
  } catch (err) {
    document.getElementById('sys-status').textContent = 'Server Offline';
  }
}

async function generateSpeech() {
  const text = textEl.value.trim();
  if (!text) {
    alert('Vui lòng nhập văn bản cần đọc.');
    return;
  }

  generateBtn.disabled = true;
  btnIcon.textContent = '⏳';
  btnText.textContent = 'Đang xử lý...';

  const format = formatEl.value;
  const streamMode = streamEl.value;
  const body = {
    model: 'Kokoro-Vietnamese',
    input: text,
    voice: voiceEl.value,
    response_format: format,
    speed: parseFloat(speedEl.value)
  };

  if (streamMode !== 'false') {
    body.stream_format = streamMode;
  }

  const t0 = performance.now();
  try {
    const res = await fetch('/v1/audio/speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: { message: res.statusText } }));
      throw new Error(err.error?.message || 'Lỗi tổng hợp âm thanh');
    }

    const elapsed = Math.round(performance.now() - t0);
    playbackSection.style.display = 'flex';
    metricLatency.textContent = `Thời gian xử lý: ${elapsed}ms`;
    statsBadge.textContent = `${format.toUpperCase()} (${elapsed}ms)`;

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    player.src = url;
    player.play().catch(() => {});
  } catch (err) {
    alert('Lỗi: ' + err.message);
  } finally {
    generateBtn.disabled = false;
    btnIcon.textContent = '▶';
    btnText.textContent = 'Tổng Hợp Giọng Nói';
  }
}

loadHealthAndVoices();
</script>
</body>
</html>"""


@app.get("/", include_in_schema=False)
async def index() -> HTMLResponse:
    """The TTS playground."""
    return HTMLResponse(_HTML)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.server:app",
        host=settings.host,
        port=settings.port,
        log_level=_log_level_str.lower(),
        workers=1,
    )
