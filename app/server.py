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
from .ui import INDEX_HTML
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

    if not engine.is_loaded:
        if engine._load_error:
            raise APIError(
                f"the model failed to load: {engine._load_error}",
                status_code=503,
                code="model_load_failed",
            )
        raise APIError(
            "the model is still loading, please retry in a moment",
            status_code=503,
            code="model_loading",
        )

    try:
        resolved_voice = resolve(req.voice, settings.default_voice)  # validate early
    except Exception as exc:
        raise APIError(
            str(exc), status_code=400, code="invalid_voice", param="voice"
        ) from exc
    req.voice = resolved_voice

    fmt = req.response_format
    stream_format = req.stream_format
    if not stream_format and req.stream is True:
        stream_format = "audio"
    streaming = stream_format in ("sse", "audio")

    logger.info(
        "synthesize %d chars | voice=%s speed=%.2f format=%s stream=%s task=%s lang=%s",
        len(req.input),
        req.voice,
        req.speed,
        fmt,
        stream_format or "false",
        req.task_type or "tts",
        req.language or "vi",
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
                language=req.language,
                task_type=req.task_type,
                instructions=req.instructions,
                reference_audio=req.reference_audio,
                use_speaker_embedding_only=req.use_speaker_embedding_only,
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
                language=req.language,
                task_type=req.task_type,
                instructions=req.instructions,
                reference_audio=req.reference_audio,
                use_speaker_embedding_only=req.use_speaker_embedding_only,
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
            language=req.language,
            task_type=req.task_type,
            instructions=req.instructions,
            reference_audio=req.reference_audio,
            use_speaker_embedding_only=req.use_speaker_embedding_only,
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
    language: str | None = "vi",
    task_type: str | None = None,
    instructions: str | None = None,
    reference_audio: str | None = None,
    use_speaker_embedding_only: bool | None = None,
) -> AsyncIterator[str]:
    """Generator for ``stream_format=sse``.

    Synthesis runs in a thread-pool worker; the audio is encoded and emitted
    as soon as each sentence finishes so playback can start before the full
    text has been synthesized.
    """

    try:
        async for chunk, flush_flag in _threaded_stream(
            text,
            voice=voice,
            speed=speed,
            fmt=fmt,
            volume_multiplier=volume_multiplier,
            language=language,
            task_type=task_type,
            instructions=instructions,
            reference_audio=reference_audio,
            use_speaker_embedding_only=use_speaker_embedding_only,
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
    except (asyncio.CancelledError, GeneratorExit):
        return


async def _stream_audio(
    text: str,
    *,
    voice: str | None,
    speed: float | None,
    fmt: str,
    volume_multiplier: float,
    prefix: bytes,
    language: str | None = "vi",
    task_type: str | None = None,
    instructions: str | None = None,
    reference_audio: str | None = None,
    use_speaker_embedding_only: bool | None = None,
) -> AsyncIterator[bytes]:
    """Generator for ``stream_format=audio`` (raw-byte streaming)."""
    first = True
    try:
        async for chunk, flush_flag in _threaded_stream(
            text,
            voice=voice,
            speed=speed,
            fmt=fmt,
            volume_multiplier=volume_multiplier,
            language=language,
            task_type=task_type,
            instructions=instructions,
            reference_audio=reference_audio,
            use_speaker_embedding_only=use_speaker_embedding_only,
        ):
            data = (prefix + chunk) if first and prefix else chunk
            first = False
            yield data
            if flush_flag:
                break
    except (asyncio.CancelledError, GeneratorExit):
        return


async def _threaded_stream(
    text: str,
    *,
    voice: str | None,
    speed: float | None,
    fmt: str,
    volume_multiplier: float,
    language: str | None = "vi",
    task_type: str | None = None,
    instructions: str | None = None,
    reference_audio: str | None = None,
    use_speaker_embedding_only: bool | None = None,
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
                    language=language,
                    task_type=task_type,
                    instructions=instructions,
                    reference_audio=reference_audio,
                    use_speaker_embedding_only=use_speaker_embedding_only,
                )
            )
        except Exception as exc:
            logger.warning("Streaming producer synthesis error: %s", exc)
            return

        for i, np_chunk in enumerate(chunks):
            try:
                data = encode_stream_chunk(np_chunk, fmt, engine.sample_rate)
            except Exception as exc:
                logger.warning("Streaming producer chunk encoding error: %s", exc)
                return
            yield (data, i == len(chunks) - 1)

    loop = asyncio.get_running_loop()
    producer = _produce()
    try:
        while True:
            try:
                item = await loop.run_in_executor(None, next, producer)
            except StopIteration:
                return
            except Exception as exc:
                logger.debug("Streaming consumer stopped: %s", exc)
                return
            yield item  # type: ignore[misc]
    except (asyncio.CancelledError, GeneratorExit):
        return


# --------------------------------------------------------------------------- #
# playground
# --------------------------------------------------------------------------- #


@app.get("/", include_in_schema=False)
async def index() -> HTMLResponse:
    """The TTS playground & developer studio."""
    return HTMLResponse(INDEX_HTML)


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
