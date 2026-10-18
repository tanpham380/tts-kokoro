"""Request models and the OpenAI error envelope.

The error shape matters as much as the success shape: OpenAI SDKs read
``error.message`` / ``error.type`` / ``error.code``, and GPUStack surfaces
``error.code`` in its UI, so everything the server raises goes through
:class:`APIError` rather than FastAPI's default ``{"detail": ...}``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import (
    DEFAULT_RESPONSE_FORMAT,
    DEFAULT_SERVED_NAME,
    RESPONSE_FORMATS,
    STREAM_FORMATS,
    Settings,
)

#: Type string OpenAI uses for each status code we can return.
_ERROR_TYPES: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    404: "invalid_request_error",
    413: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "server_error",
    503: "server_error",
}

class APIError(Exception):
    """An error rendered as an OpenAI ``{"error": {...}}`` body."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.param = param

    @property
    def error_type(self) -> str:
        return _ERROR_TYPES.get(self.status_code, "invalid_request_error")

    def payload(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }

class SpeechRequest(BaseModel):
    """Body of ``POST /v1/audio/speech``.

    Field names follow OpenAI's ``/v1/audio/speech``. ``extra="allow"`` keeps
    clients that send OpenAI-only fields such as ``instructions`` or
    ``timestamp_granularities`` working instead of failing validation; they are
    ignored rather than rejected because TTS callers commonly forward a shared
    options dict.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model: str = Field(default=DEFAULT_SERVED_NAME)
    input: str = Field(..., description="Text to synthesize, Vietnamese or English")
    voice: str | None = Field(default=None)
    response_format: str = Field(default=DEFAULT_RESPONSE_FORMAT)
    speed: float = Field(default=1.0)
    #: ``sse`` streams OpenAI-style ``speech.audio.delta`` events; ``audio``
    #: streams raw bytes with a chunked transfer encoding.
    stream_format: Literal["sse", "audio"] | None = Field(default=None)
    #: Standard OpenAI boolean streaming flag. When true, defaults to raw audio streaming.
    stream: bool | None = Field(default=None)
    #: Accepted for OpenAI parity. Kokoro has no gain control, so anything
    #: other than 1.0 is applied as a linear multiplier on the samples.
    volume_multiplier: float = Field(default=1.0)

    @field_validator("input")
    @classmethod
    def _input_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("input must not be empty")
        return value

    @field_validator("speed")
    @classmethod
    def _speed_in_range(cls, value: float) -> float:
        if not 0.25 <= value <= 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")
        return value

    @field_validator("volume_multiplier")
    @classmethod
    def _volume_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("volume_multiplier must be positive")
        return value

    @field_validator("response_format")
    @classmethod
    def _known_format(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in RESPONSE_FORMATS:
            raise ValueError(
                f"response_format must be one of {', '.join(RESPONSE_FORMATS)}"
            )
        return value

    @field_validator("stream_format")
    @classmethod
    def _known_stream_format(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        if value not in STREAM_FORMATS:
            raise ValueError(f"stream_format must be one of {', '.join(STREAM_FORMATS)}")
        return value

    def check_limits(self, settings: Settings) -> None:
        """Reject inputs the server is configured not to serve."""
        if settings.max_text_chars and len(self.input) > settings.max_text_chars:
            raise APIError(
                f"input is {len(self.input)} characters, which exceeds the "
                f"configured limit of {settings.max_text_chars}",
                status_code=413,
                code="input_too_long",
                param="input",
            )

class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "local"
    language: str = "vi"
    sample_rate: int = 24000
    voices: int = 0

class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]

class VoiceCard(BaseModel):
    id: str
    name: str
    voice_id: str
    language: str
    gender: str
    style: str

class VoiceList(BaseModel):
    object: Literal["list"] = "list"
    data: list[VoiceCard]

class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    engine: str = "kokoro-vietnamese"
    model: str
    loaded: bool
    loading: bool
    device: str | None = None
    voices: int = 0
    version: str = ""
    error: str | None = None
