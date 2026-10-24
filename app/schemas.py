"""Request models and the OpenAI error envelope.

The error shape matters as much as the success shape: OpenAI SDKs read
``error.message`` / ``error.type`` / ``error.code``, and GPUStack surfaces
``error.code`` in its UI, so everything the server raises goes through
:class:`APIError` rather than FastAPI's default ``{"detail": ...}``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.param = param
        #: Seconds a client should wait before retrying. Only meaningful on a
        #: 503; rendered as the ``Retry-After`` header.
        self.retry_after = retry_after

    @property
    def error_type(self) -> str:
        return _ERROR_TYPES.get(self.status_code, "invalid_request_error")

    def headers(self) -> dict[str, str]:
        """Response headers this error implies.

        ``Retry-After`` is only sent on a 503, and only when a wait was
        estimated — a bare ``Retry-After: 0`` would invite an immediate retry
        storm, which is the opposite of what backpressure is for.
        """
        if self.status_code == 503 and self.retry_after is not None:
            return {"Retry-After": str(max(1, round(self.retry_after)))}
        return {}

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

    Supports OpenAI standard fields as well as GPUStack TTS Playground / CustomVoice fields:
    - ``task_type``: e.g. "CustomVoice", "ZeroShot", "Text-to-Speech", "TTS", "VoiceDesign"
    - ``language``: "vi", "en", "auto", "vietnamese", etc.
    - ``instructions``: Style / emotion / delivery instructions
    - ``max_new_tokens``: Generation token limit
    - ``reference_audio`` / ``ref_audio``: Reference audio for speaker cloning / prompt
    - ``transcript_of_reference_audio`` / ``ref_text``: Transcript of reference audio for ICL
    - ``use_speaker_embedding_only``: Use speaker embedding only without ICL
    - ``format``: GPUStack playground alias for ``response_format``
    - ``enable_streaming``: GPUStack playground alias for ``stream``
    - ``text`` / ``prompt``: Input text aliases
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model: str = Field(default=DEFAULT_SERVED_NAME)
    input: str = Field(default="", description="Text to synthesize, Vietnamese or English")
    text: str | None = Field(default=None, description="Alternative alias for input")
    prompt: str | None = Field(default=None, description="Alternative alias for input")

    voice: str | None = Field(default=None)
    response_format: str = Field(default=DEFAULT_RESPONSE_FORMAT)
    format: str | None = Field(default=None, description="GPUStack alias for response_format")

    speed: float = Field(default=1.0)
    stream_format: Literal["sse", "audio"] | None = Field(default=None)
    stream: bool | None = Field(default=None)
    enable_streaming: bool | None = Field(default=None, description="GPUStack alias for stream")
    volume_multiplier: float = Field(default=1.0)

    # --- GPUStack Advanced Parameters ---
    task_type: str | None = Field(
        default=None,
        description="GPUStack task type, e.g. 'CustomVoice', 'Text-to-Speech', 'ZeroShot'",
    )
    language: str | None = Field(
        default="vi",
        description="Spoken language, e.g. 'vi', 'en', 'auto', 'vietnamese'",
    )
    instructions: str | None = Field(
        default=None,
        description="Style, emotion or tempo instructions",
    )
    max_new_tokens: int | None = Field(
        default=None,
        description="Max generation token limit",
    )
    reference_audio: str | None = Field(
        default=None,
        description="Reference audio path, URL, or base64 audio string",
    )
    ref_audio: str | None = Field(
        default=None,
        description="GPUStack alias for reference_audio",
    )
    transcript_of_reference_audio: str | None = Field(
        default=None,
        description="Transcript of reference audio for ICL (In-Context Learning) mode",
    )
    ref_text: str | None = Field(
        default=None,
        description="GPUStack alias for transcript_of_reference_audio",
    )
    use_speaker_embedding_only: bool | None = Field(
        default=None,
        description="Use speaker embedding only without In-Context Learning (no ICL)",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_gpustack_and_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # 1. Alias input / text / prompt
        input_val = data.get("input")
        if not input_val or not str(input_val).strip():
            fallback_text = data.get("text") or data.get("prompt")
            if fallback_text and str(fallback_text).strip():
                data["input"] = str(fallback_text).strip()

        # 2. Alias format -> response_format
        fmt = data.get("format")
        if fmt and str(fmt).strip():
            cur_resp = data.get("response_format")
            if not cur_resp or cur_resp == DEFAULT_RESPONSE_FORMAT:
                data["response_format"] = str(fmt).strip().lower()

        # 3. Alias enable_streaming -> stream
        if "enable_streaming" in data and "stream" not in data:
            data["stream"] = bool(data["enable_streaming"])

        # 4. Alias ref_audio -> reference_audio
        if "ref_audio" in data and not data.get("reference_audio"):
            data["reference_audio"] = data["ref_audio"]

        # 5. Alias ref_text -> transcript_of_reference_audio
        if "ref_text" in data and not data.get("transcript_of_reference_audio"):
            data["transcript_of_reference_audio"] = data["ref_text"]

        # 6. Sanitize empty strings to None (Fixes GPUStack Issue #4565)
        for key in ("reference_audio", "ref_audio", "transcript_of_reference_audio", "ref_text"):
            if key in data and isinstance(data[key], str) and not data[key].strip():
                data[key] = None

        return data

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

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str | None) -> str | None:
        if not value:
            return "vi"
        val = value.strip().lower()
        if val in ("vietnamese", "tiếng việt", "vi-vn"):
            return "vi"
        if val in ("english", "tiếng anh", "en-us", "en-gb"):
            return "en"
        return val

    @field_validator("max_new_tokens")
    @classmethod
    def _positive_tokens(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_new_tokens must be positive")
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
