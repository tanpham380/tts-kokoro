---
trigger: glob
globs: "app/server.py, app/schemas.py, app/voices.py, app/config.py"
description: "OpenAI-compatible HTTP contract, error envelope, request aliases, voice resolution and settings wiring."
---

# API Contract Rules

## The contract is OpenAI's, not ours

`POST /v1/audio/speech` must stay a drop-in replacement. Clients that work
today: `openai-python`, `openai-node`, Open WebUI, LibreChat, Dify, LangChain,
and the GPUStack Audio Playground.

- Response bodies for errors are always `{"error": {"message", "type",
  "param", "code"}}` via `APIError.payload()`. Never let a raw exception reach
  FastAPI's default handler.
- `_ERROR_TYPES` maps status → OpenAI `type`. Add a status there before using
  it.
- `GET /v1/models`, `GET /v1/voices`, `GET /v1/audio/voices` are aliases of
  each other's data shape. Keep `object: "list"`.

## Request aliases are load-bearing

`SpeechRequest._normalize_gpustack_and_aliases` maps GPUStack's field names
onto OpenAI's. Every alias exists because a real client sends it:

| Alias | Canonical |
|---|---|
| `text`, `prompt` | `input` |
| `format` | `response_format` |
| `enable_streaming` | `stream` |
| `ref_audio` | `reference_audio` |
| `ref_text` | `transcript_of_reference_audio` |

`model_config = ConfigDict(extra="allow")` is deliberate — GPUStack sends
fields we do not model and must not 422 on them. Empty strings are sanitized to
`None` (GPUStack issue #4565).

## Voice resolution

`resolve(name, default)` in `app/voices.py` accepts, in order: canonical id,
OpenAI alias, label, composite prefix (`vi_vn_female_…`), separator-free match
(`ngochuyen`). It raises `VoiceError`, which `server.py` converts to a 400 with
`code="invalid_voice"`. Resolve **early** in the handler so a bad voice fails
before the model is touched.

## Settings

`Settings` is a frozen dataclass. Adding a field means three edits:

1. the field + default in `app/config.py`
2. the `from_env()` mapping (with a `_get` / `_flag` / `_maybe_*` helper)
3. the CLI flag in `app/cli.py` (`build_parser` + `collect_overrides`)

`Settings.merge()` rejects unknown keys — keep it that way.

## Streaming

- `stream_format=sse` → `text/event-stream` with `speech.audio.delta` events.
- `stream_format=audio` → chunked bytes with the container's MIME type.
- `stream=true` alone defaults to `audio`.
- `mp3`/`aac`/`opus` are forced to `wav` when streaming, with a warning log.
- SSE responses set `X-Accel-Buffering: no` so nginx does not buffer them.
