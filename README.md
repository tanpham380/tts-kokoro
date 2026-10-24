<div align="center">

# Kokoro-Vietnamese TTS Server

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-00a393.svg)](https://fastapi.tiangolo.com/)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-API%20Compatible-412991.svg)](https://github.com/openai/openai-python)
[![GPUStack](https://img.shields.io/badge/GPUStack-Ready-ff6900.svg)](https://github.com/gpustack/gpustack)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Hugging%20Face-ffd21e.svg)](https://huggingface.co/contextboxai/Kokoro-Vietnamese)

**High-performance Vietnamese Text-to-Speech API server** — 100% OpenAI-compatible, streaming-ready, and designed for production deployment.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [Local Installation](#local-installation)
  - [Docker & Docker Compose](#docker--docker-compose)
- [API Reference](#api-reference)
- [OpenAI Client Examples](#openai-client-examples)
- [Voices](#voices)
- [Environment Variables](#environment-variables)
- [GPUStack Deployment](#gpustack-deployment)
- [Web Studio Playground](#web-studio-playground)
- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Testing](#testing)
- [License](#license)

---

## Overview

This project wraps the **[Kokoro-Vietnamese](https://huggingface.co/contextboxai/Kokoro-Vietnamese)** model (based on [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad, StyleTTS 2 + iSTFTNet) into a production-grade **FastAPI** server that speaks Vietnamese naturally.

It serves a drop-in replacement for OpenAI's `POST /v1/audio/speech` endpoint, supports 14 Vietnamese voicepacks, and is purpose-built for deployment platforms like **GPUStack**.

### Key Highlights

| Aspect | Detail |
|---|---|
| **Model** | Kokoro-Vietnamese (82M params, StyleTTS 2 architecture) |
| **API Standard** | OpenAI-compatible (`/v1/audio/speech`) |
| **Voices** | 14 Vietnamese voicepacks + OpenAI aliases |
| **Streaming** | OpenAI SSE (`speech.audio.delta`) + raw audio chunks |
| **Languages** | Vietnamese (primary), English fallback |
| **License** | Apache-2.0 |

---

## Features

### OpenAI 100% Compatible

- **`POST /v1/audio/speech`** — main TTS endpoint, accepts standard OpenAI parameters
- **`GET /v1/models`** — model listing
- **`GET /v1/voices` & `/v1/audio/voices`** — voice pack catalogue
- Works out-of-the-box with `openai-python`, `openai-node`, Open WebUI, Dify, LangChain

### 14 Vietnamese Voicepacks

| Gender | Voices |
|---|---|
| **Female** | `diem_trinh`, `mai_linh`, `mai_loan`, `my_yen`, `ngoc_huyen`, `thuc_trinh` |
| **Male** | `duc_an`, `duc_duy`, `hung_thinh`, `manh_dung`, `phat_tai`, `thanh_dat`, `tuan_ngoc` |
| **Neutral** | `storyvert` (audiobook) |

All 13 OpenAI voice aliases (`alloy`, `nova`, `onyx`, `echo`, `shimmer`, `fable`, `coral`, `ash`, `ballad`, `marin`, `sage`, `verse`, `cedar`) are automatically mapped to the closest Vietnamese voice.

### Streaming

Two streaming modes for different use cases:

| Mode | Parameter | Use Case |
|---|---|---|
| **SSE Stream** | `stream_format=sse` | Open WebUI, LibreChat, Dify (Server-Sent Events) |
| **Audio Stream** | `stream_format=audio` | Direct browser playback (chunked bytes) |

### Production-Ready

- **Shared Model Architecture** — Kokoro-82M backbone loaded once; voicepacks are lightweight vectors (~200KB) swapped at runtime
- **Dynamic Crossfade** — 50ms linear crossfade at sentence boundaries eliminates clicks/pops
- **Text Normalization** — Vietnamese-aware regex normalizer handles standalone letters ("Y" -> /i/), abbreviations, units
- **Scheduler & Queue** — concurrency slot manager with configurable queue, backpressure, and timeout
- **Health Probes** — `/health`, `/ready`, `/live` for Kubernetes and GPUStack
- **Metrics** — `/metrics` endpoint with RTF (Real-Time Factor) and request counters

### GPUStack Integration

- Includes [`deploy/gpustack-backend.yaml`](deploy/gpustack-backend.yaml) and [`deploy/gpustack-backend-source.yaml`](deploy/gpustack-backend-source.yaml)
- Auto-detection of HuggingFace repo IDs or local directory mounts
- Support for `cuda` (NVIDIA GPU) and `cpu` profiles
- Full compatibility with all GPUStack Audio Playground parameters

### Web Studio (`/`)

A modern dark-mode glassmorphic playground with:
- Voice selector, speed/input controls
- Real-time synthesis latency display
- Built-in audio player
- Sample sentences for quick testing
- Advanced parameter panel (GPUStack-compatible)

---

## Prerequisites

- **Python** >= 3.10
- **ffmpeg** — audio encoding
- **espeak-ng** — English phonemization fallback
- **NVIDIA GPU** (optional but recommended for production)

Install system dependencies:

```bash
# macOS
brew install ffmpeg espeak-ng

# Ubuntu/Debian
sudo apt-get install ffmpeg libsndfile1 espeak-ng espeak-ng-data
```

---

## Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/tanpham380/tts-kokoro.git
cd tts-kokoro

# Install dependencies
pip install -e .

# Start the server
kokoro-vi-server --port 8880
```

Open the Web Studio at: **http://localhost:8880/**

### Docker & Docker Compose

#### Option 1: Docker Compose (Recommended)

```bash
# CPU profile (port 8881)
docker compose --profile cpu up -d

# GPU profile (port 8882)
docker compose --profile gpu up -d
```

#### Option 2: Manual Build

```bash
# GPU (CUDA)
docker build -f Dockerfile.gpu -t kokoro-vi-server:gpu .
docker run -d --gpus all -p 8880:8880 \
  -v ~/.cache/huggingface:/cache/huggingface \
  -e DEVICE=cuda \
  kokoro-vi-server:gpu

# CPU only
docker build -f Dockerfile -t kokoro-vi-server:cpu .
docker run -d -p 8880:8880 \
  -e DEVICE=cpu \
  kokoro-vi-server:cpu
```

---

## API Reference

### `POST /v1/audio/speech`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | `Kokoro-Vietnamese` | Model name (must be registered) |
| `input` | string | **required** | Text to synthesize |
| `voice` | string | `diem_trinh` | Voice ID or OpenAI alias |
| `speed` | float | `1.0` | Playback speed (0.5-2.0) |
| `format` / `response_format` | string | `mp3` | Output format: `mp3`, `wav`, `opus`, `aac`, `flac`, `pcm` |
| `stream_format` | string | — | `sse` for OpenAI streaming, `audio` for raw bytes |
| `language` | string | `vi` | Language hint (`vi`, `en`, `auto`) |

### `GET /v1/models`

Returns a list of available models in OpenAI format.

### `GET /v1/voices` | `GET /v1/audio/voices`

Returns metadata for all 14 voicepacks (ID, label, gender, style, language).

---

## OpenAI Client Examples

### cURL

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kokoro-Vietnamese",
    "input": "Xin chào! Đây là bài kiểm tra giọng nói tiếng Việt.",
    "voice": "diem_trinh",
    "response_format": "mp3",
    "speed": 1.0
  }' \
  --output output.mp3
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8880/v1",
    api_key="not-needed"
)

# Synchronous
response = client.audio.speech.create(
    model="Kokoro-Vietnamese",
    voice="mai_linh",
    input="Hôm nay thời tiết Hà Nội và Sài Gòn đều rất đẹp.",
    response_format="mp3",
    speed=1.05
)
response.stream_to_file("output.mp3")

# Streaming
with client.audio.speech.with_streaming_response.create(
    model="Kokoro-Vietnamese",
    voice="thanh_dat",
    input="Từng câu văn sẽ được chuyển thành âm thanh ngay khi mô hình xử lý xong.",
    response_format="wav",
    stream=True
) as response:
    response.stream_to_file("stream_output.wav")
```

### Node.js (OpenAI SDK)

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8880/v1',
  apiKey: 'not-needed'
});

const response = await client.audio.speech.create({
  model: 'Kokoro-Vietnamese',
  input: 'Xin chào! Đây là bài kiểm tra TTS tiếng Việt.',
  voice: 'diem_trinh',
  response_format: 'mp3',
  speed: 1.0
});

const buffer = Buffer.from(await response.arrayBuffer());
// ...
```

---

## Voices

| Voice ID | Label | Gender | Style | OpenAI Alias |
|---|---|---|---|---|
| `diem_trinh` | Diem Trinh | Female | Narration | `alloy` |
| `mai_linh` | Mai Linh | Female | General | `coral` |
| `mai_loan` | Mai Loan | Female | General | `nova` |
| `my_yen` | My Yen | Female | General | `marin` |
| `ngoc_huyen` | Ngoc Huyen | Female | General | `sage` |
| `thuc_trinh` | Thuc Trinh | Female | General | `shimmer` |
| `duc_an` | Duc An | Male | General | `ash` |
| `duc_duy` | Duc Duy | Male | General | `echo` |
| `hung_thinh` | Hung Thinh | Male | General | `verse` |
| `manh_dung` | Manh Dung | Male | General | `onyx` |
| `phat_tai` | Phat Tai | Male | General | — |
| `thanh_dat` | Thanh Dat | Male | General | `cedar` |
| `tuan_ngoc` | Tuan Ngoc | Male | General | `ballad` |
| `storyvert` | Storyvert | Neutral | Audiobook | `fable` |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `contextboxai/Kokoro-Vietnamese` | HuggingFace repo ID or local directory |
| `SERVED_MODEL_NAME` | `Kokoro-Vietnamese` | Model name exposed on `/v1/models` |
| `DEFAULT_VOICE` | `diem_trinh` | Default voice when client doesn't specify |
| `DEVICE` | `auto` | Hardware device: `auto`, `cuda`, `cpu`, `mps` |
| `PRELOAD` | `true` | Pre-load model at startup |
| `WARMUP` | `true` | Warm up CUDA kernels at startup |
| `MAX_CONCURRENCY` | `1` | Max parallel decodes |
| `MAX_QUEUE` | `256` | Max queue size (return 503 when full) |
| `QUEUE_TIMEOUT` | `60` | Max queue wait time (seconds) |
| `SYNTHESIS_TIMEOUT` | `300` | Max seconds one synthesis may run (0 disables) |
| `CACHE_MAX_ENTRIES` | `256` | Audio cache entry cap (0 disables) |
| `CACHE_MAX_BYTES` | `268435456` | Audio cache byte cap (0 disables) |
| `CROSSFADE_MS` | `50` | Crossfade duration between sentences (ms) |
| `SPEED` | `1.0` | Default synthesis speed |
| `PORT` | `8880` | HTTP listen port |
| `API_KEY` | `None` | Bearer token for API authentication |
| `LOG_LEVEL` | `info` | Logging verbosity |
| `LOG_JSON` | `false` | Emit one JSON object per log line |

### Observability

| Endpoint | Format | Use |
|---|---|---|
| `GET /metrics` | JSON | Humans, Web Studio |
| `GET /metrics/prometheus` | Prometheus text | Scrape this |

Every response carries `X-Request-ID`; send your own to have it echoed back.
`POST /v1/audio/speech` also returns `X-Cache: HIT` or `MISS`.

### Text normalization

`app/text/` normalizes Vietnamese before phonemization. The passes handle
numbers, dates, times, currency, units, percentages and standalone letters.

The word tables in `app/text/data/` are copied verbatim from
[NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing)'s
Vietnamese grammar (Apache-2.0 — see `app/text/data/NOTICE`). Only the **data**
was copied, not the Python: NeMo's grammar needs `pynini`, a 147.6 MB
manylinux-only wheel with no Windows build, while the tables are 4.1 KB of TSV
read by ~200 lines of standard-library code in `app/text/data.py`.

There is deliberately **no abbreviation dictionary** — Vietnamese abbreviations
are an open set, so only closed sets (the Latin alphabet, the number/date/unit
grammar) are handled, and by rule. See
[`.agents/rules/text-normalization.md`](.agents/rules/text-normalization.md).

---

## GPUStack Deployment

### Step 1: Import Backend

1. Open GPUStack Admin -> **Inference Backends** -> **Add Backend**
2. Select **YAML** mode
3. Paste the content of [`deploy/gpustack-backend.yaml`](deploy/gpustack-backend.yaml)
4. Backend name: `Kokoro-TTS-Vietnamese-custom`

### Step 2: Deploy Model

1. Go to **Models** -> **Deploy Model**
2. **Category**: `Audio / Text to Speech`
3. **Backend**: Select `Kokoro-TTS-Vietnamese-custom`
4. **Hugging Face Model**: `contextboxai/Kokoro-Vietnamese`

GPUStack will:
- Automatically download the model
- Mount it to `{{model_path}}` inside the container
- Launch `kokoro-vi-server` with GPUStack-injected parameters
- Server transitions to `ready` state via `/health` and `/ready` probes

---

## Web Studio Playground

The server includes a built-in web playground at `http://localhost:8880/` featuring:

- Dark-mode glassmorphic UI
- Voice selector with all 14 Vietnamese voices
- Speed adjustment and text input
- Real-time synthesis latency display
- Integrated audio player
- Sample sentences for quick testing
- Advanced parameter panel compatible with GPUStack Playground

---

## Project Structure

```
tts-kokoro/
├── app/
│   ├── __init__.py         # Server version
│   ├── audio.py            # PCM processing, encoding, crossfade, streaming
│   ├── cli.py              # CLI entrypoint (GPUStack-compatible flags)
│   ├── config.py           # Environment-driven Settings dataclass
│   ├── engine.py           # VoiceRuntime, model lifecycle, synthesis pipeline
│   ├── schemas.py          # Pydantic request/response schemas
│   ├── server.py           # FastAPI routes and middleware
│   ├── ui.py               # Web Studio HTML/CSS/JS
│   └── voices.py           # Voice catalogue and OpenAI alias mapping
├── deploy/
│   ├── gpustack-backend.yaml          # GPUStack Custom Inference Backend
│   └── gpustack-backend-source.yaml   # GPUStack InferenceBackendSource
├── docker/
│   └── docker-compose.yaml  # Docker Compose (CPU + GPU profiles)
├── scripts/
│   ├── smoke.sh             # Quick endpoint verification
│   └── test_openai_client.py # OpenAI SDK test client
├── tests/                   # Test suite (unit + integration)
├── Dockerfile              # Production multi-stage (CPU)
├── Dockerfile.gpu          # Production multi-stage (CUDA)
├── pyproject.toml          # Package definition
└── README.md
```

---

## Architecture Overview

```
Client Request
      |
      v
+-----------------+
|   FastAPI Server |  <- POST /v1/audio/speech
|   (app/server.py)|
+--------+---------+
         |
    +----v-----+
    |  Engine   |  <- Text normalization (vig2p)
    |(app/engine|      Sentence splitting
    |   .py)    |      Phonemization
    +----+-----+
         |
    +----v-----+
    | VoiceRT   |  <- Shared KModel backbone
    | (app/engine|    Voicepack swap per request
    |   .py)    |      Dynamic crossfade (50ms)
    +----+-----+
         |
    +----v-----+
    |  Audio    |  <- PCM -> encoding (mp3/opus/aac/wav/flac)
    |(app/audio |      Streaming chunk emission
    |   .py)    |
    +----------+
```

### Key Design Decisions

1. **Shared Model, Swappable Voicepacks** — The 82M-parameter backbone is loaded once. Each of the 14 voices is a ~200KB vector, swapped in microseconds.
2. **Sentence Splitting with Crossfade** — Long input is split into sentences, decoded independently, and joined with a 50ms linear crossfade to eliminate clicks.
3. **Vietnamese-Aware Normalization** — Before phonemization, standalone letters ("Y", "A", "O") are mapped to Vietnamese readings, abbreviations ("TP.HCM" -> "Thanh pho Ho Chi Minh") are expanded, and units are vocalized.
4. **Concurrency Scheduling** — A slot manager controls parallel decodes, enforces queue limits, and returns HTTP 503 under overload.

---

## Testing

```bash
# Run the full test suite
python3 -m pytest tests/

# Quick smoke test (requires running server)
bash scripts/smoke.sh
```

---

## License

Apache License 2.0 — see the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0) for details.

The underlying **Kokoro-Vietnamese** model weights are licensed by their respective authors. Please refer to the model card on [HuggingFace](https://huggingface.co/contextboxai/Kokoro-Vietnamese) for model-specific licensing terms.

---

## Acknowledgements

- **[hexgrad](https://huggingface.co/hexgrad)** — original Kokoro-82M model
- **[contextboxai](https://huggingface.co/contextboxai)** — Kokoro Vietnamese fine-tuning
- **[iamdinhthuan](https://github.com/iamdinhthuan)** — G2P and `dinhthuan/kokoro-vi-ngoc-huyen` checkpoint
- **[Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)** — serving architecture reference
- **[GPUStack](https://github.com/gpustack/gpustack)** — model serving platform
- **[vig2p](https://github.com/vigotech/vig2p)** — Vietnamese G2P library
