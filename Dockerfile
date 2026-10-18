# syntax=docker/dockerfile
# CPU + shared-base Dockerfile, the pattern used by [speakcrate](/hwdsl2/speakcrate) and [Kokoro-FastAPI](/remsky/Kokoro-FastAPI).
#
# Two-stage build with Astral uv. Stage 1 installs the Python deps; stage 2 is
# a slim image that just runs the server.

# --------------------------------------------------------------------------- #
# Stage 1: Builder
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU wheel of torch is smaller than the CUDA one and lets CPU images build fast.
# The GPU image (built from this base with the torch url swapped) gets the CUDA
# variant instead.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN uv pip install torch --index-url ${TORCH_INDEX}

WORKDIR /build
COPY pyproject.toml README.md ./
COPY app ./app

RUN uv pip install . && rm -rf /root/.cache/uv

# --------------------------------------------------------------------------- #
# Stage 2: Runtime (CPU or GPU, flipped by the --build-arg)
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        espeak-ng \
        espeak-ng-data \
        curl \
        ca-certificates \
    && mkdir -p /usr/share/espeak-ng-data \
    && ln -sf /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY app ./app

ENV HF_HOME=/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=contextboxai/Kokoro-Vietnamese \
    SERVED_MODEL_NAME=Kokoro-Vietnamese \
    DEFAULT_VOICE=diem_trinh \
    HOST=0.0.0.0 \
    PORT=8880 \
    DEVICE=cpu \
    PRELOAD=true \
    LOG_LEVEL=info

VOLUME ["/cache/huggingface"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -fsS http://127.0.0.1:${PORT}/ready || exit 1

EXPOSE 8880

ENTRYPOINT ["kokoro-vi-server"]
CMD ["--host", "0.0.0.0", "--port", "8880"]