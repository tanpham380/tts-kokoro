# syntax=docker/dockerfile:1
# --------------------------------------------------------------------------- #
# GPU Dockerfile: nvidia/cuda runtime + torch cu126 / cu124 wheel
# Matches QUAN-ASR Dockerfile (GPU) architecture.
# --------------------------------------------------------------------------- #

ARG CUDA=12.6.1
ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126

# --------------------------------------------------------------------------- #
# Stage 1: Builder
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:${CUDA}-base-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION
ARG TORCH_INDEX

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
        build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN uv venv /opt/venv --python python${PYTHON_VERSION}
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install torch --index-url ${TORCH_INDEX}

WORKDIR /build
COPY pyproject.toml README.md* ./
COPY app ./app

RUN uv pip install . && rm -rf /root/.cache/uv

# --------------------------------------------------------------------------- #
# Stage 2: Runtime
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:${CUDA}-base-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        ffmpeg \
        libsndfile1 \
        espeak-ng \
        espeak-ng-data \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y \
    && mkdir -p /usr/share/espeak-ng-data \
    && ln -sf /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md* ./
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
    DEVICE=cuda \
    PRELOAD=true \
    LOG_LEVEL=info

VOLUME ["/cache/huggingface"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://127.0.0.1:${PORT}/ready || exit 1

EXPOSE 8880

ENTRYPOINT ["kokoro-vi-server"]
CMD ["--host", "0.0.0.0", "--port", "8880"]