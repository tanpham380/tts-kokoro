---
trigger: model_decision
description: "Apply when editing Dockerfiles, docker-compose, GitLab CI, GPUStack backend YAML, or anything that changes how the image is built, tagged or deployed."
---

# Build & Deploy Rules

## Images

| File | Base | Torch | Tag |
|---|---|---|---|
| `Dockerfile` / `Dockerfile.gpu` | `nvidia/cuda:12.6.1` | cu126 | `:<tag>-gpu` |
| `Dockerfile.cpu` | `python:3.11-slim` | cpu | `:<tag>-cpu` |

Both are two-stage: a `uv` builder installs into `/opt/venv`, the runtime stage
copies that venv and adds `ffmpeg`, `libsndfile1`, `espeak-ng`. `torch` is
installed **before** `uv pip install .` so the platform wheel wins over the
default one pulled in by `kokoro-vietnamese`.

- Keep `CUDA` and `TORCH_INDEX` in step in `.gitlab-ci.yml`. A cu126 image with
  a cu121 index silently installs the wrong wheel.
- `espeak-ng-data` is symlinked into `/usr/share/espeak-ng-data`; the English
  fallback G2P needs it.
- `HF_HOME=/cache/huggingface` is a declared `VOLUME`. Do not move it without
  updating `docker/docker-compose.yaml`.

## CI

`.gitlab-ci.yml` triggers only on tags matching `^v-?.*-release$`. It builds
both images with `buildx`, pushes four tags each (registry-scoped and
root-scoped, `:<tag>` and `:latest`), then cuts a GitLab Release. There is
deliberately **no deploy stage** — GPUStack pulls the images.

## GPUStack

`deploy/gpustack-backend.yaml` and `deploy/gpustack-backend-source.yaml` define
the custom backend. GPUStack appends `--model-path` and `--port`, which is why
`app/cli.py` accepts both and why a flag left unset stays `None` so the env var
still applies.

When GPUStack mounts a model directory, `_build_runtime()` and
`_load_voicepack()` resolve files locally and make **zero** network calls. That
behaviour is tested in `test_gpustack_local_directory_mount_resolution` — do not
break it by reordering the candidate lists.

## Local

```bash
docker compose --profile cpu up -d   # port 8881
docker compose --profile gpu up -d   # port 8882
./scripts/smoke.sh http://127.0.0.1:8881
```

`scripts/smoke.sh` is bash + curl + grep. It checks `/health`, `/ready`,
`/v1/models`, `/v1/voices`, an MP3 synthesis and an `alloy` alias synthesis.
Keep it dependency-free.
