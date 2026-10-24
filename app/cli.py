"""Command line entrypoint for ``kokoro-vi-server`` / ``tts-kokoro``.

The flags exist so the image can be launched directly by a GPUStack backend
(which always appends ``--model-path`` and ``--port``) as well as by hand. A
flag left unset stays ``None`` so ``Settings.from_env`` still picks up the
environment variable; a flag passed on the command line wins over both.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import __version__

TRUTHY = frozenset({"1", "true", "yes", "y", "on", "t"})

def str2bool(value: str | bool | None) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return value.strip().lower() in TRUTHY

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kokoro-vi-server",
        description="OpenAI-compatible Vietnamese TTS server powered by Kokoro-Vietnamese.",
    )
    parser.add_argument("--version", action="version", version=f"kokoro-vi-server {__version__}")
    parser.add_argument("--host", default=None, help="bind host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default 8880)")
    parser.add_argument("--device", default=None, help="cuda, cuda:1, mps, cpu or auto")
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        default=None,
        help="HF repo id or local directory holding kokoro_vi.pth and voicepacks/",
    )
    parser.add_argument(
        "--served-model-name", default=None,
        help="name clients pass as 'model' (default Kokoro-Vietnamese)",
    )
    parser.add_argument("--default-voice", default=None, help="one of the fourteen voicepacks (default diem_trinh)")
    parser.add_argument(
        "--voicepack-dir", default=None,
        help="directory to keep downloaded voicepacks in, so restarts reuse them",
    )
    parser.add_argument(
        "--preload", type=str2bool, default=None, nargs="?", const=True, metavar="BOOL",
        help="load the model during startup rather than on the first request",
    )
    parser.add_argument("--no-warmup", dest="warmup", action="store_false", default=None,
                        help="skip the warmup sentence after loading")
    parser.add_argument("--api-key", default=None, help="require a bearer token on every request when set")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--max-concurrency", type=int, dest="max_concurrency", default=None,
        help="decodes allowed to run at once; the model is small, so 1 fits a single GPU",
    )
    parser.add_argument(
        "--max-queue", type=int, dest="max_queue", default=None,
        help="requests permitted to wait for a queue slot; a full queue returns 503",
    )
    parser.add_argument("--log-level", default=None, choices=["debug", "info", "warning", "error"])
    parser.add_argument("--cors-origins", default=None, help="comma-separated list (default *)")
    return parser

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args, _unknown = build_parser().parse_known_args(argv)
    if isinstance(args.preload, str):
        args.preload = str2bool(args.preload)
    return args

def collect_overrides(args: argparse.Namespace) -> dict[str, object]:
    """Map CLI flags onto ``Settings`` fields.

    A flag that was never passed is ``None`` and therefore excluded, so the
    env-var fallback inside ``Settings.from_env`` still applies.
    """
    overrides: dict[str, object] = {}
    for name in (
        "device", "model_path", "served_model_name", "default_voice",
        "voicepack_dir", "api_key", "hf_token", "max_concurrency", "max_queue",
        "log_level",
    ):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value

    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.preload is not None:
        overrides["preload"] = bool(args.preload)
    if args.warmup is False:
        overrides["warmup"] = False
    if args.cors_origins is not None:
        overrides["cors_origins"] = tuple(
            part.strip() for part in args.cors_origins.split(",") if part.strip()
        )
    return overrides

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    from .config import Settings
    from .server import app
    from .server import settings as module_settings

    overrides = collect_overrides(args)

    # Re-bind the settings the module-level engine reads.
    fused = Settings.from_env().merge(**overrides)
    module_settings.__dict__.update(fused.__dict__)

    import uvicorn

    uvicorn.run(
        app,
        host=fused.host,
        port=fused.port,
        log_level=fused.log_level.lower(),
        workers=1,
    )

if __name__ == "__main__":
    main()
