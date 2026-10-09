# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.11",
#     "torchvision",
#     "transformers>=5.10.2",
#     "accelerate",
#     "safetensors",
#     "huggingface-hub",
#     "pillow",
#     "fastapi>=0.104.0",
#     "uvicorn>=0.24.0",
# ]
# ///
"""Serve Cloudflare Clef / Clef-flash locally behind the Jev ``/v1/systemone`` API.

Clef's open weights (Apache 2.0, ``Cloudflare/clef-flash`` and ``Cloudflare/clef``
on Hugging Face) need a newer ``transformers`` than the rest of this project, so
the server runs as a standalone script in its own environment::

    uv run model/dataset/multilingual/clef_server.py                # clef-flash, port 8002
    uv run model/dataset/multilingual/clef_server.py --model Cloudflare/clef --device cuda

Then point the pipeline at it with ``--decision_backend=clef-local``.

The first start downloads the weights (about 19 GB for Clef-flash) and imports
``joint_schema_model.py`` from the model repository; pin ``--revision`` to run a
reviewed copy of that code.
"""

import argparse
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, Header, HTTPException

DEFAULT_MODEL = "Cloudflare/clef-flash"
DEFAULT_PORT = 8002
DEFAULT_MAX_LENGTH = 16384

Answerer = Callable[[dict[str, Any]], dict[str, Any]]


def pick_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to cuda, then mps (Apple Silicon), then cpu."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_answerer(
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    dtype: str | None = None,
    revision: str | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Answerer:
    """Load a Clef release and return a function answering SystemOne requests."""
    import torch
    from huggingface_hub import snapshot_download

    path = Path(model)
    if not path.is_dir():
        path = Path(snapshot_download(model, revision=revision))
    # The release ships its own model code next to the weights.
    sys.path.insert(0, str(path))
    import joint_schema_model

    device = pick_device(device)
    torch_dtype = getattr(
        torch, dtype or ("float32" if device == "cpu" else "bfloat16")
    )
    print(f"Loading {model} on {device} ({torch_dtype})")
    clef, processor = joint_schema_model.load_release_model(
        path, device=device, dtype=torch_dtype
    )

    def answer(request: dict[str, Any]) -> dict[str, Any]:
        return joint_schema_model.systemone(
            clef, processor, request, max_length=max_length
        )

    return answer


def create_app(
    answer: Answerer, model_name: str = DEFAULT_MODEL, api_key: str | None = None
) -> FastAPI:
    """FastAPI app exposing ``POST /v1/systemone``; ``answer`` runs the model."""
    app = FastAPI(title="Clef local server")
    lock = threading.Lock()  # one forward pass at a time on a single device

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": model_name}

    @app.post("/v1/systemone")
    def systemone(
        request: dict[str, Any] = Body(...),
        authorization: str | None = Header(None),
    ) -> dict[str, Any]:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="Invalid API key")
        if not isinstance(request.get("model"), str):
            request = {**request, "model": model_name}
        try:
            with lock:
                return answer(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Local Clef decision-model server")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local directory"
    )
    parser.add_argument("--revision", default=None, help="Model repo revision to pin")
    parser.add_argument(
        "--device", default="auto", help="auto, cuda, mps or cpu (default: auto)"
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="torch dtype name (default: bfloat16, float32 on cpu)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum input tokens per request",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    import uvicorn

    answer = load_answerer(
        args.model, args.device, args.dtype, args.revision, args.max_length
    )
    app = create_app(answer, args.model, os.getenv("CLEF_API_KEY"))
    print(f"Serving {args.model} at http://{args.host}:{args.port}/v1/systemone")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
