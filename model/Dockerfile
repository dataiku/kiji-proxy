# Training image for the PII detection model.
#
# Runs the Metaflow pipeline in model/flows/training_pipeline.py:
#   export (optional) -> preprocess -> train -> evaluate -> ONNX export -> sign
#
# Build from the repository root (needs pyproject.toml + uv.lock in context):
#   docker build -f model/Dockerfile -t kiji-model-trainer .
#
# Run (CPU):
#   docker run --rm \
#     -v "$PWD/model/dataset/data_samples:/workspace/model/dataset/data_samples:ro" \
#     -v "$PWD/model/trained:/workspace/model/trained" \
#     -v "$PWD/model/quantized:/workspace/model/quantized" \
#     kiji-model-trainer
#
# Run (NVIDIA GPU) — the locked torch wheel bundles its own CUDA runtime, so
# only the host driver plus the container toolkit are required:
#   docker run --rm --gpus all --shm-size=2g ... kiji-model-trainer

ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim-bookworm AS base

# git: Metaflow records repository provenance for each run
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the version that produced uv.lock
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace

# Dependencies first, so source edits don't invalidate the (multi-GB) install layer.
# --frozen installs exactly what uv.lock pins without re-resolving. (Not --locked:
# uv.lock trails pyproject.toml by a version bump, which is unrelated to deps.)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project \
        --extra training --extra quantization --extra signing

# Training code. model/flows/{src,dataset} are symlinks into model/, which is
# how `python model/flows/training_pipeline.py` resolves `from src.config ...`.
COPY README.md ./
COPY model/ ./model/

# The ONNX-export step's parity check imports tests.benchmark.run for its
# OnnxPIIModel, resolved from the repo root that src/parity_benchmark.py pushes
# onto sys.path — so the trainer needs these at runtime, not just under pytest.
COPY tests/__init__.py ./tests/
COPY tests/benchmark/ ./tests/benchmark/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen \
        --extra training --extra quantization --extra signing

# Paths in training_config.toml are relative to the working directory, so the
# layout under /workspace must mirror the repository root.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    WANDB_DISABLED=true \
    WANDB_MODE=disabled \
    HOME=/workspace/.home \
    HF_HOME=/workspace/.cache/huggingface \
    METAFLOW_USER=kiji-trainer \
    METAFLOW_DEFAULT_METADATA=local \
    METAFLOW_DATASTORE_SYSROOT_LOCAL=/workspace/.metaflow

# Bind-mount points for datasets in and artifacts out. Created up front so they
# exist (and are writable) even when nothing is mounted over them. Caches and
# Metaflow run state are made world-writable so that
# `docker run --user $(id -u):$(id -g)` — the usual way to keep bind-mounted
# artifacts owned by the host user — still has somewhere to write.
RUN set -eux; \
    groupadd --gid 1000 trainer; \
    useradd --uid 1000 --gid 1000 --no-log-init --home-dir /workspace/.home trainer; \
    mkdir -p \
        /workspace/.home \
        /workspace/.cache/huggingface \
        /workspace/.metaflow \
        /workspace/model/dataset/data_samples/training_samples \
        /workspace/model/trained \
        /workspace/model/quantized; \
    chown -R 1000:1000 /workspace; \
    chmod -R 0777 \
        /workspace/.home \
        /workspace/.cache \
        /workspace/.metaflow \
        /workspace/model/trained \
        /workspace/model/quantized

USER trainer

# `run` is Metaflow's subcommand; append flags such as
# `--config config-file /workspace/custom.toml` before it, or `--help` to explore.
ENTRYPOINT ["/opt/venv/bin/python", "model/flows/training_pipeline.py"]
CMD ["run"]
