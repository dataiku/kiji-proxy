# PII Detection Model Training Pipeline

Metaflow pipeline for PII detection model training.

## Pipeline Steps

1. Data export from Label Studio (optional, can be skipped with `pipeline.skip_export = true`)
2. Dataset loading and preprocessing
3. PII detection model training
4. Model evaluation
5. Model export (ONNX) with parity checks; quantization is disabled by default
6. Model signing (cryptographic hash)

## Usage

```bash
# Run locally (from project root)
uv run --extra training --extra signing python model/flows/training_pipeline.py run

# ONNX export currently uses the dependencies in the quantization extra.
uv run --extra training --extra quantization --extra signing python model/flows/training_pipeline.py run

# Custom config file
uv run --extra training python model/flows/training_pipeline.py --config-file custom_config.toml run

# Remote Kubernetes execution (uncomment @pypi and @kubernetes decorators first)
python model/flows/training_pipeline.py --environment=pypi run --with kubernetes
```

Or use the helper script:

```bash
./model/flows/run_training.sh
./model/flows/run_training.sh --config custom_config.toml
```

## Running in Docker

`model/Dockerfile` packages the pipeline and its `training` / `quantization` /
`signing` extras into a reproducible image, so training does not depend on the
host toolchain. Build from the repository root — the build context needs
`pyproject.toml` and `uv.lock`:

```bash
make docker-train-build                        # docker build -f model/Dockerfile -t kiji-model-trainer .

make docker-train                              # CPU
make docker-train GPU=1                        # NVIDIA GPUs (--gpus all)
make docker-train NUM_SAMPLES=500              # quick smoke run on a subset
make docker-train ARGS="--config config-file /workspace/custom.toml"
```

The equivalent plain Docker invocation:

```bash
docker run --rm --gpus all --shm-size=2g \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/model/dataset/data_samples:/workspace/model/dataset/data_samples:ro" \
  -v "$PWD/model/trained:/workspace/model/trained" \
  -v "$PWD/model/quantized:/workspace/model/quantized" \
  kiji-model-trainer run
```

Notes:

- The container's `/workspace` mirrors the repository root, because the paths in
  `training_config.toml` are resolved relative to the working directory.
- Datasets go in and artifacts come out through bind mounts; nothing is baked
  into the image. `model/dataset/data_samples` is excluded by `.dockerignore`.
- One image covers CPU and GPU: the locked `torch` wheel bundles its own CUDA
  runtime, so only the host NVIDIA driver and container toolkit are needed.
- Useful environment variables: `HF_TOKEN` (gated Hugging Face datasets),
  `NUM_SAMPLES`, `NUM_AI4PRIVACY_SAMPLES`, `LABEL_STUDIO_API_KEY` (only when
  `pipeline.skip_export = false`), and `MODEL_SIGNING_KEY_PATH` — mount the key
  read-only if you sign with a private key.
- The entrypoint is the flow itself, so any Metaflow subcommand works:
  `docker run --rm kiji-model-trainer show`.

Run the checkpoint-vs-ONNX parity check directly:

```bash
uv run python -m model.src.parity_benchmark \
  --checkpoint ./model/trained \
  --onnx-model ./model/quantized \
  --onnx-file model.onnx
```

## Configuration

Edit `training_config.toml` to change:

- `model.name` - Base model (default: microsoft/deberta-v3-small)
- `training.num_epochs` - Number of epochs
- `training.batch_size` - Batch size
- `training.learning_rate` - Learning rate
- `data.subsample_count` - Limit samples for testing (0 = use all)
- `paths.training_samples_dir` - Path to training data
- `paths.output_dir` - Where to save trained model
- `labelstudio.project_id` - Label Studio project ID (required for export step)
- `labelstudio.base_url` - Label Studio base URL (default: http://localhost:8080)
- `labelstudio.api_key` - Label Studio API key (or set LABEL_STUDIO_API_KEY env var)
- `pipeline.skip_export` - Skip Label Studio export step (default: false)
