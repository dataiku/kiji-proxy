# PII Detection Model Training Pipeline

Metaflow pipeline for PII detection model training.

## Pipeline Steps

1. Data export from Label Studio (optional, can be skipped with `pipeline.skip_export = true`)
2. Dataset loading and preprocessing
3. Model training with multi-task learning
4. Model evaluation
5. Model quantization (ONNX) - Linux only
6. Model signing (cryptographic hash)

## Usage

```bash
# Run locally (from project root)
uv run --extra training --extra signing python model/flows/training_pipeline.py run

# With quantization (Linux only)
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

## Configuration

Edit `training_config.toml` to change:

- `model.name` - Base model (default: distilbert-base-cased)
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
