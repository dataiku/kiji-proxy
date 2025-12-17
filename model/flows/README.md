# PII Detection Model Training Pipeline

Metaflow pipeline for PII detection model training.

## Pipeline Steps

1. Dataset loading and preprocessing
2. Model training with multi-task learning
3. Model evaluation
4. Model quantization (ONNX) - Linux only
5. Model signing (cryptographic hash)

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
