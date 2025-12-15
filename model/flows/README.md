# Metaflow Training Pipelines

This directory contains Metaflow pipelines for training the PII detection model on Kubernetes with GPU support.

## 📁 Directory Structure

```
model/flows/
├── training_pipeline.py   # Main training pipeline
├── training_config.toml   # Training hyperparameters
├── run_training.sh        # Local execution script
├── dataset -> ../dataset  # Symlink to dataset module
└── src -> ../src          # Symlink to model source
```

## 🚀 Quick Start

### Local Execution

```bash
# Run training locally
cd model/flows
./run_training.sh

# Or run directly with Python
python training_pipeline.py run
```

### Kubernetes Execution

```bash
# Run on Kubernetes with GPU nodes
python training_pipeline.py --environment=<your-env> run

# Example with Nebius H100 cluster
python training_pipeline.py --environment=fast-bakery run
```

## ⚙️ Configuration

### training_config.toml

```toml
[training]
model_name = "distilbert-base-cased"
num_epochs = 3
batch_size = 16
learning_rate = 2e-5
weight_decay = 0.01
warmup_steps = 500

[data]
max_length = 512
train_split = 0.9

[output]
output_dir = "../trained"
```

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | `training_config.toml` | Path to config file |
| `--dataset-path` | `../dataset/reviewed_samples` | Training data directory |
| `--output-dir` | `../trained` | Model output directory |

## 🏗️ Pipeline Architecture

```
┌─────────────────┐
│     start       │  Initialize config, load parameters
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  load_dataset   │  Load and preprocess training samples
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   train_model   │  Fine-tune DistilBERT on GPU
│   (@kubernetes) │  - 8x H100 GPUs
│   (@gpu_profile)│  - Distributed training
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ evaluate_model  │  Compute metrics on validation set
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  export_model   │  Save model artifacts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      end        │  Cleanup and summary
└─────────────────┘
```

## 🔧 Kubernetes Configuration

The pipeline uses these decorators for cloud execution:

```python
@kubernetes(
    gpu=8,
    memory=640000,  # 640GB RAM
    cpu=56,
    compute_pool="obp-nebius-h100-8gpu",
    image="docker.io/eddieob/vllm-inference:latest",
)
@gpu_profile(interval=1)
```

### Compute Pools

| Pool | GPUs | Memory | Use Case |
|------|------|--------|----------|
| `obp-nebius-h100-8gpu` | 8x H100 | 640GB | Large model training |
| `default` | 1x T4 | 16GB | Testing/small models |

## 📊 Monitoring

### Metaflow Cards

The pipeline generates real-time progress cards:

```bash
# View cards in browser
python training_pipeline.py card view training
```

### GPU Profiling

GPU utilization is tracked via `@gpu_profile`:

```bash
# View GPU metrics
python training_pipeline.py card view training --id gpu_profile
```

## 🛠️ Development

### Running Steps Individually

```bash
# Run specific step
python training_pipeline.py run --run-id-file run_id.txt
python training_pipeline.py resume --origin-run-id <run-id> --step train_model
```

### Debugging

```bash
# Enable verbose logging
python training_pipeline.py --log-level debug run

# Local mode (no Kubernetes)
python training_pipeline.py --local run
```

## 📚 Related Documentation

- [Dataset Generation](../dataset/README.md) - Generate training data
- [Trained Model](../trained/README.md) - Model files and serving
- [Metaflow Docs](https://docs.metaflow.org/) - Metaflow documentation
