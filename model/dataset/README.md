# PII Training Data Generation

This directory contains tools for generating synthetic PII training data using LLMs with structured outputs.

## Directory Structure

```
model/dataset/
├── doubleword/                  # Batch processing via Doubleword API
│   ├── pipeline.py              # Automated dataset generation pipeline
│   ├── pipeline_state.py        # State management for resumability
│   ├── batch_generator.py       # Batch request generation
│   ├── batch_monitor.py         # Automatic polling & monitoring
│   ├── doubleword_client.py     # Doubleword API client
│   └── result_processor.py      # Result processing
│
├── openai/                      # Direct generation via OpenAI API
│   ├── training_set.py          # Training set generator
│   ├── api_clients.py           # OpenAI API client
│   ├── prompts.py               # Prompt templates for generation/review
│   └── schemas.py               # JSON schemas for structured outputs
│
├── labelstudio/                 # Label Studio integration
│   ├── labelstudio_format.py    # Convert samples to Label Studio format
│   ├── export_annotations.py    # Export annotations from Label Studio
│   └── import_predictions.py    # Import predictions to Label Studio
│
├── data_samples/                # Generated data samples
│   ├── samples/                 # Raw generated samples
│   ├── reviewed_samples/        # LLM-reviewed and corrected samples
│   ├── annotation_samples/      # Label Studio-ready samples
│   └── training_samples/        # Final training samples (used by training pipeline)
│
├── huggingface/                 # HuggingFace Hub integration
│   ├── upload_dataset_to_hf.py   # Export dataset to HuggingFace Hub (Parquet)
│   ├── download_dataset_from_hf.py # Import dataset from HuggingFace Hub to local JSON
│   └── upload_model_to_hf.py    # Upload trained/quantized model to HuggingFace Hub
│
├── label_utils.py               # PII label definitions and utilities
├── file_operations.py           # File I/O utilities
└── tokenization.py              # Tokenization for training samples
```

## Quick Start

### Option 1: Batch Processing (Doubleword) - Recommended for Large Datasets

The Doubleword pipeline automates the entire workflow with automatic polling and resumability:

```bash
# Set API key
export DOUBLEWORD_API_KEY="your-api-key"

# Generate 100 samples - fully automated
uv run python -m model.dataset.doubleword.pipeline \
  --command=start \
  --num_samples=100

# Generate with optional review stage for higher quality
uv run python -m model.dataset.doubleword.pipeline \
  --command=start \
  --num_samples=100 \
  --enable_review

# Check status anytime
uv run python -m model.dataset.doubleword.pipeline --command=status

# Resume if interrupted
uv run python -m model.dataset.doubleword.pipeline --command=resume
```

**See [DOUBLEWORD_QUICKSTART.md](./doubleword/DOUBLEWORD_QUICKSTART.md) for complete documentation.**

### Option 2: Direct Generation (OpenAI) - For Quick Testing

```bash
# Set API key
export OPENAI_API_KEY="your-api-key"

# Generate 10 samples using OpenAI API
uv run python -m model.dataset.openai.training_set --num_samples=10

# Generate using a custom OpenAI-compatible API
export URL=http://your-server:8000/v1/chat/completions
uv run python -m model.dataset.openai.training_set --num_samples=100 --api_url=$URL

# High-throughput generation with parallel workers
uv run python -m model.dataset.openai.training_set --num_samples=1000 --max_workers=12
```

### HuggingFace Dataset Sharing

The dataset can be shared via HuggingFace Hub as a Parquet-backed dataset. The upload script converts from the internal Label Studio format to a clean training format with `text`, `privacy_mask`, `coreferences`, `language`, and `country` columns, then creates train/test splits.

```bash
export HF_TOKEN=hf_xxxxx

# Upload as private dataset (default)
uv run python model/dataset/huggingface/upload_dataset_to_hf.py --repo-id "username/kiji-pii-training-data"

# Upload as public dataset
uv run python model/dataset/huggingface/upload_dataset_to_hf.py --repo-id "username/kiji-pii-training-data" --public

# Custom test split ratio (default: 0.1)
uv run python model/dataset/huggingface/upload_dataset_to_hf.py --repo-id "username/kiji-pii-training-data" --test-split-ratio 0.2
```

Consumers can load the dataset with:

```python
from datasets import load_dataset
ds = load_dataset("username/kiji-pii-training-data")
```

To bootstrap a local training environment from a shared HuggingFace dataset:

```bash
# Download all splits to local training_samples/ directory
uv run python model/dataset/huggingface/download_dataset_from_hf.py --repo-id "username/kiji-pii-training-data"

# Download to a custom directory
uv run python model/dataset/huggingface/download_dataset_from_hf.py --repo-id "username/kiji-pii-training-data" --output-dir path/to/output

# Download only specific splits
uv run python model/dataset/huggingface/download_dataset_from_hf.py --repo-id "username/kiji-pii-training-data" --splits train
```

This converts each row back to Label Studio JSON files that the training pipeline can consume directly.

### Upload Models to HuggingFace

Upload trained and/or quantized models to HuggingFace Hub with auto-generated model cards. The script supports cross-linking to show the full lineage: dataset -> trained model -> quantized model.

```bash
export HF_TOKEN=hf_xxxxx

# Upload trained model (links to dataset + quantized model)
uv run python model/dataset/huggingface/upload_model_to_hf.py \
  --variant trained \
  --repo-id "username/kiji-pii-model" \
  --dataset-repo-id "username/kiji-pii-training-data" \
  --quantized-repo-id "username/kiji-pii-model-onnx" \
  --public

# Upload quantized ONNX model (links back to trained model + dataset)
uv run python model/dataset/huggingface/upload_model_to_hf.py \
  --variant quantized \
  --repo-id "username/kiji-pii-model-onnx" \
  --trained-repo-id "username/kiji-pii-model" \
  --dataset-repo-id "username/kiji-pii-training-data" \
  --public
```

## Which Method to Use?

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Doubleword** | Large datasets (100+) | Automated, resumable, cost-effective batch pricing | Requires waiting for batch completion |
| **OpenAI** | Quick testing, small datasets | Immediate results, real-time feedback | Higher cost per sample |

**Recommendation:** Use **Doubleword** for production datasets (>100 samples), use **OpenAI** for testing and iteration.

## Configuration Options

### Doubleword Pipeline

| Flag | Default | Description |
|------|---------|-------------|
| `--command` | `start` | Command: start, status, resume, reset, cancel |
| `--num_samples` | 100 | Number of samples to generate |
| `--api_model` | `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` | Model name for generation |
| `--[no]auto_poll` | `true` | Automatically wait for batch completion |
| `--[no]enable_review` | `false` | Enable optional review stage |
| `--poll_interval` | 60 | Seconds between status checks |
| `--output_dir` | `model/dataset/doubleword` | Output directory for samples |
| `--max_workers` | auto | Parallel workers (default: min(32, num_samples + 4)) |
| `--log_level` | INFO | Logging verbosity |

### OpenAI Training Set

| Flag | Default | Description |
|------|---------|-------------|
| `--num_samples` | 5 | Number of samples to generate |
| `--api_url` | None | Custom API URL for OpenAI-compatible servers |
| `--api_model` | `gpt-4.1-mini` | Model name (default optimized for cost) |
| `--training_output_dir` | `model/dataset` | Output directory for samples |
| `--max_workers` | auto | Parallel workers (default: min(12, num_samples + 4)) |
| `--log_level` | WARNING | Logging verbosity |

### Upload to HuggingFace

| Flag | Default | Description |
|------|---------|-------------|
| `--repo-id` | (required) | HuggingFace repo ID (e.g., `username/dataset-name`) |
| `--samples-dir` | `model/dataset/data_samples/training_samples` | Directory containing Label Studio JSON samples |
| `--public` | False | Make dataset public (default: private) |
| `--test-split-ratio` | 0.1 | Fraction of data for the test split |

### Download from HuggingFace

| Flag | Default | Description |
|------|---------|-------------|
| `--repo-id` | (required) | HuggingFace repo ID (e.g., `username/dataset-name`) |
| `--output-dir` | `model/dataset/data_samples/training_samples` | Directory to save Label Studio JSON files |
| `--splits` | all | Which splits to download (e.g., `train test`) |

## Sample Format

Generated samples follow this JSON structure:

```json
{
  "text": "Contact Dr. Maria Santos at maria.santos@hospital.org or call +1-555-123-4567.",
  "language": "English",
  "country": "United States",
  "privacy_mask": [
    {"value": "Maria", "label": "FIRSTNAME"},
    {"value": "Santos", "label": "SURNAME"},
    {"value": "maria.santos@hospital.org", "label": "EMAIL"},
    {"value": "+1-555-123-4567", "label": "PHONENUMBER"}
  ],
  "coreferences": [
    {
      "cluster_id": 0,
      "mentions": [
        {"text": "Dr. Maria Santos", "type": "name", "privacy_mask_labels": ["FIRSTNAME", "SURNAME"]},
        {"text": "maria.santos", "type": "reference"}
      ],
      "entity_type": "person"
    }
  ]
}
```

## Supported PII Labels

| Label | Description |
|-------|-------------|
| `FIRSTNAME` | Given names |
| `SURNAME` | Family names |
| `EMAIL` | Email addresses |
| `PHONENUMBER` | Phone numbers |
| `DATEOFBIRTH` | Birth dates |
| `SSN` | Social Security Numbers |
| `CREDITCARDNUMBER` | Credit card numbers |
| `STREET` | Street addresses |
| `BUILDINGNUM` | Building/house numbers |
| `CITY` | City names |
| `STATE` | State/province names |
| `ZIP` | ZIP/postal codes |
| `COUNTRY` | Country names |
| `IBAN` | International Bank Account Numbers |
| `DRIVERLICENSENUM` | Driver's license numbers |
| `PASSPORTID` | Passport ID |
| `NATIONALID` | National ID |
| `TAXNUM` | Tax identification numbers |
| `COMPANYNAME` | Company/organization names |
| `URL` | Website URLs |
| `USERNAME` | Usernames |
| `PASSWORD` | Passwords |
| `LICENSEPLATENUM` | Vehicle license plates |
| `AGE` | Age values |

## Generation Pipelines

### Doubleword Batch Pipeline

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Generate NER    │───►│  Automatic Poll  │───►│  Generate Coref  │
│  Batch Requests  │    │  & Download      │    │  Batch Requests  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
         │                                                │
         ▼                                                ▼
    Submit to API                                    Submit to API
         │                                                │
         ▼                                                ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Automatic Poll  │───►│  Review (opt)    │───►│  Process Results │
│  & Download      │    │  Quality Check   │    │  & Convert       │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                                                          │
                                                          ▼
                                                 ┌──────────────────┐
                                                 │  Label Studio    │
                                                 │  Samples         │
                                                 └──────────────────┘
```

**Features:**
- Fully automated with `--auto_poll`
- Resumable at any stage
- State persistence across runs
- Parallel request generation
- Optional review stage with `--enable_review`

### OpenAI Direct Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Generate      │───►│   Review        │───►│   Save          │
│   (API call)    │    │   (API call)    │    │   (JSON file)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                       │
        ▼                      ▼                       ▼
   data_samples/         data_samples/          data_samples/
     samples/          reviewed_samples/     annotation_samples/
```

**Steps:**
1. **Generate**: OpenAI creates text with embedded PII and annotations
2. **Review**: Second API call validates and corrects labels
3. **Convert**: Transform to Label Studio format
4. **Save**: Final samples saved to `data_samples/annotation_samples/`

## Troubleshooting

### SSL Errors
```
[SSL: WRONG_VERSION_NUMBER] wrong version number
```
Use `http://` not `https://` for local servers.

### 404 Not Found
```
Client error '404 Not Found' for url 'http://server:8000'
```
Include the full API path: `http://server:8000/v1/chat/completions`

### Empty Responses
If samples fail with JSON parse errors, ensure:
- Server supports structured outputs (`guided_json`)
- Model supports chat completions format

### Rate Limiting
If throughput drops or errors increase:
```bash
# Reduce parallel workers
--max_workers=8
```

## Related Documentation

- **[DOUBLEWORD_QUICKSTART.md](./doubleword/DOUBLEWORD_QUICKSTART.md)** - Complete guide to the Doubleword pipeline
- [Label Studio README](./labelstudio/README.md) - Label Studio integration guide
- [Main README](../../README.md) - Project overview
