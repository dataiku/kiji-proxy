# PII Training Data Generation

This directory contains tools for generating synthetic PII training data using LLMs with structured outputs.

## 📁 Directory Structure

```
model/dataset/
├── pipeline.py                  # 🆕 Automated dataset generation pipeline (Doubleword)
├── doubleword_client.py         # 🆕 Doubleword API client
├── batch_generator.py           # 🆕 Batch request generation
├── batch_monitor.py             # 🆕 Automatic polling & monitoring
├── result_processor.py          # 🆕 Result processing
├── pipeline_state.py            # 🆕 State management for resumability
│
├── training_set.py              # Direct LLM generation (OpenAI/vLLM)
├── upload_to_hf.py              # Upload samples to HuggingFace Hub
├── api_clients.py               # LLM client implementations (OpenAI, Ollama, vLLM)
├── prompts.py                   # Prompt templates for generation/review
├── schemas.py                   # JSON schemas for structured outputs
├── label_utils.py               # PII label definitions and utilities
├── file_operations.py           # File I/O utilities
├── tokenization.py              # Tokenization for training samples
├── to_labelstudio.py            # Label Studio format conversion
│
├── samples/                     # Raw generated samples
├── reviewed_samples/            # LLM-reviewed and corrected samples
└── annotation_samples/          # Label Studio-ready samples
```

## 🚀 Quick Start

### Option 1: Automated Pipeline (Doubleword Batch API) - Recommended for Large Datasets

The new pipeline automates the entire workflow with automatic polling and resumability:

```bash
# Set API key
export DOUBLEWORD_API_KEY="your-api-key"

# Generate 100 samples - fully automated!
python -m model.dataset.pipeline \
  --command=start \
  --num_samples=100

# Check status anytime
python -m model.dataset.pipeline --command=status

# Resume if interrupted
python -m model.dataset.pipeline --command=resume
```

**See [PIPELINE_README.md](./PIPELINE_README.md) for complete documentation.**

### Option 2: Direct Generation (OpenAI/vLLM) - For Quick Testing

```bash
# Generate 100 samples using OpenAI API
uv run python model/dataset/training_set.py --num_samples=100

# Generate using a remote vLLM server (OpenAI-compatible API)
export URL=http://your-vllm-server:8000/v1/chat/completions
uv run python model/dataset/training_set.py --num_samples=1000 --api_url=$URL

# High-throughput generation with parallel workers
uv run python model/dataset/training_set.py --num_samples=10000 --api_url=$URL --max_workers=250
```

### Upload to HuggingFace

```bash
export HF_TOKEN=hf_xxxxx

# Upload as private dataset
python model/dataset/upload_to_hf.py --repo-id "username/pii-training-data"

# Upload as public dataset
python model/dataset/upload_to_hf.py --repo-id "username/pii-training-data" --public
```

## 🎯 Which Generation Method to Use?

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Pipeline (Doubleword)** | Large datasets (100+) | Automated, resumable, cost-effective batch pricing | Requires waiting for batch completion |
| **Direct (OpenAI/vLLM)** | Quick testing, small datasets | Immediate results, real-time feedback | Higher cost per sample, requires running server |

**Recommendation:** Use the **pipeline** for production datasets (>100 samples), use **direct generation** for testing and iteration.

## ⚙️ Configuration Options

### pipeline.py (Doubleword Batch API)

| Flag | Default | Description |
|------|---------|-------------|
| `--command` | `start` | Command: start, status, resume, reset, cancel |
| `--num_samples` | 100 | Number of samples to generate |
| `--api_model` | `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` | Model name for generation |
| `--[no]auto_poll` | `true` | Automatically wait for batch completion (use `--noauto_poll` to disable) |
| `--poll_interval` | 60 | Seconds between status checks |
| `--output_dir` | `model/dataset` | Output directory for samples |
| `--max_workers` | auto | Parallel workers (default: min(32, num_samples + 4)) |
| `--log_level` | INFO | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |

### training_set.py (Direct Generation)

| Flag | Default | Description |
|------|---------|-------------|
| `--num_samples` | 5 | Number of samples to generate |
| `--api_url` | None | Custom API URL for vLLM or other OpenAI-compatible servers |
| `--use_ollama` | False | Use local Ollama instead of OpenAI |
| `--output_dir` | `model/dataset` | Output directory for samples |
| `--max_workers` | auto | Parallel workers (default: min(12, num_samples + 4)) |
| `--log_level` | WARNING | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |

### upload_to_hf.py

| Flag | Default | Description |
|------|---------|-------------|
| `--repo-id` | (required) | HuggingFace repo ID (e.g., `username/dataset-name`) |
| `--samples-dir` | `model/dataset/training_samples` | Directory containing JSON samples |
| `--public` | False | Make dataset public (default: private) |

## 🔧 Backend Options

### Doubleword Batch API (Recommended)

Cost-effective batch processing with automatic queuing:

```bash
export DOUBLEWORD_API_KEY="your-key"

python -m model.dataset.pipeline \
  --command=start \
  --num_samples=1000 \
  --api_model="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
```

**Advantages:**
- No infrastructure setup required
- Batch pricing (lower cost per sample)
- Automatic queuing and rate limiting
- Resumable from any point

**See [PIPELINE_README.md](./PIPELINE_README.md) for complete guide.**

### vLLM Backend (For Direct Generation)

For large-scale direct generation, use a vLLM server with GPT-OSS or similar models:

#### 1. Start vLLM Server

```bash
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --port 8000
```

#### 2. Generate Data

```bash
export URL=http://server-ip:8000/v1/chat/completions

# Start with small batch to verify
uv run python model/dataset/training_set.py --num_samples=10 --api_url=$URL

# Scale up with parallel workers
uv run python model/dataset/training_set.py --num_samples=100000 --api_url=$URL --max_workers=500
```

#### 3. Monitor Server

Watch vLLM server logs for throughput metrics:
```
Engine 000: Avg prompt throughput: 4418.9 tokens/s, Avg generation throughput: 4981.0 tokens/s,
Running: 248 reqs, Waiting: 0 reqs, GPU KV cache usage: 3.1%
```

**Tuning tips:**
- Keep KV cache usage under 80%
- Match `--max_workers` to server's `--max-num-seqs` (default: 1024)
- If waiting queue grows, reduce workers

## 📊 Sample Format

Generated samples follow this JSON structure:

```json
{
  "text": "Contact Dr. Maria Santos at maria.santos@hospital.org or call +1-555-123-4567.",
  "privacy_mask": [
    {"value": "Maria", "label": "FIRSTNAME"},
    {"value": "Santos", "label": "SURNAME"},
    {"value": "maria.santos@hospital.org", "label": "EMAIL"},
    {"value": "+1-555-123-4567", "label": "PHONENUMBER"}
  ],
  "coreferences": [
    {
      "cluster_id": 0,
      "mentions": ["Dr. Maria Santos", "maria.santos"],
      "entity_type": "person"
    }
  ]
}
```

## 🏷️ Supported PII Labels

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

## 🔄 Generation Pipelines

### Automated Pipeline (Doubleword)

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
│  Automatic Poll  │───►│  Process Results │───►│  Label Studio    │
│  & Download      │    │  & Convert       │    │  Samples         │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

**Features:**
- ✅ Fully automated with `--auto_poll`
- ✅ Resumable at any stage
- ✅ State persistence across runs
- ✅ Parallel request generation

### Direct Generation Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Generate      │───►│   Review        │───►│   Save          │
│   (LLM call)    │    │   (LLM call)    │    │   (JSON file)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │
        ▼                      ▼
   Structured JSON        Corrected JSON
   with PII labels        with validated labels
```

**Steps:**
1. **Generate**: LLM creates text with embedded PII and annotations
2. **Review**: Second LLM call validates and corrects labels
3. **Save**: Final sample saved to `reviewed_samples/`

## 🛠️ Troubleshooting

### SSL Errors
```
[SSL: WRONG_VERSION_NUMBER] wrong version number
```
Use `http://` not `https://` for vLLM servers.

### 404 Not Found
```
Client error '404 Not Found' for url 'http://server:8000'
```
Include the full API path: `http://server:8000/v1/chat/completions`

### Empty Responses
If samples fail with JSON parse errors, ensure:
- vLLM server supports structured outputs (`guided_json`)
- Model supports chat completions format

### Rate Limiting
If throughput drops or errors increase:
```bash
# Reduce parallel workers
--max_workers=100
```

## 📚 Related Documentation

- **[PIPELINE_README.md](./PIPELINE_README.md)** - Complete guide to the automated pipeline (Doubleword)
- [Metaflow Training Pipeline](../flows/README.md) - Kubernetes-based model training
- [Trained Model](../trained/README.md) - Model files and serving
- [Main README](../../README.md) - Project overview

## 🔄 Migration Guide

If you're using the old manual workflow with `training_set_doubleword*.py` scripts:

### Old Workflow (Deprecated)
```bash
python model/dataset/training_set_doubleword.py --num_samples=100
# Manually download NER results...
python model/dataset/training_set_doubleword_coref.py --ner_results_file=results.jsonl
# Manually download coref results...
python model/dataset/training_set_doubleword_result_processing.py --batch_output_file_ids=id1,id2
```

### New Workflow (Recommended)
```bash
python -m model.dataset.pipeline --command=start --num_samples=100
```

**Benefits:**
- ✅ No manual downloads required
- ✅ Automatic polling and state management
- ✅ Resumable from any point
- ✅ Single command operation
- ✅ Better error handling and logging

The old scripts still work but will be removed in a future version. Please migrate to the new pipeline.
