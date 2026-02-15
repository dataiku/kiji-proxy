# Chapter 7: Customizing the PII Model

Kiji ships with a pre-trained PII detection model, but you can train your own to add new entity types, improve accuracy for your domain, or support additional languages. This chapter walks through the full customization workflow: generating training data, reviewing it in Label Studio, running the training pipeline, and loading your custom model into the app.

## Overview

The customization workflow has four stages:

```
1. Generate Data  ──►  2. Review in Label Studio  ──►  3. Train with Metaflow  ──►  4. Load in App
```

## Generating training data

There are two ways to get training data into the pipeline.

### Option 1: Generate synthetic data with an LLM

Kiji includes tooling to generate synthetic PII samples using either the Doubleword batch API or the OpenAI API. Both produce Label Studio-compatible JSON files containing text with annotated PII entities.

The generation prompts and entity definitions live in `model/dataset/label_utils.py`. The `LABEL_DESCRIPTIONS` dictionary is the single source of truth for all PII labels. Each entry defines the label name, color, selection probability, and generation hints. To add a new entity type or adjust how existing types are generated, edit this dictionary.

For example, to add a `MEDICAL_RECORD_NUMBER` entity type:

```python
"MEDICAL_RECORD_NUMBER": {
    "name": "Medical Record Number",
    "color": "#00acc1",
    "chance": 0.1,
    "hints": ["Use formats like MRN-123456 or various hospital ID formats"],
},
```

After modifying the labels, regenerate the Label Studio configuration so it includes your new entity types:

```bash
uv run python -c "from model.dataset.label_utils import LabelUtils; LabelUtils.update_labelstudio_config()"
```

#### Doubleword (recommended for large datasets)

The Doubleword pipeline automates the full generation workflow with batch processing, automatic polling, and resumability:

```bash
export DOUBLEWORD_API_KEY="your-api-key"

# Generate 500 samples
uv run python -m model.dataset.doubleword.pipeline \
  --command=start \
  --num_samples=500

# Enable an optional review stage for higher quality
uv run python -m model.dataset.doubleword.pipeline \
  --command=start \
  --num_samples=500 \
  --enable_review

# Check progress
uv run python -m model.dataset.doubleword.pipeline --command=status

# Resume after interruption
uv run python -m model.dataset.doubleword.pipeline --command=resume
```

Generated samples are saved to `model/dataset/data_samples/annotation_samples/`.

<!-- Screenshot: Terminal output showing Doubleword pipeline progress -->

#### OpenAI (for quick testing and small datasets)

For fast iteration, use the OpenAI direct API:

```bash
export OPENAI_API_KEY="your-api-key"

# Generate 10 samples
uv run python -m model.dataset.openai.training_set --num_samples=10

# Use a custom OpenAI-compatible API server
export URL=http://your-server:8000/v1/chat/completions
uv run python -m model.dataset.openai.training_set --num_samples=100 --api_url=$URL

# Increase parallelism for faster generation
uv run python -m model.dataset.openai.training_set --num_samples=200 --max_workers=12
```

Generated samples are saved to `model/dataset/data_samples/annotation_samples/`.

<!-- Screenshot: Example of a generated JSON sample file -->

### Option 2: Use a HuggingFace dataset

You can bootstrap your training data from an existing HuggingFace dataset. The public dataset `dataiku/kiji-pii-training-data` contains pre-labeled samples ready for training or further annotation.

Download and convert the dataset to local Label Studio JSON files:

```bash
# Download all splits to the default training_samples directory
uv run python model/dataset/huggingface/download_dataset_from_hf.py \
  --repo-id "dataiku/kiji-pii-training-data"

# Download to a custom directory
uv run python model/dataset/huggingface/download_dataset_from_hf.py \
  --repo-id "dataiku/kiji-pii-training-data" \
  --output-dir path/to/output

# Download only specific splits
uv run python model/dataset/huggingface/download_dataset_from_hf.py \
  --repo-id "dataiku/kiji-pii-training-data" \
  --splits train
```

Each row is converted to an individual JSON file in the Label Studio format that the training pipeline consumes directly.

To expand the dataset with your own samples, generate additional data using Option 1 and merge them into the same directory. You can also upload your expanded dataset back to HuggingFace:

```bash
export HF_TOKEN=hf_xxxxx

uv run python model/dataset/huggingface/upload_dataset_to_hf.py \
  --repo-id "your-org/kiji-pii-training-data" \
  --public
```

## Reviewing data in Label Studio

Regardless of how you generated your data, you should review it in Label Studio before training. This lets you correct mislabeled entities, add missing annotations, and ensure quality.

### Install and start Label Studio

```bash
# Install Label Studio dependencies
uv sync --extra labelstudio

# Start Label Studio
uv run label-studio start
```

Label Studio will open in your browser at `http://localhost:8080`.

<!-- Screenshot: Label Studio welcome screen -->

### Create a project

1. Click **Create Project** on the welcome screen.
2. Give the project a descriptive name (e.g., "Kiji PII - Custom Entities").
3. Skip the Data Import tab for now (we will import via script).
4. On the **Labeling Setup** tab, click **Custom template** at the bottom of the template list.
5. Paste the contents of `model/dataset/labelstudio/LabelingConfig.txt` into the code editor. This configuration is auto-generated from `label_utils.py` and includes all your entity types with color coding.
6. Click **Save**.

<!-- Screenshot: Label Studio project creation with custom labeling config -->

### Import data

Use the import script to load your generated samples with pre-annotations:

```bash
# Set environment variables
export LABEL_STUDIO_API_KEY='your-api-key'   # From Label Studio > Account & Settings > Personal Access Token
export LABEL_STUDIO_PROJECT_ID='your-project-id'  # From the project URL (e.g., /projects/3/)

# Import annotation samples
cd model/dataset/labelstudio
python import_predictions.py
```

The script imports all JSON files from `model/dataset/data_samples/annotation_samples/` with their pre-annotations, so you can review and correct them rather than labeling from scratch.

<!-- Screenshot: Label Studio data manager showing imported tasks with pre-annotations -->

### Review and correct annotations

1. Click **Label All Tasks** to enter the labeling workflow.
2. Review the pre-annotated entities. Click on labels to select them, then highlight text to apply them.
3. For coreference relations, select a pronoun region, click the "create relation" button, then click the noun it refers to.
4. Click **Submit** to save and move to the next task.

<!-- Screenshot: Label Studio annotation interface showing PII entities and coreference relations -->

### Export reviewed annotations

After reviewing, export the corrected annotations to the training samples directory:

```bash
export LABEL_STUDIO_API_KEY='your-api-key'
export LABEL_STUDIO_PROJECT_ID='your-project-id'

cd model/dataset/labelstudio
python export_annotations.py
```

The exported samples are saved to `model/dataset/data_samples/training_samples/` in the format the training pipeline expects.

## Training the model with Metaflow

The training pipeline is orchestrated by Metaflow and configured via `model/flows/training_config.toml`.

### Configure the pipeline

Edit `model/flows/training_config.toml` to adjust training parameters:

```toml
[model]
name = "distilbert-base-cased"

[training]
num_epochs = 15
batch_size = 16
learning_rate = 3e-5
early_stopping_enabled = true
early_stopping_patience = 3
early_stopping_threshold = 0.01

[paths]
training_samples_dir = "model/dataset/data_samples/training_samples"
output_dir = "model/trained"

[pipeline]
skip_export = true        # true if you already exported from Label Studio manually
skip_quantization = false  # ONNX quantization (Linux only)
skip_signing = false       # Cryptographic model signing
```

Key settings:
- **`skip_export`**: Set to `true` if you already exported from Label Studio (Step 2). Set to `false` to have the pipeline export directly from Label Studio (requires `labelstudio.project_id` and `LABEL_STUDIO_API_KEY`).
- **`skip_quantization`**: ONNX quantization only works on Linux. Set to `true` on macOS.
- **`training_samples_dir`**: Points to the directory containing your reviewed training samples.

### Run the pipeline

```bash
# Run from the project root
./model/flows/run_training.sh

# Or run directly with uv
uv run --extra training --extra signing \
  python model/flows/training_pipeline.py run

# With ONNX quantization (Linux only)
uv run --extra training --extra quantization --extra signing \
  python model/flows/training_pipeline.py run

# Use a custom config file
./model/flows/run_training.sh --config custom_config.toml
```

The pipeline executes the following steps:

1. **Export data** (optional) - Pulls annotations from Label Studio via its SDK
2. **Preprocess data** - Loads JSON samples, tokenizes text, aligns labels
3. **Train model** - Multi-task learning for PII detection and coreference resolution
4. **Evaluate model** - Runs inference on test cases and reports metrics
5. **Quantize model** - Exports to ONNX and applies quantization (Linux only)
6. **Sign model** - Generates a cryptographic signature for integrity verification

<!-- Screenshot: Terminal output showing Metaflow pipeline steps completing -->

### Pipeline outputs

After training completes, you will find:

```
model/
├── trained/          # Full trained model (PyTorch)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── label_mappings.json
├── quantized/        # Quantized ONNX model (smaller, faster)
│   ├── model_quantized.onnx
│   ├── tokenizer.json
│   └── label_mappings.json
└── quantized.sig     # Cryptographic signature
```

### Upload your model to HuggingFace (optional)

Share your trained model via HuggingFace Hub:

```bash
export HF_TOKEN=hf_xxxxx

# Upload the quantized ONNX model
uv run python model/dataset/huggingface/upload_model_to_hf.py \
  --variant quantized \
  --repo-id "your-org/kiji-pii-model-onnx" \
  --dataset-repo-id "your-org/kiji-pii-training-data" \
  --public
```

## Loading your custom model in the app

Once you have a trained model, you can load it into the Kiji desktop app through the Advanced Settings modal.

### Model directory requirements

Your model directory must contain these three files:

- `model_quantized.onnx` - The quantized ONNX model
- `tokenizer.json` - The tokenizer configuration
- `label_mappings.json` - The label-to-ID mappings

These are produced automatically by the training pipeline in the `model/quantized/` directory.

### Load the model

1. Open the Kiji desktop app.
2. Open the hamburger menu and click **Advanced Settings**.
3. In the **Load Custom Kiji PII Model** section, enter the path to your model directory or click **Browse** to select it.
4. Click **Reload Model**.
5. The status indicator will show **Healthy** if the model loaded successfully.

<!-- Screenshot: Advanced Settings modal showing the Load Custom Kiji PII Model section with a healthy model loaded -->

### Adjust detection sensitivity

After loading a custom model, you may want to adjust the PII detection sensitivity. In the same Advanced Settings modal:

- **Low** (0.1 threshold) - Catches more potential PII but may have more false positives
- **Medium** (0.25 threshold) - Balanced detection (default)
- **High** (0.5 threshold) - More precise but may miss some PII

<!-- Screenshot: Advanced Settings modal showing the PII Detection Sensitivity selector -->

## End-to-end example

Here is a complete walkthrough of adding a new `MEDICAL_RECORD_NUMBER` entity type:

1. **Add the label** to `model/dataset/label_utils.py`:
   ```python
   "MEDICAL_RECORD_NUMBER": {
       "name": "Medical Record Number",
       "color": "#00acc1",
       "chance": 0.1,
       "hints": ["Use formats like MRN-123456 or various hospital ID formats"],
   },
   ```

2. **Regenerate the Label Studio config**:
   ```bash
   uv run python -c "from model.dataset.label_utils import LabelUtils; LabelUtils.update_labelstudio_config()"
   ```

3. **Generate training data** (using OpenAI for a quick test):
   ```bash
   export OPENAI_API_KEY="your-key"
   uv run python -m model.dataset.openai.training_set --num_samples=100
   ```

4. **Review in Label Studio**:
   ```bash
   uv sync --extra labelstudio
   uv run label-studio start
   # Create a project, import data, review annotations, export
   ```

5. **Train the model**:
   ```bash
   ./model/flows/run_training.sh
   ```

6. **Load in the app**: Open Advanced Settings, browse to `model/quantized/`, and click Reload Model.

## Next steps

- [Chapter 2: Development Guide](02-development-guide.md) - Set up your development environment
- [Chapter 5: Advanced Topics](05-advanced-topics.md) - Model signing and security
- [Dataset README](../model/dataset/README.md) - Detailed data generation reference
- [Label Studio README](../model/dataset/labelstudio/README.md) - Full Label Studio integration guide
