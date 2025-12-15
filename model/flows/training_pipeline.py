"""
Metaflow pipeline for PII detection model training and deployment.

This pipeline orchestrates the complete ML workflow:
1. Dataset loading and preprocessing
2. Model training with multi-task learning
3. Model evaluation
4. Model quantization (ONNX)
5. Model signing (cryptographic hash)
6. Model upload (Hugging Face Hub) - optional
7. Trigger DMG rebuild - optional

Usage:
    # Run entire pipeline locally (uses training_config.toml defaults)
    python model/flows/training_pipeline.py run

    # Run with custom config file (Config is specified BEFORE run command)
    python model/flows/training_pipeline.py \
        --config config-file custom_config.toml \
        run

    # Override specific parameters at runtime (Parameters are AFTER run)
    python model/flows/training_pipeline.py run \
        --num-epochs 3 \
        --batch-size 32

    # Quick test run (skip quantization, signing, upload)
    python model/flows/training_pipeline.py run \
        --num-epochs 1 \
        --skip-quantization \
        --skip-signing

    # View latest run
    python model/flows/training_pipeline.py show latest
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Suppress tokenizers parallelism warning (fork after parallelism)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from metaflow import Config, FlowSpec, IncludeFile, Parameter, card, resources, kubernetes, pypi, step, checkpoint, model, current, retry

# Core dependencies for all steps
CORE_PACKAGES = {
    "torch": ">=2.0.0",
    "transformers": ">=4.20.0",
    "numpy": ">=1.21.0",
    "safetensors": ">=0.3.0",
    "datasets": ">=2.0.0",  # HuggingFace datasets for preprocessing
    "scikit-learn": ">=1.0.0",  # For metrics computation
    "accelerate": ">=0.26.0",  # For HuggingFace Trainer
    "absl-py": ">=2.0.0",  # Google abseil for flags/logging
    "python-dotenv": ">=1.0.0",  # For environment variable loading
    "tqdm": ">=4.64.0",  # Progress bars
    "httpx": ">=0.24.0",  # HTTP client
}

# Additional packages for quantization
QUANT_PACKAGES = {
    **CORE_PACKAGES,
    "optimum[onnxruntime]": ">=1.15.0",
    "onnx": ">=1.15.0",
    "onnxruntime": ">=1.16.0",
    "onnxscript": ">=0.1.0",  # Required for torch.onnx.export
}

# Add project root to path (flow is at project root)
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class PIITrainingPipeline(FlowSpec):
    """
    End-to-end ML pipeline for PII detection model training.

    This flow coordinates:
    - Dataset preprocessing and validation
    - Model training with multi-task learning (PII + co-reference)
    - Model evaluation and metrics tracking
    - Model quantization for production
    - Model signing for security
    - Optional: Upload to Hugging Face Hub

    Configuration is loaded from training_config.toml by default.
    Use --config-file to specify a different config file.
    Use Parameters to override config values at runtime.
    """

    # Configuration file (deployment-time defaults)
    # Uses TOML parser to load training hyperparameters
    config_file = Config(
        "config-file",
        default="training_config.toml",
        parser="tomllib.loads",
        help="TOML config file with training hyperparameters",
    )

    # Runtime overrides (these take precedence over config file)
    num_epochs = Parameter(
        "num-epochs",
        help="Number of training epochs (overrides config)",
        default=None,
        type=int,
    )

    batch_size = Parameter(
        "batch-size",
        help="Training batch size (overrides config)",
        default=None,
        type=int,
    )

    learning_rate = Parameter(
        "learning-rate",
        help="Learning rate for training (overrides config)",
        default=None,
        type=float,
    )

    model_name = Parameter(
        "model-name",
        help="Base model to fine-tune (overrides config)",
        default=None,
        type=str,
    )

    training_samples_dir = Parameter(
        "training-samples-dir",
        help="Directory containing training samples (overrides config)",
        default=None,
        type=str,
    )

    output_dir = Parameter(
        "output-dir",
        help="Directory to save trained model (overrides config)",
        default=None,
        type=str,
    )

    # Dataset file (IncludeFile reads and packages the file for remote execution)
    dataset_zip = IncludeFile(
        "dataset-zip",
        help="Path to dataset.zip containing training samples",
        default="dataset.zip",
        is_text=False,  # Keep as bytes for zip file
    )

    # Skip flags (runtime only)
    skip_quantization = Parameter(
        "skip-quantization",
        help="Skip ONNX quantization step",
        default=False,
        type=bool,
    )

    skip_signing = Parameter(
        "skip-signing",
        help="Skip model signing step",
        default=False,
        type=bool,
    )

    skip_hf_upload = Parameter(
        "skip-hf-upload",
        help="Skip Hugging Face upload",
        default=True,
        type=bool,
    )

    def _get_effective_config(self):
        """
        Merge config file values with parameter overrides.
        Parameters take precedence over config file values.
        """
        cfg = self.config_file  # Parsed TOML dict

        # Helper to get value with parameter override
        def get_val(param_val, *cfg_keys, default=None):
            if param_val is not None:
                return param_val
            # Navigate nested config
            val = cfg
            for key in cfg_keys:
                if isinstance(val, dict) and key in val:
                    val = val[key]
                else:
                    return default
            return val if val != cfg else default

        return {
            "model_name": get_val(
                self.model_name, "model", "name", default="distilbert-base-cased"
            ),
            "num_epochs": get_val(self.num_epochs, "training", "num_epochs", default=5),
            "batch_size": get_val(self.batch_size, "training", "batch_size", default=16),
            "learning_rate": get_val(
                self.learning_rate, "training", "learning_rate", default=3e-5
            ),
            "training_samples_dir": get_val(
                self.training_samples_dir,
                "paths",
                "training_samples_dir",
                default="model/dataset/reviewed_samples",
            ),
            "output_dir": get_val(
                self.output_dir, "paths", "output_dir", default="model/trained"
            ),
        }

    @pypi(packages=CORE_PACKAGES, python="3.10")
    @step
    def start(self):
        """Initialize the pipeline and set up configuration."""
        import io
        import zipfile

        from src.config import EnvironmentSetup, TrainingConfig

        print("=" * 80)
        print("PII Detection Model Training Pipeline")
        print("=" * 80)

        # Disable wandb
        EnvironmentSetup.disable_wandb()

        # Check GPU availability
        EnvironmentSetup.check_gpu()

        # Merge config file with parameter overrides
        effective_config = self._get_effective_config()

        # Extract dataset from IncludeFile (works both locally and on remote execution)
        samples_dir = Path(effective_config["training_samples_dir"])
        if not samples_dir.exists() or not list(samples_dir.glob("*.json")):
            if self.dataset_zip:
                print(f"\nExtracting training data from included dataset.zip...")
                # IncludeFile provides bytes, wrap in BytesIO for zipfile
                zip_bytes = io.BytesIO(self.dataset_zip)
                with zipfile.ZipFile(zip_bytes, "r") as zf:
                    zf.extractall(".")
                print(f"  Extracted to {samples_dir}")
            else:
                print(f"\nWarning: No training data found at {samples_dir}")
                print("  Please provide --dataset-zip or set --training-samples-dir")

        # Create training configuration from merged values
        self.config = TrainingConfig(
            model_name=effective_config["model_name"],
            num_epochs=effective_config["num_epochs"],
            batch_size=effective_config["batch_size"],
            learning_rate=effective_config["learning_rate"],
            training_samples_dir=effective_config["training_samples_dir"],
            output_dir=effective_config["output_dir"],
        )

        # Store metadata
        self.pipeline_start_time = datetime.utcnow().isoformat()

        # Show config source info
        print(f"\nConfiguration (from config file + parameter overrides):")
        print(f"  Config file: {self.config_file}")
        print(f"  Model: {self.config.model_name}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch Size: {self.config.batch_size}")
        print(f"  Learning Rate: {self.config.learning_rate}")
        print(f"  Training Samples: {self.config.training_samples_dir}")
        print(f"  Output Dir: {self.config.output_dir}")

        self.next(self.preprocess_data)

    @pypi(packages=CORE_PACKAGES, python="3.10")
    @resources(memory=8000, cpu=4)
    @step
    def preprocess_data(self):
        """Load and preprocess training data."""
        from src.preprocessing import DatasetProcessor

        print("\n" + "=" * 80)
        print("Step 1: Data Preprocessing")
        print("=" * 80)

        # Initialize processor
        processor = DatasetProcessor(self.config)

        # Load and process datasets
        print(f"Loading data from {self.config.training_samples_dir}...")
        train_dataset, val_dataset, mappings, coref_info = processor.prepare_datasets()

        self.train_size = len(train_dataset)
        self.val_size = len(val_dataset)

        # Store datasets and mappings as artifacts
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.label_mappings = mappings
        self.coref_info = coref_info

        print(f"\nDataset Summary:")
        print(f"  Training samples: {self.train_size}")
        print(f"  Validation samples: {self.val_size}")

        self.next(self.train_model)

    @retry(times=2)
    @checkpoint
    @pypi(packages=CORE_PACKAGES, python="3.10")
    # @resources(memory=16000, cpu=8, gpu=1)
    @kubernetes(gpu=1, memory=16000, cpu=8, compute_pool="gpu-pool-training") # obp-nebius-h100-1gpu
    @step
    def train_model(self):
        """Train the multi-task PII detection model."""
        from src.trainer import PIITrainer

        print("\n" + "=" * 80)
        print("Step 2: Model Training")
        print("=" * 80)

        # Check if resuming from a checkpoint (e.g., after retry)
        if current.checkpoint.is_loaded:
            print("Resuming from checkpoint...")
            # TODO: Load model state from current.checkpoint.directory for mid-training recovery
            # For now, we checkpoint at the end of training, so retry starts fresh

        # Initialize trainer
        trainer_instance = PIITrainer(self.config)
        trainer_instance.load_label_mappings(self.label_mappings, self.coref_info)
        trainer_instance.initialize_model()

        # Train model
        print("\nStarting training...")
        start_time = time.time()
        trained_trainer = trainer_instance.train(self.train_dataset, self.val_dataset)
        training_time = time.time() - start_time

        print(f"\nTraining completed in {training_time / 60:.1f} minutes")

        # Evaluate and extract metrics
        results = trainer_instance.evaluate(self.val_dataset, trained_trainer)

        self.training_metrics = {
            "training_time_seconds": training_time,
            "eval_pii_f1_weighted": results.get("eval_pii_f1_weighted"),
            "eval_pii_f1_macro": results.get("eval_pii_f1_macro"),
            "eval_pii_precision_weighted": results.get("eval_pii_precision_weighted"),
            "eval_pii_recall_weighted": results.get("eval_pii_recall_weighted"),
            "eval_coref_f1_weighted": results.get("eval_coref_f1_weighted"),
            "eval_coref_f1_macro": results.get("eval_coref_f1_macro"),
        }

        self.model_path = self.config.output_dir

        # Ensure label_mappings.json is in the model directory before checkpointing
        # (preprocessing saves it to output_dir, but K8s step may not have it)
        model_dir = Path(self.model_path)
        mappings_path = model_dir / "label_mappings.json"
        if not mappings_path.exists():
            print(f"Writing label_mappings.json to {mappings_path}")
            with mappings_path.open("w") as f:
                json.dump(self.label_mappings, f, indent=2)

        # Save model checkpoint using Metaflow's checkpoint system
        # This persists the model across steps (K8s -> local or remote)
        print(f"\nSaving model checkpoint from: {self.model_path}")
        if model_dir.exists():
            self.trained_model = current.checkpoint.save(
                str(model_dir),
                metadata={
                    "pii_f1_weighted": self.training_metrics.get("eval_pii_f1_weighted"),
                    "pii_f1_macro": self.training_metrics.get("eval_pii_f1_macro"),
                    "training_time_seconds": training_time,
                },
                name="trained_model",
                latest=True,
            )
            print(f"  Model checkpoint saved: {self.trained_model}")
        else:
            print(f"  Warning: Model directory not found at {self.model_path}")
            self.trained_model = None

        print(f"\nModel saved to: {self.model_path}")
        print(f"PII F1 (weighted): {self.training_metrics.get('eval_pii_f1_weighted', 'N/A'):.4f}")

        self.next(self.evaluate_model)

    @model(load="trained_model")
    @pypi(packages=CORE_PACKAGES, python="3.10")
    @resources(memory=8000, cpu=4)
    @step
    def evaluate_model(self):
        """Evaluate the trained model on test cases."""
        from src.eval_model import PIIModelLoader

        print("\n" + "=" * 80)
        print("Step 3: Model Evaluation")
        print("=" * 80)

        # Load model from Metaflow checkpoint
        # The @model decorator loads the checkpoint to current.model.loaded["trained_model"]
        model_checkpoint_path = current.model.loaded["trained_model"]
        print(f"Loading model from checkpoint: {model_checkpoint_path}")

        # Load model
        loader = PIIModelLoader(model_checkpoint_path)
        loader.load_model()

        # Run evaluation on test cases
        test_cases = [
            "My name is John Smith and my email is john@example.com",
            "Call me at 555-123-4567 or email sarah.miller@company.com",
            "SSN: 123-45-6789, DOB: 01/15/1990",
            "I live at 123 Main Street, Springfield, IL 62701",
            "Dr. Emily Chen can be reached at emily.chen@hospital.com",
        ]

        evaluation_results = []
        inference_times = []

        for text in test_cases:
            entities, coref_clusters, inference_time = loader.predict(text)
            evaluation_results.append({
                "text": text,
                "num_entities": len(entities),
                "num_clusters": len(coref_clusters),
                "inference_time_ms": inference_time,
            })
            inference_times.append(inference_time)

        self.evaluation_results = evaluation_results
        self.avg_inference_time_ms = sum(inference_times) / len(inference_times)

        print(f"\nEvaluation Summary:")
        print(f"  Test cases processed: {len(test_cases)}")
        print(f"  Average inference time: {self.avg_inference_time_ms:.2f} ms")
        print(f"  Throughput: {1000 / self.avg_inference_time_ms:.1f} texts/second")

        self.next(self.quantize_model)

    @retry(times=2)
    @checkpoint
    @model(load="trained_model")
    @pypi(packages=QUANT_PACKAGES, python="3.10")
    @kubernetes(memory=10000, cpu=6, compute_pool="c5-2x-task")  # Run on K8s to avoid local onnxruntime issues
    @step
    def quantize_model(self):
        """Quantize model to ONNX format for production deployment."""
        if self.skip_quantization:
            print("\nSkipping model quantization")
            self.quantized_model_path = None
            self.quantized_model = None
            self.next(self.sign_model)
            return

        print("\n" + "=" * 80)
        print("Step 4: Model Quantization")
        print("=" * 80)

        from src.quantitize import (
            export_to_onnx,
            load_multitask_model,
            quantize_model,
        )

        try:
            # Load model from checkpoint
            model_checkpoint_path = current.model.loaded["trained_model"]
            print(f"Loading model from checkpoint: {model_checkpoint_path}")
            model, label_mappings, tokenizer = load_multitask_model(model_checkpoint_path)

            # Export to ONNX
            quantized_output = "model/quantized"
            export_to_onnx(model, tokenizer, quantized_output)

            # Save label mappings
            output_path = Path(quantized_output)
            with (output_path / "label_mappings.json").open("w") as f:
                json.dump(label_mappings, f, indent=2)

            # Quantize
            quantize_model(str(output_path), str(output_path))

            self.quantized_model_path = quantized_output

            # Save quantized model checkpoint
            print(f"\nSaving quantized model checkpoint from: {quantized_output}")
            self.quantized_model = current.checkpoint.save(
                quantized_output,
                metadata={
                    "quantization_mode": "avx512_vnni",
                    "source_model": "trained_model",
                },
                name="quantized_model",
                latest=True,
            )
            print(f"  Quantized model checkpoint saved: {self.quantized_model}")

        except Exception as e:
            print(f"\nQuantization failed: {e}")
            import traceback
            traceback.print_exc()
            self.quantized_model_path = None
            self.quantized_model = None

        self.next(self.sign_model)

    @model(load=["trained_model", "quantized_model"])
    @pypi(packages=CORE_PACKAGES, python="3.10")
    @step
    def sign_model(self):
        """Sign model with cryptographic hash for integrity verification."""
        if self.skip_signing:
            print("\nSkipping model signing")
            self.model_signature = None
            self.next(self.upload_model)
            return

        print("\n" + "=" * 80)
        print("Step 5: Model Signing")
        print("=" * 80)

        try:
            from src.model_signing import sign_trained_model

            # Prefer quantized model if available, otherwise use trained model
            # @model decorator loads checkpoints to current.model.loaded
            if "quantized_model" in current.model.loaded and current.model.loaded["quantized_model"]:
                model_to_sign = current.model.loaded["quantized_model"]
                model_type = "quantized"
                print(f"Loading quantized model for signing: {model_to_sign}")
            else:
                model_to_sign = current.model.loaded["trained_model"]
                model_type = "trained"
                print(f"Loading trained model for signing: {model_to_sign}")

            model_hash = sign_trained_model(model_to_sign)

            self.model_signature = {
                "sha256": model_hash,
                "signed_at": datetime.utcnow().isoformat(),
                "model_type": model_type,
            }

            print(f"\nModel signed successfully!")
            print(f"  Model type: {model_type}")
            print(f"  SHA-256: {model_hash}")

        except ImportError:
            print("\nModel signing not available (install model-signing package)")
            self.model_signature = None
        except Exception as e:
            print(f"\nModel signing failed: {e}")
            import traceback
            traceback.print_exc()
            self.model_signature = None

        self.next(self.upload_model)

    @pypi(packages=CORE_PACKAGES, python="3.10")
    @step
    def upload_model(self):
        """Upload model to Hugging Face Hub (optional)."""
        if self.skip_hf_upload:
            print("\nSkipping Hugging Face upload")
            self.hf_model_url = None
            self.next(self.end)
            return

        print("\n" + "=" * 80)
        print("Step 6: Hugging Face Upload")
        print("=" * 80)

        # TODO: Implement HF upload when Issue #33 is completed
        print("\nHugging Face upload not yet implemented (see Issue #33)")
        self.hf_model_url = None

        self.next(self.end)

    @pypi(packages=CORE_PACKAGES, python="3.10")
    @card
    @step
    def end(self):
        """Finalize pipeline and generate summary report."""
        print("\n" + "=" * 80)
        print("Pipeline Complete!")
        print("=" * 80)

        # Calculate total pipeline time
        end_time = datetime.utcnow()
        start_time = datetime.fromisoformat(self.pipeline_start_time)
        duration = (end_time - start_time).total_seconds()

        # Generate summary
        self.pipeline_summary = {
            "duration_seconds": duration,
            "configuration": {
                "model_name": self.model_name,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
            },
            "dataset": {
                "train_size": self.train_size,
                "val_size": self.val_size,
            },
            "training": {
                "model_path": self.model_path,
                "metrics": self.training_metrics,
            },
            "evaluation": {
                "avg_inference_time_ms": self.avg_inference_time_ms,
                "num_test_cases": len(self.evaluation_results),
            },
            "quantization": {
                "enabled": not self.skip_quantization,
                "model_path": self.quantized_model_path,
            },
            "signing": {
                "enabled": not self.skip_signing,
                "signature": self.model_signature,
            },
        }

        # Print summary
        print(f"\nPipeline Summary:")
        print(f"  Duration: {duration:.0f}s ({duration / 60:.1f} minutes)")
        print(f"  Model: {self.model_path}")
        print(f"  PII F1: {self.training_metrics.get('eval_pii_f1_weighted', 'N/A'):.4f}")
        print(f"  Avg Inference: {self.avg_inference_time_ms:.2f} ms")
        print(f"  Quantized: {'Yes' if self.quantized_model_path else 'No'}")
        print(f"  Signed: {'Yes' if self.model_signature else 'No'}")

        # Save summary to file
        summary_path = Path(self.model_path) / "pipeline_summary.json"
        with summary_path.open("w") as f:
            json.dump(self.pipeline_summary, f, indent=2, default=str)

        print(f"\nSummary saved to: {summary_path}")
        print("=" * 80)


if __name__ == "__main__":
    PIITrainingPipeline()
