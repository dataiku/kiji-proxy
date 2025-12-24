"""
Metaflow pipeline for PII detection model training.

This pipeline orchestrates:
1. Dataset loading and preprocessing from model/dataset/reviewed_samples/
2. Model training with multi-task learning
3. Model evaluation
4. Model quantization (ONNX)
5. Model signing (cryptographic hash)

Usage:
    # Run locally (with uv extras)
    uv run --extra training --extra signing python model/flows/training_pipeline.py run
    # If on linux VM, we can do quantization too:
    uv run --extra training --extra quantization --extra signing python model/flows/training_pipeline.py run

    # Custom config file
    uv run --extra training python model/flows/training_pipeline.py --config-file custom_config.toml run

    # Remote Kubernetes execution (uncomment @pypi and @kubernetes decorators for dependencies)
    python model/flows/training_pipeline.py --environment=pypi run --with kubernetes
"""

import json
import os
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

from metaflow import (
    Config,
    FlowSpec,
    card,
    checkpoint,
    current,
    environment,
    model,
    retry,
    step,
)

##################################################################
# Why not use the pyproject.toml dependencies?
# Because the current Metaflow implementation only uses
# the pyproject.toml dependencies in the top-level dependencies.
# A short-term Metaflow limitation should not necessitate a change
# to this project's pyproject.toml structure. For now, we leverage
# Metaflow's robust environment building utilities as a stop gap.
# In the future, we can obviate the need for @pypi if desirable.
# TODO (Eddie): Update/remove this after feature support for
#       python flow.py --environment=uv:extra=training run
# is merged into the Metaflow codebase.
##################################################################
BASE_PACKAGES = {
    "torch": ">=2.0.0",
    "transformers": ">=4.20.0",
    "numpy": ">=1.21.0",
    "datasets": ">=2.0.0",
    "huggingface-hub": ">=0.20.0",
    "safetensors": ">=0.3.0",
    "absl-py": ">=2.0.0",
    "python-dotenv": ">=1.0.0",
}
###############################
TRAINING_PACKAGES = {
    **BASE_PACKAGES,
    "scikit-learn": ">=1.0.0",
    "accelerate": ">=0.26.0",
    "tqdm": ">=4.64.0",
    "scipy": ">=1.0.0",
}
###############################
QUANTIZATION_PACKAGES = {
    **BASE_PACKAGES,
    "optimum[onnxruntime]": ">=1.15.0",
    "onnx": ">=1.15.0",
    "onnxruntime": ">=1.16.0",
    "onnxscript": ">=0.1.0",
}
###############################
SIGNING_PACKAGES = {
    "model-signing": ">=1.1.1",
}
##################################################################


class PIITrainingPipeline(FlowSpec):
    """
    End-to-end ML pipeline for PII detection model training.

    Configuration is loaded from training_config.toml.
    All settings are controlled via the config file.
    """

    config_file = Config(
        "config-file",
        default=os.path.join(os.path.dirname(__file__), "training_config.toml"),
        parser=tomllib.loads,
        help="TOML config file with training hyperparameters",
    )

    # Dataset is now directly accessed from model/dataset/reviewed_samples/


    # @pypi(packages=BASE_PACKAGES, python="3.13")
    @step
    def start(self):
        """Initialize pipeline configuration."""
        from src.config import EnvironmentSetup, TrainingConfig

        print("PII Detection Model Training Pipeline")
        print("-" * 40)

        EnvironmentSetup.disable_wandb()
        EnvironmentSetup.check_gpu()

        cfg = self.config_file
        self.config = TrainingConfig(
            model_name=cfg.get("model", {}).get("name", "distilbert-base-cased"),
            num_epochs=cfg.get("training", {}).get("num_epochs", 5),
            batch_size=cfg.get("training", {}).get("batch_size", 16),
            learning_rate=cfg.get("training", {}).get("learning_rate", 3e-5),
            training_samples_dir=cfg.get("paths", {}).get(
                "training_samples_dir", "model/dataset/reviewed_samples"
            ),
            output_dir=cfg.get("paths", {}).get("output_dir", "model/trained"),
        )
        self.skip_quantization = cfg.get("pipeline", {}).get("skip_quantization", False)
        self.skip_signing = cfg.get("pipeline", {}).get("skip_signing", False)
        self.subsample_count = cfg.get("data", {}).get("subsample_count", 0)
        self.pipeline_start_time = datetime.utcnow().isoformat()

        print(f"Model: {self.config.model_name}")
        print(
            f"Epochs: {self.config.num_epochs}, Batch: {self.config.batch_size}, LR: {self.config.learning_rate}"
        )
        print(f"Data: {self.config.training_samples_dir}")

        self.next(self.preprocess_data)

    # @pypi(packages=TRAINING_PACKAGES, python="3.13")
    # @kubernetes(memory=8000, cpu=4)
    @environment(vars={"TOKENIZERS_PARALLELISM": "false"})
    @step
    def preprocess_data(self):
        """Load and preprocess training data from reviewed_samples directory."""
        from src.preprocessing import DatasetProcessor

        # Use the reviewed_samples directory directly
        # This is the curated, high-quality dataset for training
        reviewed_samples_dir = Path("model/dataset/reviewed_samples")

        # Verify the dataset directory exists and contains data
        if not reviewed_samples_dir.exists():
            raise ValueError(
                f"Dataset directory not found: {reviewed_samples_dir}. "
                "Please ensure the reviewed_samples directory is present."
            )

        json_files = list(reviewed_samples_dir.glob("*.json"))
        if not json_files:
            raise ValueError(
                f"No JSON files found in {reviewed_samples_dir}. "
                "Please ensure the dataset is properly populated."
            )

        print(f"Found {len(json_files)} reviewed samples in {reviewed_samples_dir}")

        # Update config to use reviewed_samples directory
        self.config.training_samples_dir = str(reviewed_samples_dir)

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Process the dataset
        processor = DatasetProcessor(self.config)
        train_dataset, val_dataset, mappings, coref_info = processor.prepare_datasets(
            subsample_count=self.subsample_count
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.label_mappings = mappings
        self.coref_info = coref_info

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"PII labels: {len(mappings['pii']['label2id'])}")

        self.next(self.train_model)

    # @pypi(packages=TRAINING_PACKAGES, python="3.13")
    # @kubernetes(memory=16000, cpu=8)
    @environment(vars={"TOKENIZERS_PARALLELISM": "false", "WANDB_DISABLED": "true"})
    @retry(times=2)
    @checkpoint
    @step
    def train_model(self):
        """Train the multi-task PII detection model."""
        from src.trainer import PIITrainer

        if current.checkpoint.is_loaded:
            print("Resuming from checkpoint...")

        trainer = PIITrainer(self.config)
        trainer.load_label_mappings(self.label_mappings, self.coref_info)
        trainer.initialize_model()

        start_time = time.time()
        trained_trainer = trainer.train(self.train_dataset, self.val_dataset)
        training_time = time.time() - start_time

        results = trainer.evaluate(self.val_dataset, trained_trainer)

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

        # Ensure label_mappings.json exists
        model_dir = Path(self.model_path)
        mappings_path = model_dir / "label_mappings.json"
        if not mappings_path.exists():
            with mappings_path.open("w") as f:
                json.dump(self.label_mappings, f, indent=2)

        # Save checkpoint
        if model_dir.exists():
            self.trained_model = current.checkpoint.save(
                str(model_dir),
                metadata={
                    "pii_f1_weighted": self.training_metrics.get(
                        "eval_pii_f1_weighted"
                    ),
                    "training_time_seconds": training_time,
                },
                name="trained_model",
                latest=True,
            )
        else:
            self.trained_model = None

        pii_f1 = self.training_metrics.get("eval_pii_f1_weighted")
        print(
            f"Training: {training_time / 60:.1f}min, PII F1: {pii_f1:.4f}"
            if pii_f1
            else "Training complete"
        )

        self.next(self.evaluate_model)

    # @pypi(packages=BASE_PACKAGES, python="3.13")
    @environment(vars={"TOKENIZERS_PARALLELISM": "false"})
    @model(load="trained_model")
    @step
    def evaluate_model(self):
        """
        Evaluate the trained model on fixed test cases.
        This is a quick way to check if the model is working as expected, and loads on an independent machine/environment from training.
        """
        from src.eval_model import PIIModelLoader

        model_path = current.model.loaded["trained_model"]
        loader = PIIModelLoader(model_path)
        loader.load_model()

        test_cases = [
            "My name is John Smith and my email is john@example.com",
            "Call me at 555-123-4567 or email sarah.miller@company.com",
            "SSN: 123-45-6789, DOB: 01/15/1990",
            "I live at 123 Main Street, Springfield, IL 62701",
        ]

        inference_times = []
        for text in test_cases:
            _, _, inference_time = loader.predict(text)
            inference_times.append(inference_time)

        self.avg_inference_time_ms = sum(inference_times) / len(inference_times)
        self.evaluation_results = [{"inference_time_ms": t} for t in inference_times]

        print(
            f"Avg inference: {self.avg_inference_time_ms:.2f}ms ({1000 / self.avg_inference_time_ms:.0f} texts/sec)"
        )

        self.is_linux = sys.platform == "linux"
        if not self.is_linux:
            print("Skipping ONNX quantization on Mac or Windows")
            self.quantized_model_path = None
            self.quantized_model = None

        self.next(
            {True: self.quantize_model, False: self.sign_model}, condition="is_linux"
        )

    # @pypi(packages=QUANTIZATION_PACKAGES, python="3.13")
    # @kubernetes(memory=10000, cpu=6)
    @environment(vars={"TOKENIZERS_PARALLELISM": "false"})
    @retry(times=2)
    @checkpoint
    @model(load="trained_model")
    @step
    def quantize_model(self):
        """Quantize model to ONNX format."""

        from src.quantitize import export_to_onnx, load_multitask_model, quantize_model

        try:
            model_path = current.model.loaded["trained_model"]
            model, label_mappings, tokenizer = load_multitask_model(model_path)

            quantized_output = "model/quantized"
            export_to_onnx(model, tokenizer, quantized_output)

            output_path = Path(quantized_output)
            with (output_path / "label_mappings.json").open("w") as f:
                json.dump(label_mappings, f, indent=2)

            quantize_model(str(output_path), str(output_path))

            self.quantized_model_path = quantized_output
            self.quantized_model = current.checkpoint.save(
                quantized_output,
                metadata={"quantization_mode": "avx512_vnni"},
                name="quantized_model",
                latest=True,
            )
            print(f"Quantized model saved: {quantized_output}")

        except Exception as e:
            print(f"Quantization failed: {e}")
            self.quantized_model_path = None
            self.quantized_model = None

        self.next(self.sign_model)

    # @pypi(packages=SIGNING_PACKAGES, python="3.13")
    @model(load=["trained_model"])  # Only require trained_model; quantized is optional
    @step
    def sign_model(self):
        """Sign model with cryptographic hash."""
        try:
            from src.model_signing import sign_trained_model

            # Try to load quantized model if it exists
            quantized_path = None
            if getattr(self, "quantized_model", None) is not None:
                try:
                    quantized_path = current.checkpoint.load(self.quantized_model)
                    print("Loaded quantized model from checkpoint")
                except Exception as e:
                    print(f"Could not load quantized model: {e}")

            if quantized_path:
                model_to_sign = quantized_path
                model_type = "quantized"
            else:
                model_to_sign = current.model.loaded["trained_model"]
                model_type = "trained"

            model_hash = sign_trained_model(model_to_sign)
            self.model_signature = {
                "sha256": model_hash,
                "signed_at": datetime.utcnow().isoformat(),
                "model_type": model_type,
            }
            print(f"Signed ({model_type}): {model_hash[:16]}...")

        except Exception as e:
            print(f"Signing failed: {e}")
            self.model_signature = None

        self.next(self.end)

    # @pypi(packages=BASE_PACKAGES, python="3.13")
    @card
    @step
    def end(self):
        """Generate summary report."""

        end_time = datetime.utcnow()
        start_time = datetime.fromisoformat(self.pipeline_start_time)
        duration = (end_time - start_time).total_seconds()

        self.pipeline_summary = {
            "duration_seconds": duration,
            "config": {
                "model": self.config.model_name,
                "epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
            },
            "dataset": {},
            "metrics": self.training_metrics,
            "quantized": self.quantized_model_path is not None,
            "signed": self.model_signature is not None,
        }


if __name__ == "__main__":
    PIITrainingPipeline()
