"""
Metaflow pipeline for PII detection model training.

This pipeline orchestrates:
1. Data export from Label Studio (optional, can be skipped)
2. Dataset loading and preprocessing from model/dataset/data_samples/training_samples/
3. PII detection model training
4. Model evaluation
5. Model export (ONNX, with optional quantized side artifact)
6. Model signing (cryptographic hash)

Usage:
    # Run locally (with uv extras)
    uv run --extra training --extra quantization --extra signing python model/flows/training_pipeline.py run

    # Custom config file
    uv run --extra training python model/flows/training_pipeline.py --config config-file custom_config.toml run

    # Remote Kubernetes execution (uncomment @pypi and @kubernetes decorators for dependencies)
    python model/flows/training_pipeline.py --environment=pypi run --with kubernetes
"""

import json
import os
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
EXPORT_PACKAGES = {
    **BASE_PACKAGES,
    "label-studio-sdk": ">=0.0.24",
    "requests": ">=2.28.0",
    "tqdm": ">=4.64.0",
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

    # Dataset is now directly accessed from model/dataset/training_samples/

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
        training_cfg = cfg.get("training", {})
        self.config = TrainingConfig(
            model_name=cfg.get("model", {}).get("name", "microsoft/deberta-v3-base"),
            num_epochs=training_cfg.get("num_epochs", 5),
            batch_size=training_cfg.get("batch_size", 16),
            learning_rate=training_cfg.get("learning_rate", 3e-5),
            training_samples_dir=cfg.get("paths", {}).get(
                "training_samples_dir", "model/dataset/data_samples/training_samples"
            ),
            output_dir=cfg.get("paths", {}).get("output_dir", "model/trained"),
            warmup_steps=training_cfg.get("warmup_steps", 200),
            weight_decay=training_cfg.get("weight_decay", 0.01),
            eval_steps=training_cfg.get("eval_steps", 500),
            early_stopping_enabled=training_cfg.get("early_stopping_enabled", True),
            early_stopping_patience=training_cfg.get("early_stopping_patience", 3),
            early_stopping_threshold=training_cfg.get("early_stopping_threshold", 0.01),
            num_ai4privacy_samples=int(
                os.environ.get(
                    "NUM_AI4PRIVACY_SAMPLES",
                    cfg.get("data", {}).get("num_ai4privacy_samples", -1),
                )
            ),
            lr_scheduler_type=training_cfg.get(
                "lr_scheduler_type", "cosine_with_restarts"
            ),
            lr_scheduler_num_cycles=training_cfg.get("lr_scheduler_num_cycles", 3),
            layerwise_lr_decay=training_cfg.get("layerwise_lr_decay", 0.95),
            bf16=training_cfg.get("bf16", False),
            torch_compile=training_cfg.get("torch_compile", False),
            max_eval_samples=training_cfg.get("max_eval_samples", 0),
            balanced_validation_split=training_cfg.get(
                "balanced_validation_split", True
            ),
            auxiliary_ce_loss_weight=training_cfg.get("auxiliary_ce_loss_weight", 0.2),
            audit_allowlist=cfg.get("data", {}).get("audit_allowlist", ""),
        )
        self.skip_export = cfg.get("pipeline", {}).get("skip_export", False)
        self.skip_quantization = cfg.get("pipeline", {}).get("skip_quantization", True)
        self.skip_signing = cfg.get("pipeline", {}).get("skip_signing", False)
        self.subsample_count = int(
            os.environ.get(
                "NUM_SAMPLES",
                cfg.get("data", {}).get("subsample_count", 0),
            )
        )
        self.pipeline_start_time = datetime.utcnow().isoformat()
        # Store raw config for export step
        self.raw_config = cfg

        print(f"Model: {self.config.model_name}")
        print(
            f"Epochs: {self.config.num_epochs}, Batch: {self.config.batch_size}, LR: {self.config.learning_rate}"
        )
        print(f"Data: {self.config.training_samples_dir}")

        self.next(
            {True: self.preprocess_data, False: self.export_data},
            condition="skip_export",
        )

    # @pypi(packages=EXPORT_PACKAGES, python="3.13")
    @step
    def export_data(self):
        """Export data from Label Studio to local files."""
        from src.export_data import ExportDataProcessor

        print("Exporting data from Label Studio...")
        print("-" * 40)

        # Initialize export processor with config
        processor = ExportDataProcessor(self.config, raw_config=self.raw_config)

        # Export data
        results = processor.export_data()

        print(
            f"✅ Exported {results['exported_count']} samples to {results['output_dir']}"
        )

        self.next(self.preprocess_data)

    # @pypi(packages=TRAINING_PACKAGES, python="3.13")
    # @kubernetes(memory=8000, cpu=4)
    @environment(vars={"TOKENIZERS_PARALLELISM": "false"})
    @step
    def preprocess_data(self):
        """Load and preprocess training data from training_samples directory."""
        from src.preprocessing import DatasetProcessor

        # Use the training_samples directory from config
        training_samples_dir = Path(self.config.training_samples_dir)

        # Verify the dataset directory exists and contains data
        if not training_samples_dir.exists():
            raise ValueError(
                f"Dataset directory not found: {training_samples_dir}. "
                "Please ensure the training_samples directory is present, "
                "or set paths.training_samples_dir in your config file."
            )

        json_files = list(training_samples_dir.glob("*.json"))
        if not json_files:
            raise ValueError(
                f"No JSON files found in {training_samples_dir}. "
                "Please ensure the dataset is properly populated."
            )

        print(f"Found {len(json_files)} training samples in {training_samples_dir}")

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Process the dataset
        processor = DatasetProcessor(self.config)
        train_dataset, val_dataset, mappings = processor.prepare_datasets(
            subsample_count=self.subsample_count
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.label_mappings = mappings

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
        """Train the PII detection model."""
        from src.trainer import PIITrainer

        if current.checkpoint.is_loaded:
            print("Resuming from checkpoint...")

        trainer = PIITrainer(self.config)
        trainer.load_label_mappings(self.label_mappings)
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
            _, inference_time = loader.predict(text)
            inference_times.append(inference_time)

        self.avg_inference_time_ms = sum(inference_times) / len(inference_times)
        self.evaluation_results = [{"inference_time_ms": t} for t in inference_times]

        print(
            f"Avg inference: {self.avg_inference_time_ms:.2f}ms ({1000 / self.avg_inference_time_ms:.0f} texts/sec)"
        )

        self.next(self.quantize_model)

    # @pypi(packages=QUANTIZATION_PACKAGES, python="3.13")
    # @kubernetes(memory=10000, cpu=6)
    @environment(vars={"TOKENIZERS_PARALLELISM": "false"})
    @retry(times=2)
    @checkpoint
    @model(load="trained_model")
    @step
    def quantize_model(self):
        """Export model to ONNX and optionally produce a quantized side artifact."""

        import shutil

        from src.parity_benchmark import (
            assert_parity,
            format_parity_report,
            run_parity_benchmark,
        )
        from src.quantitize import (
            export_to_onnx,
            load_model,
            quantize_model as quantize_onnx_model,
        )

        try:
            model_path = current.model.loaded["trained_model"]
            model, label_mappings, tokenizer = load_model(model_path)

            exported_output = "model/quantized"
            output_path = Path(exported_output)
            output_path.mkdir(parents=True, exist_ok=True)
            for stale_name in (
                "model.onnx",
                "model.onnx.data",
                "model_quantized.onnx",
                "crf_transitions.json",
            ):
                stale_path = output_path / stale_name
                if stale_path.exists():
                    stale_path.unlink()

            export_to_onnx(model, tokenizer, exported_output)

            with (output_path / "label_mappings.json").open("w") as f:
                json.dump(label_mappings, f, indent=2)

            # Copy config.json so ORTModel can auto-load the exported model.
            config_src = Path(model_path) / "config.json"
            if config_src.exists():
                shutil.copy(config_src, output_path / "config.json")

            export_parity = run_parity_benchmark(
                model_path,
                exported_output,
                onnx_file="model.onnx",
                confidence_threshold=0.0,
            )
            print(format_parity_report(export_parity))
            assert_parity(export_parity)

            # Stage the runtime artifacts inside the trained model directory
            # so the trained HF repo is self-sufficient for inference (the Go
            # backend loads model.onnx + crf_transitions.json alongside the
            # safetensors). Equivalent to running:
            #   uv run python model/src/quantitize.py \
            #       --model_path=./model/trained \
            #       --output_path=./model/trained \
            #       --skip_quantization
            trained_dir = Path(self.config.output_dir)
            if trained_dir.exists():
                for artifact_name in ("model.onnx", "crf_transitions.json"):
                    src = output_path / artifact_name
                    if src.exists():
                        shutil.copy2(src, trained_dir / artifact_name)
                        print(f"Staged {artifact_name} into {trained_dir}")

            quantized_parity = None
            self.quantized_model_path = None
            if self.skip_quantization:
                print("Skipping quantized side artifact by configuration")
            else:
                try:
                    quantize_onnx_model(str(output_path), str(output_path))
                    quantized_parity = run_parity_benchmark(
                        model_path,
                        exported_output,
                        onnx_file="model_quantized.onnx",
                        confidence_threshold=0.0,
                    )
                    print(format_parity_report(quantized_parity))
                    assert_parity(quantized_parity)
                    self.quantized_model_path = str(output_path / "model_quantized.onnx")
                    print("Quantized side artifact passed parity")
                except Exception as quantization_error:
                    failed_quantized = output_path / "model_quantized.onnx"
                    if failed_quantized.exists():
                        failed_quantized.unlink()
                    print(
                        "Quantized side artifact failed parity and will not be used: "
                        f"{quantization_error}"
                    )

            parity_reports = {
                "export": export_parity.to_dict(),
            }
            if quantized_parity is not None:
                parity_reports["quantized"] = quantized_parity.to_dict()
            self.parity_reports = parity_reports

            self.exported_model_path = exported_output
            self.exported_model = current.checkpoint.save(
                exported_output,
                metadata={
                    "default_onnx_file": "model.onnx",
                    "quantized_model_path": self.quantized_model_path,
                },
                name="exported_model",
                latest=True,
            )

            print(f"Exported ONNX model saved: {exported_output}/model.onnx")

        except Exception as e:
            print(f"ONNX export failed: {e}")
            self.exported_model_path = None
            self.exported_model = None
            self.quantized_model_path = None
            self.parity_reports = None
            raise

        self.next(self.sign_model)

    # @pypi(packages=SIGNING_PACKAGES, python="3.13")
    @checkpoint
    @model(load=["trained_model"])  # Keep trained model as a fallback if export failed.
    @step
    def sign_model(self):
        """Sign model with cryptographic hash."""
        try:
            from src.model_signing import sign_trained_model

            # Get private key path from environment
            private_key_path = os.getenv("MODEL_SIGNING_KEY_PATH")

            # Sign the exported ONNX directory by default. This is the artifact the app loads.
            exported_path = None
            if getattr(self, "exported_model", None) is not None:
                try:
                    exported_path = current.checkpoint.load(self.exported_model)
                    print("Loaded exported ONNX model from checkpoint")
                except Exception as e:
                    print(f"Could not load exported ONNX model: {e}")

            if exported_path:
                model_to_sign = exported_path
                model_type = "onnx"
            else:
                model_to_sign = current.model.loaded["trained_model"]
                model_type = "trained"

            model_hash = sign_trained_model(
                model_to_sign, private_key_path=private_key_path
            )
            self.model_signature = {
                "sha256": model_hash,
                "signed_at": datetime.utcnow().isoformat(),
                "model_type": model_type,
                "signing_method": ("private_key" if private_key_path else "hash_only"),
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
            "exported": self.exported_model_path is not None,
            "quantized": self.quantized_model_path is not None,
            "signed": self.model_signature is not None,
            "parity": getattr(self, "parity_reports", None),
        }


if __name__ == "__main__":
    PIITrainingPipeline()
