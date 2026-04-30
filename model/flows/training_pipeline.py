"""
Metaflow pipeline for PII detection model training.

This pipeline orchestrates:
1. Data export from Label Studio (optional, can be skipped)
2. Dataset loading and preprocessing from model/dataset/data_samples/training_samples/
3. PII detection model training
4. Model evaluation
5. ONNX export (fp32) + quantization variant generation via model.src.quantize_variants
6. Sweep eval — benchmarks every variant against ai4privacy/pii-masking-300k and
   fails the run if the primary variant (fp16 by default) drops below threshold
7. Model signing — fp32 export + every variant gets an independent signature

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
        self.skip_quantization = cfg.get("pipeline", {}).get("skip_quantization", False)
        self.skip_signing = cfg.get("pipeline", {}).get("skip_signing", False)
        # Comma-separated variant names (or "all") — passed straight to
        # model/src/quantize_variants.py. Defaults to the 9 production-ready
        # variants we actually ship; excludes int8_static (slow, needs
        # calibration data) and the *_reduce_range / matmul_weight_only
        # experiments.
        self.quantization_variants = cfg.get("pipeline", {}).get(
            "quantization_variants",
            "fp32,fp16,int8_dyn_default,int8_dyn_avx512_vnni,int8_dyn_avx512,"
            "int8_dyn_avx2,int4_rtn_block32,int4_rtn_block128,int4_hqq_block64",
        )
        # Sweep eval knobs
        self.sweep_num_samples = int(
            cfg.get("pipeline", {}).get("sweep_num_samples", 1000)
        )
        self.sweep_seed = int(cfg.get("pipeline", {}).get("sweep_seed", 42))
        # Primary variant is the one that ships in the DMG; pipeline fails
        # if its sweep F1 drops below the threshold so we don't accidentally
        # regress the production artifact.
        self.primary_variant = cfg.get("pipeline", {}).get("primary_variant", "fp16")
        self.primary_variant_min_f1 = float(
            cfg.get("pipeline", {}).get("primary_variant_min_f1", 0.85)
        )
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
        """Export model to ONNX (fp32) and build all quantization variants.

        Produces:
          * ``model/quantized/model.onnx`` — fp32 reference; parity-checked
            against the trained PyTorch model.
          * ``model/quant_variants/<variant>/model.onnx`` — one directory per
            entry in ``self.quantization_variants``; each is self-contained
            (tokenizer + label mappings + manifest). The end-to-end quality
            of each variant is measured by the downstream ``sweep_eval`` step.
        """

        import shutil
        import subprocess
        import sys

        from src.parity_benchmark import (
            assert_parity,
            format_parity_report,
            run_parity_benchmark,
        )
        from src.quantitize import export_to_onnx, load_model

        try:
            model_path = current.model.loaded["trained_model"]
            model, label_mappings, tokenizer = load_model(model_path)

            # 1. Export PyTorch -> ONNX (fp32 source for variants)
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
            for stale in output_path.glob("model_fp16.onnx"):
                stale.unlink()
            for stale in output_path.glob("model_int8_*.onnx"):
                stale.unlink()

            export_to_onnx(model, tokenizer, exported_output)

            with (output_path / "label_mappings.json").open("w") as f:
                json.dump(label_mappings, f, indent=2)

            config_src = Path(model_path) / "config.json"
            if config_src.exists():
                shutil.copy(config_src, output_path / "config.json")

            # 2. Parity-check the fp32 export vs the trained PyTorch model.
            export_parity = run_parity_benchmark(
                model_path,
                exported_output,
                onnx_file="model.onnx",
                confidence_threshold=0.0,
            )
            print(format_parity_report(export_parity))
            assert_parity(export_parity)

            self.exported_model_path = exported_output
            self.exported_model = current.checkpoint.save(
                exported_output,
                metadata={"default_onnx_file": "model.onnx"},
                name="exported_model",
                latest=True,
            )
            self.parity_reports = {"export": export_parity.to_dict()}

            # 3. Build quantization variants.
            variants_output = "model/quant_variants"
            if Path(variants_output).exists():
                shutil.rmtree(variants_output)

            if self.skip_quantization:
                print("Skipping variant generation by configuration")
                self.variants_path = None
                self.quantized_variants = None
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "model.src.quantize_variants",
                    "--source",
                    exported_output,
                    "--output",
                    variants_output,
                    "--variants",
                    self.quantization_variants,
                ]
                print(f"\nRunning: {' '.join(cmd)}")
                result = subprocess.run(cmd, check=False)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"quantize_variants failed (exit {result.returncode})"
                    )

                built = sorted(
                    d.name
                    for d in Path(variants_output).iterdir()
                    if d.is_dir() and (d / "model.onnx").exists()
                )
                if not built:
                    raise RuntimeError(
                        f"quantize_variants produced no variants under {variants_output}"
                    )
                print(f"Built {len(built)} variant(s): {built}")

                self.variants_path = variants_output
                self.quantized_variants = current.checkpoint.save(
                    variants_output,
                    metadata={
                        "variants_built": ",".join(built),
                        "variants_requested": self.quantization_variants,
                    },
                    name="quantized_variants",
                    latest=True,
                )

            print(f"Exported ONNX model saved: {exported_output}/model.onnx")

        except Exception as e:
            print(f"Quantize step failed: {e}")
            self.exported_model_path = None
            self.exported_model = None
            self.variants_path = None
            self.quantized_variants = None
            self.parity_reports = None
            raise

        self.next(self.sweep_eval)

    # @pypi(packages=QUANTIZATION_PACKAGES, python="3.13")
    @environment(vars={"TOKENIZERS_PARALLELISM": "false"})
    @step
    def sweep_eval(self):
        """Run the benchmark sweep across every quantization variant.

        Delegates to ``tests.benchmark.sweep_quants``, which iterates each
        variant directory and invokes ``tests.benchmark.run`` (against the
        ai4privacy/pii-masking-300k eval split). Results land as a metaflow
        artifact (``self.sweep_report``) and the pipeline fails closed if the
        primary variant's exact-span F1 drops below ``primary_variant_min_f1``.
        """
        import subprocess
        import sys

        self.sweep_report = None

        if getattr(self, "quantized_variants", None) is None:
            print("No variants produced (quantization skipped) — skipping sweep_eval")
        else:
            variants_path = current.checkpoint.load(self.quantized_variants)
            print(f"Loaded variants from checkpoint: {variants_path}")

            report_path = Path("tests/benchmark/reports/quant_sweep.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            if report_path.exists():
                report_path.unlink()

            cmd = [
                sys.executable,
                "-m",
                "tests.benchmark.sweep_quants",
                "--variants-dir",
                str(variants_path),
                "--num",
                str(self.sweep_num_samples),
                "--seed",
                str(self.sweep_seed),
                "--report",
                str(report_path),
            ]
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=False)

            if report_path.exists():
                with report_path.open() as f:
                    self.sweep_report = json.load(f)

            # sweep_quants exits 2 if any individual variant failed; we still
            # accept the report so partial results are visible, but we fail
            # the pipeline if the *primary* variant didn't make the threshold.
            primary_row = None
            if self.sweep_report:
                for v in self.sweep_report.get("variants", []):
                    if v.get("variant") == self.primary_variant:
                        primary_row = v
                        break

            if primary_row is None:
                raise RuntimeError(
                    f"Primary variant '{self.primary_variant}' missing from sweep "
                    f"report. sweep_quants exit code: {result.returncode}"
                )
            if primary_row.get("status") != "ok":
                raise RuntimeError(
                    f"Primary variant '{self.primary_variant}' did not benchmark "
                    f"successfully: {primary_row}"
                )
            primary_f1 = float(primary_row.get("exact_f1") or 0.0)
            if primary_f1 < self.primary_variant_min_f1:
                raise RuntimeError(
                    f"Primary variant '{self.primary_variant}' exact F1 "
                    f"{primary_f1:.4f} is below threshold "
                    f"{self.primary_variant_min_f1:.4f}; refusing to ship."
                )
            print(
                f"Primary variant '{self.primary_variant}' OK: "
                f"exact_f1={primary_f1:.4f}, "
                f"p50={primary_row.get('latency_ms', {}).get('p50', '?')}ms"
            )

        self.next(self.sign_model)

    # @pypi(packages=SIGNING_PACKAGES, python="3.13")
    @checkpoint
    @model(load=["trained_model"])  # Keep trained model as a fallback if export failed.
    @step
    def sign_model(self):
        """Sign the fp32 export and every quantization variant.

        Each directory gets an independent signature + manifest written by
        ``model_signing.sign_trained_model``. Output is collected in
        ``self.model_signatures`` keyed by variant name (plus ``fp32`` for the
        ``model/quantized/`` reference bundle).
        """
        from src.model_signing import sign_trained_model

        signatures: dict[str, dict] = {}

        if self.skip_signing:
            print("Skipping model signing by configuration")
        else:
            private_key_path = os.getenv("MODEL_SIGNING_KEY_PATH")
            signing_method = "private_key" if private_key_path else "hash_only"
            signed_at = datetime.utcnow().isoformat()

            # 1. Sign the fp32 export (or fall back to the trained PyTorch dir).
            fp32_path = None
            if getattr(self, "exported_model", None) is not None:
                try:
                    fp32_path = current.checkpoint.load(self.exported_model)
                except Exception as e:
                    print(f"Could not load exported ONNX checkpoint: {e}")

            if fp32_path is None:
                fp32_path = current.model.loaded["trained_model"]
                fp32_kind = "trained"
            else:
                fp32_kind = "onnx"

            try:
                h = sign_trained_model(fp32_path, private_key_path=private_key_path)
                signatures["fp32"] = {
                    "sha256": h,
                    "path": str(fp32_path),
                    "kind": fp32_kind,
                    "signing_method": signing_method,
                    "signed_at": signed_at,
                }
                print(f"Signed fp32 ({fp32_kind}): {h[:16]}...")
            except Exception as e:
                print(f"Failed to sign fp32 bundle: {e}")
                signatures["fp32"] = {"error": str(e), "path": str(fp32_path)}

            # 2. Sign every quantization variant directory.
            if getattr(self, "quantized_variants", None) is not None:
                try:
                    variants_path = current.checkpoint.load(self.quantized_variants)
                except Exception as e:
                    print(f"Could not load variants checkpoint: {e}")
                    variants_path = None

                if variants_path is not None:
                    for variant_dir in sorted(Path(variants_path).iterdir()):
                        if not variant_dir.is_dir():
                            continue
                        if not (variant_dir / "model.onnx").exists():
                            continue
                        try:
                            h = sign_trained_model(
                                str(variant_dir), private_key_path=private_key_path
                            )
                            signatures[variant_dir.name] = {
                                "sha256": h,
                                "path": str(variant_dir),
                                "kind": "variant",
                                "signing_method": signing_method,
                                "signed_at": signed_at,
                            }
                            print(f"Signed {variant_dir.name}: {h[:16]}...")
                        except Exception as e:
                            print(f"Failed to sign {variant_dir.name}: {e}")
                            signatures[variant_dir.name] = {
                                "error": str(e),
                                "path": str(variant_dir),
                            }

        self.model_signatures = signatures or None
        # Keep the legacy single-signature artifact for back-compat with any
        # external readers (UI, status pages) that key off `model_signature`.
        self.model_signature = signatures.get("fp32") if signatures else None
        self.next(self.end)

    # @pypi(packages=BASE_PACKAGES, python="3.13")
    @card
    @step
    def end(self):
        """Generate summary report."""

        end_time = datetime.utcnow()
        start_time = datetime.fromisoformat(self.pipeline_start_time)
        duration = (end_time - start_time).total_seconds()

        sweep_report = getattr(self, "sweep_report", None)
        sweep_summary = None
        if sweep_report:
            ok_variants = [
                v for v in sweep_report.get("variants", []) if v.get("status") == "ok"
            ]
            sweep_summary = {
                "num_variants_ok": len(ok_variants),
                "num_variants_total": len(sweep_report.get("variants", [])),
                "primary_variant": self.primary_variant,
                "primary_variant_exact_f1": next(
                    (
                        v.get("exact_f1")
                        for v in sweep_report.get("variants", [])
                        if v.get("variant") == self.primary_variant
                    ),
                    None,
                ),
            }

        signatures = getattr(self, "model_signatures", None) or {}
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
            "variants_built": getattr(self, "variants_path", None) is not None,
            "num_variants_signed": sum(
                1 for v in signatures.values() if "sha256" in v
            ),
            "sweep": sweep_summary,
            "parity": getattr(self, "parity_reports", None),
        }


if __name__ == "__main__":
    PIITrainingPipeline()
