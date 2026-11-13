"""Configuration and environment setup for training."""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from absl import logging


def is_tpu_available() -> bool:
    """Return True when TorchTPU has registered a usable "tpu" device."""
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None or not accelerator.is_available():
        return False
    device = accelerator.current_accelerator()
    return device is not None and device.type == "tpu"


def get_best_device() -> torch.device:
    """Get the best available device (TPU > MPS > CUDA > CPU)."""
    if is_tpu_available():
        return torch.device("tpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_accelerate_device() -> None:
    """Point the HF Trainer and accelerate at the TPU when TorchTPU is present.

    accelerate does not auto-detect TorchTPU's "tpu" device and would otherwise
    train on CPU. Must run before TrainingArguments is created.
    """
    if is_tpu_available():
        os.environ.setdefault("ACCELERATE_TORCH_DEVICE", "tpu")


@dataclass
class TrainingConfig:
    """Configuration for PII detection model training."""

    # Model settings
    # Supported base encoders (any HuggingFace AutoModel-compatible model):
    #   - distilbert-base-cased:      66M params, fastest inference
    #   - microsoft/deberta-v3-small:  44M params, better NER accuracy than DistilBERT
    #   - microsoft/deberta-v3-base:   86M params, best NER accuracy at this scale
    #   - roberta-base:               125M params, strong general-purpose encoder
    model_name: str = "microsoft/deberta-v3-base"

    # Training parameters
    num_epochs: int = 5
    batch_size: int = 16
    # Effective batch = batch_size * gradient_accumulation_steps (per device)
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-5
    max_samples: int = 400000

    # Training optimization
    warmup_steps: int = 200
    weight_decay: float = 0.01
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 500
    seed: int = 42
    bf16: bool = False  # Enable bf16 mixed precision (requires Ampere+ GPU)
    torch_compile: bool = False  # Enable torch.compile for faster training
    lr_scheduler_type: str = "cosine_with_restarts"
    lr_scheduler_num_cycles: int = 3
    layerwise_lr_decay: float = (
        0.95  # Multiplicative decay per encoder layer (1.0 = disabled)
    )

    # Output and logging
    output_dir: str = "./model/trained"
    use_wandb: bool = False
    use_custom_loss: bool = True
    class_weights: dict[int, float] = field(default_factory=dict)
    auxiliary_ce_loss_weight: float = (
        0.2  # Weighted token CE added to CRF loss (0.0 disables)
    )

    # Dataset settings
    eval_size_ratio: float = 0.1  # Validation set size as ratio of training
    max_eval_samples: int = 0  # Cap eval set size (0 = no cap)
    balanced_validation_split: bool = True
    max_sequence_length: int = 512
    # Pad each batch up to the next multiple of pad_bucket_size (capped at
    # max_sequence_length) so XLA compiles one graph per bucket instead of one
    # per batch length. None = enabled only on TPU.
    static_padding: bool | None = None
    pad_bucket_size: int = 128
    training_samples_dir: str = "model/dataset/data_samples/training_samples"  # Use training samples by default, exported from Label Studio

    # Sample filtering
    audit_allowlist: str = ""  # Path to audit_allowlist.txt (empty = no filtering)

    # External dataset augmentation
    num_ai4privacy_samples: int = (
        -1  # ai4privacy/pii-masking-300k samples to add (-1 = none, 0 = all)
    )

    # Early stopping
    early_stopping_enabled: bool = True  # Enable early stopping
    early_stopping_patience: int = (
        5  # Number of eval steps with no improvement before stopping
    )
    early_stopping_threshold: float = 0.001  # Minimum improvement (0.1%) to qualify

    def __post_init__(self):
        """Create output directory after initialization."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def print_summary(self):
        """Print configuration summary."""
        logging.info("\n📋 Training Configuration:")
        logging.info(f"  Model: {self.model_name}")
        logging.info(f"  Epochs: {self.num_epochs}")
        logging.info(
            f"  Batch Size: {self.batch_size} "
            f"x {self.gradient_accumulation_steps} accumulation steps"
        )
        logging.info(f"  Learning Rate: {self.learning_rate}")
        logging.info(f"  Max Samples: {self.max_samples}")
        logging.info(f"  Output Dir: {self.output_dir}")
        logging.info(f"  Custom Loss: {self.use_custom_loss}")
        logging.info(f"  Auxiliary CE Loss Weight: {self.auxiliary_ce_loss_weight}")


class EnvironmentSetup:
    """Handles environment setup and package installation."""

    @staticmethod
    def mount_google_drive(mount_point: str = "/content/drive"):
        """
        Mount Google Drive in Colab environment.

        Args:
            mount_point: Path where Google Drive should be mounted
        """
        try:
            from google.colab import drive

            drive.mount(mount_point)
            logging.info(f"✅ Google Drive mounted at {mount_point}")
            return True
        except ImportError:
            logging.warning("⚠️  Not running in Google Colab - skipping Drive mount")
            return False
        except Exception:
            logging.exception("❌ Failed to mount Google Drive")
            return False

    @staticmethod
    def disable_wandb():
        """Disable Weights & Biases to avoid API key prompts."""
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_PROJECT"] = ""
        os.environ["WANDB_ENTITY"] = ""
        logging.info("✅ Weights & Biases (wandb) disabled")

    @staticmethod
    def install_package(package_list: list[str], index_url: str | None = None):
        """Install packages with optional index URL."""
        cmd = [sys.executable, "-m", "pip", "install", "-q"]
        if index_url:
            cmd.extend(["--index-url", index_url])
        cmd.extend(package_list)

        try:
            subprocess.check_call(cmd)
            logging.info(f"✅ Successfully installed: {', '.join(package_list)}")
        except subprocess.CalledProcessError:
            logging.exception(f"❌ Failed to install: {', '.join(package_list)}")
            if index_url:
                logging.info("Trying fallback installation...")
                cmd_fallback = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    *package_list,
                ]
                subprocess.check_call(cmd_fallback)
                logging.info("✅ Fallback installation successful")
            else:
                raise

    @staticmethod
    def setup_pytorch():
        """Install PyTorch with CUDA support if available."""
        logging.info("Installing PyTorch...")
        try:
            EnvironmentSetup.install_package(
                ["torch", "torchvision", "torchaudio"],
                index_url="https://download.pytorch.org/whl/cu118",
            )
        except Exception:
            logging.warning("CUDA installation failed, installing CPU version...")
            EnvironmentSetup.install_package(["torch", "torchvision", "torchaudio"])

    @staticmethod
    def setup_dependencies():
        """Install all required dependencies."""
        logging.info("Installing required packages...")
        packages = [
            "transformers",
            "datasets",
            "scikit-learn",
            "tqdm",
            "psutil",
            "accelerate",
        ]
        EnvironmentSetup.install_package(packages)

    @staticmethod
    def check_gpu():
        """Check and print accelerator availability (TPU, MPS, CUDA, or CPU)."""
        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        cuda_available = torch.cuda.is_available()

        if is_tpu_available():
            logging.info("\n✅ TPU available via TorchTPU")
            logging.info(f"   Chips: {torch.accelerator.device_count()}")
            logging.info("   Device: tpu")
        elif mps_available:
            logging.info(
                f"\n✅ MPS (Metal) available: {torch.backends.mps.is_available()}"
            )
            logging.info("   Using Apple Silicon GPU acceleration")
            logging.info("   Device: mps")
        elif cuda_available:
            logging.info(f"\n✅ CUDA available: {cuda_available}")
            logging.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logging.info(
                f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
            logging.info("   Device: cuda")
        else:
            logging.info("\n⚠️  No GPU available - using CPU")
            logging.info(f"   CUDA available: {cuda_available}")
            logging.info(f"   MPS available: {mps_available}")
            logging.info("   Device: cpu")

    @staticmethod
    def get_device():
        """Get the best available device (TPU > MPS > CUDA > CPU)."""
        return get_best_device()
