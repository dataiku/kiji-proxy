"""Configuration and environment setup for training."""

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from absl import logging


@dataclass
class TrainingConfig:
    """Configuration for PII detection model training."""

    # Model settings
    model_name: str = "answerdotai/ModernBERT-base"  # 149M params, modern architecture

    # Training parameters
    num_epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 3e-5
    max_samples: int = 400000

    # Training optimization
    warmup_steps: int = 200
    weight_decay: float = 0.01
    logging_steps: int = 500
    seed: int = 42

    # Output and logging
    output_dir: str = "./model/trained"
    use_wandb: bool = False
    use_custom_loss: bool = True
    class_weights: dict[int, float] = field(default_factory=dict)

    # Dataset settings
    eval_size_ratio: float = 0.1  # Validation set size as ratio of training
    training_samples_dir: str = "model/dataset/training_samples"  # Use training samples by default, exported from Label Studio
    max_sequence_length: int = 4096

    # Multi-task learning
    pii_loss_weight: float = 1.0  # Weight for PII detection loss
    coref_loss_weight: float = (
        0.5  # Weight for co-reference detection loss (secondary task)
    )

    # Focal Loss parameters (for handling class imbalance with long sequences)
    focal_gamma: float = 2.0  # Focusing parameter (0=CE, 2=typical for detection)
    focal_alpha: float = 0.75  # Balance factor for positive classes
    label_smoothing: float = 0.1  # Prevent overconfidence

    # Early stopping
    early_stopping_enabled: bool = True  # Enable early stopping
    early_stopping_patience: int = (
        3  # Number of eval steps with no improvement before stopping
    )
    early_stopping_threshold: float = 0.01  # Minimum improvement (1%) to qualify

    def __post_init__(self):
        """Create output directory after initialization."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def print_summary(self):
        """Print configuration summary."""
        logging.info("\n📋 Training Configuration:")
        logging.info(f"  Model: {self.model_name}")
        logging.info(f"  Epochs: {self.num_epochs}")
        logging.info(f"  Batch Size: {self.batch_size}")
        logging.info(f"  Learning Rate: {self.learning_rate}")
        logging.info(f"  Max Samples: {self.max_samples}")
        logging.info(f"  Output Dir: {self.output_dir}")
        logging.info(f"  Custom Loss: {self.use_custom_loss}")


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
        """Check and print GPU availability (MPS, CUDA, or CPU)."""
        # Check MPS (Apple Silicon) first
        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        cuda_available = torch.cuda.is_available()

        if mps_available:
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
        """Get the best available device (MPS > CUDA > CPU)."""
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
