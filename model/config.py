"""Configuration and environment setup for training."""
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for PII detection model training."""

    # Model settings
    model_name: str = "distilbert-base-cased"  # 66M params, fast

    # Training parameters
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 3e-5
    max_samples: int = 400000

    # Training optimization
    warmup_steps: int = 500
    weight_decay: float = 0.01
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    seed: int = 42

    # Output and logging
    output_dir: str = "./pii_model"
    use_wandb: bool = False
    use_custom_loss: bool = True
    class_weights: dict[int, float] = field(default_factory=dict)

    # Dataset settings
    eval_size_ratio: float = 0.2  # Validation set size as ratio of training
    max_sequence_length: int = 512
    training_samples_dir: str = "dataset/training_samples"

    # Multi-task learning
    pii_loss_weight: float = 1.0  # Weight for PII detection loss
    coref_loss_weight: float = 1.0  # Weight for co-reference detection loss

    def __post_init__(self):
        """Create output directory after initialization."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def print_summary(self):
        """Print configuration summary."""
        logger.info("\n📋 Training Configuration:")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Batch Size: {self.batch_size}")
        logger.info(f"  Learning Rate: {self.learning_rate}")
        logger.info(f"  Max Samples: {self.max_samples}")
        logger.info(f"  Output Dir: {self.output_dir}")
        logger.info(f"  Custom Loss: {self.use_custom_loss}")


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
            logger.info(f"✅ Google Drive mounted at {mount_point}")
            return True
        except ImportError:
            logger.warning("⚠️  Not running in Google Colab - skipping Drive mount")
            return False
        except Exception:
            logger.exception("❌ Failed to mount Google Drive")
            return False

    @staticmethod
    def disable_wandb():
        """Disable Weights & Biases to avoid API key prompts."""
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_PROJECT"] = ""
        os.environ["WANDB_ENTITY"] = ""
        logger.info("✅ Weights & Biases (wandb) disabled")

    @staticmethod
    def install_package(package_list: list[str], index_url: str | None = None):
        """Install packages with optional index URL."""
        cmd = [sys.executable, "-m", "pip", "install", "-q"]
        if index_url:
            cmd.extend(["--index-url", index_url])
        cmd.extend(package_list)

        try:
            subprocess.check_call(cmd)
            logger.info(f"✅ Successfully installed: {', '.join(package_list)}")
        except subprocess.CalledProcessError:
            logger.exception(f"❌ Failed to install: {', '.join(package_list)}")
            if index_url:
                logger.info("Trying fallback installation...")
                cmd_fallback = [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    *package_list,
                ]
                subprocess.check_call(cmd_fallback)
                logger.info("✅ Fallback installation successful")
            else:
                raise

    @staticmethod
    def setup_pytorch():
        """Install PyTorch with CUDA support if available."""
        logger.info("Installing PyTorch...")
        try:
            EnvironmentSetup.install_package(
                ["torch", "torchvision", "torchaudio"],
                index_url="https://download.pytorch.org/whl/cu118",
            )
        except Exception:
            logger.warning("CUDA installation failed, installing CPU version...")
            EnvironmentSetup.install_package(["torch", "torchvision", "torchaudio"])

    @staticmethod
    def setup_dependencies():
        """Install all required dependencies."""
        logger.info("Installing required packages...")
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
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        cuda_available = torch.cuda.is_available()

        if mps_available:
            logger.info(f"\n✅ MPS (Metal) available: {torch.backends.mps.is_available()}")
            logger.info("   Using Apple Silicon GPU acceleration")
            logger.info("   Device: mps")
        elif cuda_available:
            logger.info(f"\n✅ CUDA available: {cuda_available}")
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
            logger.info("   Device: cuda")
        else:
            logger.info("\n⚠️  No GPU available - using CPU")
            logger.info(f"   CUDA available: {cuda_available}")
            logger.info(f"   MPS available: {mps_available}")
            logger.info("   Device: cpu")

    @staticmethod
    def get_device():
        """Get the best available device (MPS > CUDA > CPU)."""
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

