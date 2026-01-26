"""
Quantize PII Detection Model to ONNX Format

This script:
1. Loads the trained multi-task PII detection model
2. Exports it to ONNX format (PII detection only for compatibility)
3. Quantizes the model for faster inference
4. Saves the quantized model

Usage:
    # Basic usage (uses default paths):
    python quantitize.py

    # With custom paths:
    python quantitize.py --model_path=./model/trained --output_path=./model/quantized

    # With different quantization config:
    python quantitize.py --quantization_mode=avx512_vnni
"""

import json
import os
import sys
from pathlib import Path

import onnx
import torch
from absl import app, flags, logging
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from safetensors import safe_open
from transformers import AutoTokenizer, PreTrainedModel

# Add project root to path for imports BEFORE any local imports
# __file__ is model/src/quantitize.py, so parent.parent.parent is the project root
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from model.src.model_signing import sign_trained_model
except ImportError:
    # Fallback for direct execution - import from same directory
    sys.path.insert(0, str(Path(__file__).parent))
    from model_signing import sign_trained_model

# Define command-line flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "model_path", "./model/trained", "Path to the trained model directory"
)

flags.DEFINE_string(
    "output_path", "./model/quantized", "Path to save the quantized ONNX model"
)

flags.DEFINE_enum(
    "quantization_mode",
    "avx512_vnni",
    ["avx512_vnni", "avx2", "q8"],
    "Quantization mode",
)

flags.DEFINE_integer("opset", 18, "ONNX opset version")

flags.DEFINE_boolean(
    "skip_quantization", False, "Skip quantization, only export to ONNX"
)

flags.DEFINE_boolean(
    "static_quantization",
    False,
    "Use static quantization with calibration data (higher quality but slower)",
)

flags.DEFINE_integer(
    "calibration_samples",
    100,
    "Number of samples to use for static quantization calibration",
)

try:
    from model.src.model import MultiTaskPIIConfig, MultiTaskPIIDetectionModel
except ImportError:
    try:
        from src.model import MultiTaskPIIConfig, MultiTaskPIIDetectionModel
    except ImportError:
        # Fallback to importing from same directory
        sys.path.insert(0, str(Path(__file__).parent))
        from model import MultiTaskPIIConfig, MultiTaskPIIDetectionModel

# absl.logging is already configured, no need for basicConfig


def load_multitask_model(
    model_path: str,
) -> tuple[MultiTaskPIIDetectionModel, dict, AutoTokenizer]:
    """
    Load the multi-task model, label mappings, and tokenizer.

    Args:
        model_path: Path to the model directory

    Returns:
        Tuple of (model, label_mappings, tokenizer)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    logging.info(f"📥 Loading model from: {model_path}")

    # Load label mappings
    mappings_path = model_path / "label_mappings.json"
    if not mappings_path.exists():
        raise FileNotFoundError(f"Label mappings not found at {mappings_path}")

    with mappings_path.open() as f:
        mappings = json.load(f)

    pii_label2id = mappings["pii"]["label2id"]
    pii_id2label = {int(k): v for k, v in mappings["pii"]["id2label"].items()}
    coref_id2label = (
        {int(k): v for k, v in mappings["coref"]["id2label"].items()}
        if "coref" in mappings
        else {0: "NO_COREF", 1: "CLUSTER_0"}
    )

    logging.info(f"✅ Loaded {len(pii_label2id)} PII label mappings")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logging.info("✅ Loaded tokenizer")

    # Load model config
    config_path = model_path / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            model_config = json.load(f)
        # Get base model name from config - prioritize base_model_name field
        base_model_name = model_config.get("base_model_name")
        if not base_model_name:
            # Fall back to _name_or_path for older configs
            base_model_name = model_config.get("_name_or_path")
        if not base_model_name or base_model_name == "multitask_pii":
            # Default if not found or is model_type
            base_model_name = "answerdotai/ModernBERT-base"
    else:
        base_model_name = "answerdotai/ModernBERT-base"
        logging.warning(
            "⚠️  config.json not found, using default: answerdotai/ModernBERT-base"
        )

    # Determine number of labels
    num_pii_labels = len(pii_label2id)
    num_coref_labels = len(coref_id2label)

    # Create config for model
    config = MultiTaskPIIConfig(
        base_model_name=base_model_name,
        num_pii_labels=num_pii_labels,
        num_coref_labels=num_coref_labels,
        id2label_pii=pii_id2label,
        id2label_coref=coref_id2label,
    )

    # Initialize model with config
    model = MultiTaskPIIDetectionModel(config)

    # Load model weights
    model_weights_path = model_path / "pytorch_model.bin"
    if not model_weights_path.exists():
        # Try safetensors format
        model_weights_path = model_path / "model.safetensors"
        if not model_weights_path.exists():
            # Try to find any .bin file
            bin_files = list(model_path.glob("*.bin"))
            if bin_files:
                model_weights_path = bin_files[0]
                logging.info(f"   Found weights: {model_weights_path.name}")

    if model_weights_path.exists():
        logging.info(f"📦 Loading weights from: {model_weights_path.name}")

        # Handle safetensors files
        if model_weights_path.suffix == ".safetensors":
            state_dict = {}
            with safe_open(model_weights_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
        else:
            # Handle .bin files - use weights_only=False for PyTorch 2.6+
            state_dict = torch.load(
                model_weights_path, map_location="cpu", weights_only=False
            )

        # Handle state dict that might have 'model.' prefix
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {
                k.replace("model.", ""): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }

        # Use strict=True to catch any missing/unexpected keys
        try:
            model.load_state_dict(state_dict, strict=True)
            logging.info("✅ Model weights loaded (strict mode)")
        except RuntimeError as e:
            logging.warning(f"⚠️  Strict loading failed: {e}")
            logging.warning("   Falling back to non-strict loading...")
            model.load_state_dict(state_dict, strict=False)
            logging.info("✅ Model weights loaded (non-strict mode)")
    else:
        raise FileNotFoundError(f"Model weights not found in {model_path}")

    model.eval()

    label_mappings = {
        "pii": {"label2id": pii_label2id, "id2label": pii_id2label},
        "coref": {"id2label": coref_id2label},
    }

    return model, label_mappings, tokenizer


class MultiTaskModelWrapper(PreTrainedModel):
    """Wrapper to export multi-task model that returns tuple instead of dict.

    Inherits from PreTrainedModel to maintain compatibility with HuggingFace's
    ONNX export utilities.
    """

    config_class = MultiTaskPIIConfig

    def __init__(self, multitask_model: MultiTaskPIIDetectionModel):
        """Initialize wrapper."""
        super().__init__(multitask_model.config)
        self.model = multitask_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass returning tuple of (pii_logits, coref_logits)."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs["pii_logits"], outputs["coref_logits"]


def export_to_onnx(
    model: MultiTaskPIIDetectionModel,
    tokenizer: AutoTokenizer,
    output_path: str,
    opset: int = 18,
):
    """
    Export the full multi-task model to ONNX (both PII and co-reference detection).

    Args:
        model: The multi-task model
        tokenizer: The tokenizer
        output_path: Path to save the ONNX model
        opset: ONNX opset version
    """
    logging.info("🔄 Exporting multi-task model to ONNX...")

    # Wrap model to return tuple instead of dict (required for ONNX export)
    wrapped_model = MultiTaskModelWrapper(model)
    wrapped_model.eval()

    # Create dummy input for tracing
    dummy_text = "This is a test sentence for ONNX export."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    onnx_path = output_path / "model.onnx"

    # Export full multi-task model to ONNX
    torch.onnx.export(
        wrapped_model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["pii_logits", "coref_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "pii_logits": {0: "batch_size", 1: "sequence_length"},
            "coref_logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    logging.info(f"✅ Multi-task model exported to: {onnx_path}")
    logging.info("   Outputs: pii_logits, coref_logits")

    # Copy tokenizer files to output directory
    logging.info("📋 Copying tokenizer files...")
    tokenizer_files = [
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
        "special_tokens_map.json",
    ]

    # Try to copy from model directory first, then from base model
    for file in tokenizer_files:
        src = (
            Path(tokenizer.name_or_path) / file
            if hasattr(tokenizer, "name_or_path")
            else None
        )
        if not src or not src.exists():
            # Try loading from transformers cache or base model
            try:
                base_tokenizer = AutoTokenizer.from_pretrained(
                    model.encoder.config.name_or_path
                )
                # Tokenizer files are in cache, we'll save them
                base_tokenizer.save_pretrained(str(output_path))
                break
            except Exception:
                pass

    # Save tokenizer to output directory
    tokenizer.save_pretrained(str(output_path))
    logging.info("✅ Tokenizer files saved")

    return str(onnx_path)


def create_calibration_dataset(tokenizer, num_samples: int = 100):
    """
    Create a calibration dataset for static quantization.

    Args:
        tokenizer: The tokenizer to use
        num_samples: Number of calibration samples to generate

    Returns:
        List of calibration samples (dicts with input_ids and attention_mask)
    """
    # Sample texts for calibration - diverse examples to cover different patterns
    calibration_texts = [
        "My name is John Smith and I live at 123 Main Street, New York, NY 10001.",
        "Please contact me at john.smith@email.com or call 555-123-4567.",
        "My social security number is 123-45-6789 and my date of birth is 01/15/1985.",
        "The patient, Jane Doe, was admitted on March 15, 2024 with symptoms of fever.",
        "Account number: 9876543210, routing number: 021000021.",
        "Driver's license number IL-1234-5678-9012 expires on 12/31/2025.",
        "Credit card ending in 4242 was charged $150.00 on Amazon.com.",
        "The meeting is scheduled for tomorrow at 3:00 PM in Conference Room B.",
        "Our company address is 456 Corporate Blvd, Suite 200, Chicago, IL 60601.",
        "For support, email support@company.com or visit www.company.com/help.",
    ]

    calibration_data = []
    for i in range(num_samples):
        text = calibration_texts[i % len(calibration_texts)]
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        calibration_data.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }
        )

    return calibration_data


def quantize_model(
    onnx_path: str,
    output_path: str,
    quantization_mode: str = "avx512_vnni",
    use_static: bool = False,
    tokenizer: AutoTokenizer | None = None,
    calibration_samples: int = 100,
):
    """
    Quantize an ONNX model directory and save the quantized ONNX model to the specified output directory.

    Parameters:
        onnx_path (str): Path to the ONNX model file or to a directory containing ONNX model files.
        output_path (str): Directory where the quantized model and related artifacts will be written.
        quantization_mode (str): Quantization configuration to use. Supported values: "avx512_vnni", "avx2", "q8".
        use_static (bool): If True, use static quantization with calibration data for higher quality.
        tokenizer: Tokenizer for creating calibration data (required if use_static=True).
        calibration_samples (int): Number of samples for static quantization calibration.
    """
    logging.info("🔢 Quantizing model...")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use optimum for quantization
    # ORTQuantizer expects a model directory, not a file path

    # Create quantizer from model directory with explicit file name
    model_dir = Path(onnx_path).parent if Path(onnx_path).is_file() else Path(onnx_path)

    # Remove old quantized model if it exists to avoid "too many ONNX files" error
    old_quantized = model_dir / "model_quantized.onnx"
    if old_quantized.exists():
        logging.info(f"   Removing old quantized model: {old_quantized}")
        old_quantized.unlink()

    quantizer = ORTQuantizer.from_pretrained(str(model_dir), file_name="model.onnx")

    # Select quantization config based on mode and static/dynamic
    is_static = use_static and tokenizer is not None
    if is_static:
        logging.info(
            "   Using STATIC quantization with calibration data (higher quality)"
        )
    else:
        logging.info("   Using DYNAMIC quantization")

    if quantization_mode == "avx512_vnni":
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=is_static)
    elif quantization_mode == "avx2":
        qconfig = AutoQuantizationConfig.avx2(is_static=is_static)
    elif quantization_mode == "q8":
        # q8 doesn't support is_static parameter
        qconfig = (
            AutoQuantizationConfig.arm64(is_static=is_static)
            if is_static
            else AutoQuantizationConfig.arm64(is_static=False)
        )
    else:
        logging.warning(
            f"Unknown quantization mode: {quantization_mode}, using avx512_vnni"
        )
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=is_static)

    logging.info(f"   Using quantization mode: {quantization_mode}")

    # Quantize - with or without calibration data
    if is_static and tokenizer is not None:
        logging.info(
            f"   Creating calibration dataset with {calibration_samples} samples..."
        )
        # Note: Static quantization with calibration data requires additional setup
        # For now, we use dynamic quantization which doesn't need calibration
        logging.warning(
            "   Static quantization not fully implemented, falling back to dynamic"
        )
        quantizer.quantize(save_dir=str(output_path), quantization_config=qconfig)
    else:
        quantizer.quantize(save_dir=str(output_path), quantization_config=qconfig)

    logging.info(f"✅ Quantized model saved to: {output_path}")

    # Load and inspect the quantized model
    quantized_model_path = output_path / "model_quantized.onnx"
    if not quantized_model_path.exists():
        # Try to find any .onnx file
        onnx_files = list(output_path.glob("*.onnx"))
        if onnx_files:
            quantized_model_path = onnx_files[0]
            logging.info(f"   Found quantized model: {quantized_model_path.name}")

    if quantized_model_path.exists():
        model_onnx = onnx.load(str(quantized_model_path))
        logging.info("\n📊 Quantized Model Information:")
        logging.info(f"   Inputs: {[input.name for input in model_onnx.graph.input]}")
        logging.info(
            f"   Outputs: {[output.name for output in model_onnx.graph.output]}"
        )

        # # signing model
        # model_hash = sign_trained_model(quantized_model_path)
        # logging.info(f"   Model hash: {model_hash}")

        # Get model size
        model_size_mb = quantized_model_path.stat().st_size / (1024 * 1024)
        logging.info(f"   Model size: {model_size_mb:.2f} MB")
    else:
        logging.warning("⚠️  Could not find quantized model file")


def main(argv):
    """
    Orchestrates loading a trained multi-task PII detection model, exporting it to ONNX, optionally quantizing the ONNX model, signing and saving artifacts (tokenizer, label mappings, config), and handling errors.

    This function performs high-level orchestration for the CLI: it loads the trained model and tokenizer from FLAGS.model_path, exports the model to ONNX in FLAGS.output_path, signs the exported model, writes label mappings and (if present) the original config.json to the output directory, and — unless --skip_quantization is set — quantizes the ONNX model. On successful quantization the non-quantized ONNX file is removed. Any unhandled exception is logged and causes process exit with code 1.

    Parameters:
        argv: Ignored. Present to match the CLI entrypoint signature.
    """
    del argv  # Unused

    logging.info("=" * 80)
    logging.info("PII Detection Model Quantization")
    logging.info("=" * 80)

    try:
        # Load model
        model, label_mappings, tokenizer = load_multitask_model(FLAGS.model_path)

        # Export to ONNX
        export_to_onnx(model, tokenizer, FLAGS.output_path, FLAGS.opset)
        # signing model
        print(f"__{FLAGS.output_path}__")
        model_hash = sign_trained_model(FLAGS.output_path)
        logging.info(f"   Model hash: {model_hash}")

        # Save label mappings to output directory
        output_path = Path(FLAGS.output_path)
        mappings_path = output_path / "label_mappings.json"
        with mappings_path.open("w") as f:
            json.dump(label_mappings, f, indent=2)
        logging.info(f"✅ Label mappings saved to: {mappings_path}")

        # Copy config.json if it exists
        config_path = Path(FLAGS.model_path) / "config.json"
        if config_path.exists():
            import shutil

            shutil.copy(config_path, output_path / "config.json")
            logging.info("✅ Config file copied")

        # Quantize if requested
        if not FLAGS.skip_quantization:
            # The output_path directory now contains model.onnx, use it for quantization
            quantize_model(
                str(output_path),
                str(output_path),
                FLAGS.quantization_mode,
                use_static=FLAGS.static_quantization,
                tokenizer=tokenizer,
                calibration_samples=FLAGS.calibration_samples,
            )
        else:
            logging.info("⏭️  Skipping quantization (--skip_quantization)")

        logging.info("\n" + "=" * 80)
        logging.info("✅ Quantization Complete!")
        logging.info("=" * 80)
        logging.info(f"Model saved to: {FLAGS.output_path}")
        if FLAGS.skip_quantization:
            logging.info(
                f"saved non-quantized ONNX model: {output_path / 'model.onnx'}"
            )
        else:
            os.remove(output_path / "model.onnx")
            logging.info(
                f"removed non-quantized ONNX model: {output_path / 'model.onnx'}"
            )
        if not FLAGS.skip_quantization:
            logging.info(
                f"saved quantized ONNX model: {output_path / 'model_quantized.onnx'}"
            )

    except Exception as e:
        logging.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    app.run(main)
