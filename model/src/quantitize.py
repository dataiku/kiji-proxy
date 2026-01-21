"""
Quantize PII Detection Model to ONNX Format

This script:
1. Loads the trained PII detection model
2. Exports it to ONNX format
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
from transformers import AutoTokenizer

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
    "arm64",
    ["avx512_vnni", "avx2", "arm64", "none"],
    "Quantization mode: avx512_vnni/avx2 for x86, arm64 for Apple Silicon, none for FP32",
)

flags.DEFINE_integer("opset", 18, "ONNX opset version")

flags.DEFINE_boolean(
    "skip_quantization", False, "Skip quantization, only export to ONNX"
)

try:
    from model.src.model import PIIDetectionModel
except ImportError:
    # Fallback to importing from same directory
    sys.path.insert(0, str(Path(__file__).parent))
    from model import PIIDetectionModel

# absl.logging is already configured, no need for basicConfig


def load_model(
    model_path: str,
) -> tuple[PIIDetectionModel, dict, AutoTokenizer]:
    """
    Load the PII detection model, label mappings, and tokenizer.

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

    label2id = mappings["label2id"]
    id2label = {int(k): v for k, v in mappings["id2label"].items()}

    logging.info(f"✅ Loaded {len(label2id)} label mappings")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logging.info("✅ Loaded tokenizer")

    # Use ModernBERT as the base model
    base_model_name = "answerdotai/ModernBERT-base"

    # Determine number of labels
    num_labels = len(label2id)

    # Load model
    model = PIIDetectionModel(
        model_name=base_model_name,
        num_labels=num_labels,
        id2label=id2label,
    )

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

        # Log state dict keys for debugging
        logging.info(f"   State dict contains {len(state_dict)} keys")
        sample_keys = list(state_dict.keys())[:5]
        logging.info(f"   Sample keys: {sample_keys}")

        # Handle state dict that might have 'model.' prefix (from Trainer wrapping)
        if any(k.startswith("model.") for k in state_dict.keys()):
            logging.info("   Removing 'model.' prefix from state dict keys")
            state_dict = {
                k.replace("model.", ""): v
                for k, v in state_dict.items()
                if k.startswith("model.")
            }

        # Get model's expected keys
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())

        # Check for mismatches
        missing_keys = model_keys - loaded_keys
        unexpected_keys = loaded_keys - model_keys

        if missing_keys:
            logging.warning(
                f"   ⚠️  Missing keys ({len(missing_keys)}): {list(missing_keys)[:5]}..."
            )
        if unexpected_keys:
            logging.warning(
                f"   ⚠️  Unexpected keys ({len(unexpected_keys)}): {list(unexpected_keys)[:5]}..."
            )

        # Check encoder weights specifically
        encoder_keys_loaded = [k for k in loaded_keys if k.startswith("encoder.")]
        encoder_keys_expected = [k for k in model_keys if k.startswith("encoder.")]
        logging.info(
            f"   Encoder weights: {len(encoder_keys_loaded)} loaded, {len(encoder_keys_expected)} expected"
        )

        if len(encoder_keys_loaded) == 0 and len(encoder_keys_expected) > 0:
            logging.error(
                "   ❌ NO ENCODER WEIGHTS FOUND! Model will use random/pretrained encoder weights."
            )
            logging.error("   This will cause training/inference mismatch!")

        # Load weights - use strict=True to catch issues
        try:
            model.load_state_dict(state_dict, strict=True)
            logging.info("✅ Model weights loaded (strict mode)")
        except RuntimeError as e:
            logging.warning(f"   Strict loading failed: {e}")
            logging.warning("   Falling back to non-strict loading...")
            model.load_state_dict(state_dict, strict=False)
            logging.info("✅ Model weights loaded (non-strict mode)")
    else:
        raise FileNotFoundError(f"Model weights not found in {model_path}")

    model.eval()

    label_mappings = {"label2id": label2id, "id2label": id2label}

    return model, label_mappings, tokenizer


class ModelWrapper(torch.nn.Module):
    """Wrapper to export model that returns tensor instead of dict."""

    def __init__(self, model: PIIDetectionModel):
        """Initialize wrapper."""
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass returning logits tensor."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs["logits"]


def export_to_onnx(
    model: PIIDetectionModel,
    tokenizer: AutoTokenizer,
    output_path: str,
    opset: int = 18,
):
    """
    Export the PII detection model to ONNX.

    Args:
        model: The PII detection model
        tokenizer: The tokenizer
        output_path: Path to save the ONNX model
        opset: ONNX opset version
    """
    logging.info("🔄 Exporting model to ONNX...")

    # Wrap model to return tensor instead of dict (required for ONNX export)
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    # Create dummy input for tracing
    dummy_text = "This is a test sentence for ONNX export."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    onnx_path = output_path / "model.onnx"

    # Export model to ONNX
    torch.onnx.export(
        wrapped_model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    logging.info(f"✅ Model exported to: {onnx_path}")
    logging.info("   Output: logits")

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


def quantize_model(
    onnx_path: str,
    output_path: str,
    quantization_mode: str = "avx512_vnni",
):
    """
    Quantize an ONNX model directory and save the quantized ONNX model to the specified output directory.

    Parameters:
        onnx_path (str): Path to the ONNX model file or to a directory containing ONNX model files. If a file path is provided, its parent directory will be used.
        output_path (str): Directory where the quantized model and related artifacts will be written. The directory will be created if it does not exist.
        quantization_mode (str): Quantization configuration to use. Supported values include "avx512_vnni", "avx2", and "q8"; unknown values default to "avx512_vnni".

    """
    logging.info("🔢 Quantizing model...")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use optimum for quantization
    # ORTQuantizer expects a model directory, not a file path

    # Create quantizer from model directory with explicit file name
    model_dir = Path(onnx_path).parent if Path(onnx_path).is_file() else Path(onnx_path)

    # Remove any existing quantized model to avoid "too many ONNX files" error
    existing_quantized = model_dir / "model_quantized.onnx"
    if existing_quantized.exists():
        existing_quantized.unlink()
        logging.info(f"   Removed existing: {existing_quantized.name}")

    quantizer = ORTQuantizer.from_pretrained(str(model_dir), file_name="model.onnx")

    # Select quantization config based on mode
    # Exclude classification head from quantization to preserve accuracy
    nodes_to_exclude = [
        "classifier",
        "/classifier/",
    ]

    if quantization_mode == "avx512_vnni":
        qconfig = AutoQuantizationConfig.avx512_vnni(
            is_static=False,
            nodes_to_exclude=nodes_to_exclude,
        )
    elif quantization_mode == "avx2":
        qconfig = AutoQuantizationConfig.avx2(
            is_static=False,
            nodes_to_exclude=nodes_to_exclude,
        )
    elif quantization_mode == "arm64":
        qconfig = AutoQuantizationConfig.arm64(
            is_static=False,
            nodes_to_exclude=nodes_to_exclude,
        )
    elif quantization_mode == "none":
        # Skip quantization, just copy the model
        logging.info("   Skipping quantization (mode=none), using FP32 model")
        import shutil

        src_model = model_dir / "model.onnx"
        dst_model = output_path / "model_quantized.onnx"
        if src_model.exists():
            shutil.copy(src_model, dst_model)
            # Also copy external data if present
            src_data = model_dir / "model.onnx.data"
            if src_data.exists():
                shutil.copy(src_data, output_path / "model.onnx.data")
            logging.info(f"✅ Copied FP32 model to: {dst_model}")
        return
    else:
        logging.warning(f"Unknown quantization mode: {quantization_mode}, using arm64")
        qconfig = AutoQuantizationConfig.arm64(
            is_static=False,
            nodes_to_exclude=nodes_to_exclude,
        )

    logging.info(f"   Using quantization mode: {quantization_mode}")

    # Quantize
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

        # Get model size
        model_size_mb = quantized_model_path.stat().st_size / (1024 * 1024)
        logging.info(f"   Model size: {model_size_mb:.2f} MB")
    else:
        logging.warning("⚠️  Could not find quantized model file")


def main(argv):
    """
    Orchestrates loading a trained PII detection model, exporting it to ONNX, optionally quantizing the ONNX model, signing and saving artifacts (tokenizer, label mappings, config), and handling errors.

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
        model, label_mappings, tokenizer = load_model(FLAGS.model_path)

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
            quantize_model(str(output_path), str(output_path), FLAGS.quantization_mode)
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
