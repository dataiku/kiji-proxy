"""
Quantize PII Detection Model to ONNX Format

This script:
1. Loads the trained PII detection model
2. Exports it to ONNX format
3. Optionally writes a quantized side artifact
4. Keeps model.onnx as the default exported model

Usage:
    # Basic usage (uses default paths):
    python quantitize.py

    # With custom paths:
    python quantitize.py --model_path=./model/trained --output_path=./model/quantized

    # With different quantization config:
    python quantitize.py --quantization_mode=avx512_vnni
"""

import json
import sys
from pathlib import Path

import torch
from absl import app, flags, logging
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

try:
    from model.src.checkpoint_utils import load_compatible_state_dict
except ImportError:
    from checkpoint_utils import load_compatible_state_dict

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
    ["avx512_vnni", "avx2", "q8", "fp16"],
    "Quantization mode. 'fp16' converts weights+activations to float16 instead "
    "of running an INT8 quantizer; useful for GPU inference (CUDA / TensorRT EP).",
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

    pii_label2id = mappings["pii"]["label2id"]
    pii_id2label = {int(k): v for k, v in mappings["pii"]["id2label"].items()}

    logging.info(f"✅ Loaded {len(pii_label2id)} PII label mappings")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logging.info("✅ Loaded tokenizer")

    # Load model config. Quantization is DeBERTa-only going forward; avoid
    # silently falling back across model families because that produces opaque
    # embedding-size errors when an old checkpoint is passed in.
    config_path = model_path / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            model_config = json.load(f)
        model_type = model_config.get("model_type", "")
        if model_type not in {"deberta-v2", "deberta"}:
            raise RuntimeError(
                f"Unsupported model_type {model_type!r} in {config_path}. "
                "Quantization only supports DeBERTa checkpoints."
            )
        base_model_name = (
            model_config.get("_name_or_path") or "microsoft/deberta-v3-base"
        )
        if base_model_name in {"deberta-v2", "deberta"}:
            base_model_name = "microsoft/deberta-v3-base"
    else:
        tokenizer_class = tokenizer.__class__.__name__
        if "deberta" not in tokenizer_class.lower():
            raise RuntimeError(
                f"config.json not found in {model_path}, and the tokenizer class "
                f"is {tokenizer_class}, not DeBERTa. This looks like a stale "
                "non-DeBERTa checkpoint; re-run training with a DeBERTa base model "
                "or pass --model_path to a DeBERTa checkpoint directory."
            )
        base_model_name = "microsoft/deberta-v3-base"
        logging.warning(
            "⚠️  config.json not found, using default: microsoft/deberta-v3-base"
        )

    # Determine number of labels
    num_pii_labels = len(pii_label2id)

    # Load PII detection model
    model = PIIDetectionModel(
        model_name=base_model_name,
        num_pii_labels=num_pii_labels,
        id2label_pii=pii_id2label,
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

        load_info = load_compatible_state_dict(
            model,
            state_dict,
            source=str(model_weights_path),
        )
        if load_info.unexpected_keys:
            logging.warning(
                "⚠️  Ignoring unexpected checkpoint keys: %s",
                load_info.unexpected_keys[:20],
            )
        logging.info("✅ Model weights loaded")
    else:
        raise FileNotFoundError(f"Model weights not found in {model_path}")

    model.eval()

    label_mappings = {
        "pii": {"label2id": pii_label2id, "id2label": pii_id2label},
    }

    return model, label_mappings, tokenizer


class ModelWrapper(torch.nn.Module):
    """Wrapper to export model that returns tensor instead of dict."""

    def __init__(self, model: PIIDetectionModel):
        """Initialize wrapper."""
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Forward pass returning pii_logits tensor."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs["pii_logits"]


def export_to_onnx(
    model: PIIDetectionModel,
    tokenizer: AutoTokenizer,
    output_path: str,
    opset: int = 18,
):
    """
    Export the PII detection model to ONNX format.

    Args:
        model: The PII detection model
        tokenizer: The tokenizer
        output_path: Path to save the ONNX model
        opset: ONNX opset version
    """
    logging.info("🔄 Exporting PII detection model to ONNX...")

    # Wrap model to return tensor instead of dict (required for ONNX export)
    wrapped_model = ModelWrapper(model)
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

    # Export PII detection model to ONNX
    # Use dynamo=False to force the legacy TorchScript-based exporter,
    # which is much faster for DeBERTa's disentangled attention layers.
    torch.onnx.export(
        wrapped_model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["pii_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "pii_logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    logging.info(f"✅ PII detection model exported to: {onnx_path}")
    logging.info("   Outputs: pii_logits")

    # Export CRF transition parameters for Viterbi decoding in Go
    if hasattr(model, "crf"):
        crf_params = {
            "transitions": model.crf.transitions.detach().cpu().numpy().tolist(),
            "start_transitions": model.crf.start_transitions.detach()
            .cpu()
            .numpy()
            .tolist(),
            "end_transitions": model.crf.end_transitions.detach()
            .cpu()
            .numpy()
            .tolist(),
        }
        crf_path = output_path / "crf_transitions.json"
        with open(crf_path, "w") as f:
            json.dump(crf_params, f)
        logging.info(f"✅ CRF transition parameters exported to: {crf_path}")

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

    # Remove truncation from tokenizer.json — the Go backend handles chunking
    # itself and the tokenizer's built-in truncation silently drops tokens
    # beyond 512, causing PII at the end of long texts to be missed.
    tokenizer_json_path = Path(output_path) / "tokenizer.json"
    if tokenizer_json_path.exists():
        with open(tokenizer_json_path) as f:
            tok_data = json.load(f)
        if tok_data.get("truncation") is not None:
            tok_data["truncation"] = None
            with open(tokenizer_json_path, "w") as f:
                json.dump(tok_data, f, indent=2, ensure_ascii=False)
            logging.info(
                "Removed truncation from tokenizer.json (handled by Go chunking)"
            )

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
        quantization_mode (str): Compression to apply.
            - "avx512_vnni" / "avx2" / "q8": INT8 dynamic quantization via Optimum.
            - "fp16": float16 conversion via onnxruntime's float16 converter
              (input/output dtypes preserved). Recommended for CUDA / TensorRT EPs.
            Unknown INT8 values default to "avx512_vnni".

    """
    logging.info("🔢 Quantizing model...")

    import shutil

    import onnx

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ORTQuantizer expects a model directory, not a file path.
    model_dir = Path(onnx_path).parent if Path(onnx_path).is_file() else Path(onnx_path)

    # Dtype-suffixed sibling so multiple builds (fp16 + int8 variants) can
    # co-exist in one directory. ``model_quantized.onnx`` is also written as a
    # backwards-compatible alias pointing at this same artifact for downstream
    # consumers (Go backend, smoke tests, training pipeline).
    if quantization_mode == "fp16":
        dtype_filename = "model_fp16.onnx"
    else:
        dtype_filename = f"model_int8_{quantization_mode}.onnx"
    dtype_path = output_path / dtype_filename
    alias_path = output_path / "model_quantized.onnx"

    # Remove the alias to avoid Optimum's "too many ONNX files" error and to
    # ensure the alias reflects the build we're about to produce. Existing
    # dtype-suffixed siblings from prior builds are preserved.
    for stale in (alias_path, output_path / "model_quantized.onnx"):
        if stale.exists() and stale != dtype_path:
            logging.info(f"   Removing previous alias: {stale}")
            stale.unlink()

    logging.info(f"   Using quantization mode: {quantization_mode}")

    if quantization_mode == "fp16":
        # Float16 conversion — keeps model layout identical, just shrinks weights
        # and activations to fp16. keep_io_types=True leaves input_ids /
        # attention_mask / pii_logits as their original dtypes so callers don't
        # need to change.
        from onnxruntime.transformers.float16 import convert_float_to_float16

        source_onnx = model_dir / "model.onnx"
        model_proto = onnx.load(str(source_onnx))
        fp16_proto = convert_float_to_float16(model_proto, keep_io_types=True)
        onnx.save(fp16_proto, str(dtype_path))
    else:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(str(model_dir), file_name="model.onnx")

        if quantization_mode == "avx512_vnni":
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
        elif quantization_mode == "avx2":
            qconfig = AutoQuantizationConfig.avx2(is_static=False)
        elif quantization_mode == "q8":
            qconfig = AutoQuantizationConfig.q8()
        else:
            logging.warning(
                f"Unknown quantization mode: {quantization_mode}, using avx512_vnni"
            )
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

        # Optimum writes to ``<save_dir>/model_quantized.onnx``. Move it to the
        # dtype-suffixed name; the alias copy is created below.
        quantizer.quantize(save_dir=str(output_path), quantization_config=qconfig)
        optimum_output = output_path / "model_quantized.onnx"
        if optimum_output.exists() and optimum_output != dtype_path:
            optimum_output.replace(dtype_path)

    # Maintain ``model_quantized.onnx`` as the canonical alias for the most
    # recent build, for backwards compatibility with downstream consumers.
    shutil.copy2(dtype_path, alias_path)

    logging.info(f"✅ Quantized model saved to: {dtype_path}")
    logging.info(f"   Alias for backwards compat: {alias_path}")

    if dtype_path.exists():
        model_onnx = onnx.load(str(dtype_path))
        logging.info("\n📊 Quantized Model Information:")
        logging.info(f"   Inputs: {[input.name for input in model_onnx.graph.input]}")
        logging.info(
            f"   Outputs: {[output.name for output in model_onnx.graph.output]}"
        )

        model_size_mb = dtype_path.stat().st_size / (1024 * 1024)
        logging.info(f"   Model size: {model_size_mb:.2f} MB")
    else:
        logging.warning("⚠️  Could not find quantized model file")

    return dtype_path


def main(argv):
    """
    Orchestrates loading a trained PII detection model, exporting it to ONNX, optionally quantizing the ONNX model, signing and saving artifacts (tokenizer, label mappings, config), and handling errors.

    This function performs high-level orchestration for the CLI: it loads the trained model and tokenizer from FLAGS.model_path, exports the model to ONNX in FLAGS.output_path, signs the exported model, writes label mappings and (if present) the original config.json to the output directory, and — unless --skip_quantization is set — writes a quantized side artifact. The non-quantized model.onnx remains the default production model. Any unhandled exception is logged and causes process exit with code 1.

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
        dtype_artifact: Path | None = None
        if not FLAGS.skip_quantization:
            # The output_path directory now contains model.onnx, use it for quantization
            dtype_artifact = quantize_model(
                str(output_path), str(output_path), FLAGS.quantization_mode
            )
        else:
            logging.info("⏭️  Skipping quantization (--skip_quantization)")

        logging.info("\n" + "=" * 80)
        logging.info("✅ Quantization Complete!")
        logging.info("=" * 80)
        logging.info(f"Model saved to: {FLAGS.output_path}")
        logging.info(f"saved default ONNX model: {output_path / 'model.onnx'}")
        if dtype_artifact is not None:
            logging.info(f"saved quantized side artifact: {dtype_artifact}")
            logging.info(
                f"  + alias for backwards compat: {output_path / 'model_quantized.onnx'}"
            )

    except Exception as e:
        logging.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    app.run(main)
