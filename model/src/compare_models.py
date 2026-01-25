"""
Compare Trained PyTorch Model with ONNX Model

This script compares the outputs of the trained PyTorch model and the quantized
ONNX model to verify that they produce consistent results.

Usage:
    python -m model.src.compare_models

    # With custom paths:
    python -m model.src.compare_models \
        --trained_model_path=./model/trained \
        --onnx_model_path=./model/quantized

    # Verbose output with detailed comparison:
    python -m model.src.compare_models --verbose
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from absl import app, flags, logging
from safetensors import safe_open
from transformers import AutoTokenizer

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from model.src.model import MultiTaskPIIDetectionModel
except ImportError:
    from .model import MultiTaskPIIDetectionModel

# Define command-line flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "trained_model_path",
    "./model/trained",
    "Path to the trained PyTorch model directory",
)

flags.DEFINE_string(
    "onnx_model_path",
    "./model/quantized",
    "Path to the quantized ONNX model directory",
)

flags.DEFINE_boolean(
    "verbose",
    False,
    "Enable verbose output with detailed comparison",
)

flags.DEFINE_float(
    "tolerance",
    0.1,
    "Tolerance for comparing logits (higher tolerance for quantized models)",
)

# Test cases for comparison
TEST_CASES = [
    "My name is John Smith and my email is john.smith@email.com.",
    "Please contact Sarah Johnson at 555-123-4567.",
    "I live at 123 Main Street, Springfield, IL 62701.",
    "The patient's SSN is 123-45-6789 and DOB is 03/15/1985.",
    "Dr. Emily Chen can be reached at emily.chen@hospital.com.",
]


def get_device():
    """Get the best available device for PyTorch."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_pytorch_model(
    model_path: str,
) -> tuple[MultiTaskPIIDetectionModel, AutoTokenizer, dict]:
    """Load the trained PyTorch model."""
    model_path = Path(model_path)
    logging.info(f"Loading PyTorch model from: {model_path}")

    # Load label mappings
    mappings_path = model_path / "label_mappings.json"
    with mappings_path.open() as f:
        mappings = json.load(f)

    pii_label2id = mappings["pii"]["label2id"]
    pii_id2label = {int(k): v for k, v in mappings["pii"]["id2label"].items()}
    coref_id2label = (
        {int(k): v for k, v in mappings["coref"]["id2label"].items()}
        if "coref" in mappings
        else {0: "NO_COREF", 1: "CLUSTER_0"}
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model config
    config_path = model_path / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            model_config = json.load(f)
        base_model_name = model_config.get(
            "_name_or_path", "answerdotai/ModernBERT-base"
        )
        if base_model_name == "modernbert":
            base_model_name = "answerdotai/ModernBERT-base"
    else:
        base_model_name = "answerdotai/ModernBERT-base"

    # Create model
    num_pii_labels = len(pii_label2id)
    num_coref_labels = len(coref_id2label)

    model = MultiTaskPIIDetectionModel(
        model_name=base_model_name,
        num_pii_labels=num_pii_labels,
        num_coref_labels=num_coref_labels,
        id2label_pii=pii_id2label,
        id2label_coref=coref_id2label,
    )

    # Load weights
    model_weights_path = model_path / "model.safetensors"
    if not model_weights_path.exists():
        model_weights_path = model_path / "pytorch_model.bin"

    if model_weights_path.suffix == ".safetensors":
        state_dict = {}
        with safe_open(model_weights_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        state_dict = torch.load(
            model_weights_path, map_location="cpu", weights_only=False
        )

    # Handle 'model.' prefix
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    logging.info(
        f"  Loaded PyTorch model with {num_pii_labels} PII labels, {num_coref_labels} coref labels"
    )

    return (
        model,
        tokenizer,
        {"pii_id2label": pii_id2label, "coref_id2label": coref_id2label},
    )


def load_onnx_model(
    model_path: str,
) -> tuple[ort.InferenceSession, AutoTokenizer, dict]:
    """Load the ONNX model."""
    model_path = Path(model_path)
    logging.info(f"Loading ONNX model from: {model_path}")

    # Find ONNX model file
    onnx_file = model_path / "model_quantized.onnx"
    if not onnx_file.exists():
        onnx_file = model_path / "model.onnx"
    if not onnx_file.exists():
        onnx_files = list(model_path.glob("*.onnx"))
        if onnx_files:
            onnx_file = onnx_files[0]
        else:
            raise FileNotFoundError(f"No ONNX model found in {model_path}")

    logging.info(f"  Using ONNX file: {onnx_file.name}")

    # Load label mappings
    mappings_path = model_path / "label_mappings.json"
    with mappings_path.open() as f:
        mappings = json.load(f)

    pii_id2label = {int(k): v for k, v in mappings["pii"]["id2label"].items()}
    coref_id2label = (
        {int(k): v for k, v in mappings["coref"]["id2label"].items()}
        if "coref" in mappings
        else {0: "NO_COREF", 1: "CLUSTER_0"}
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Create ONNX Runtime session
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_file),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    # Log model info
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    logging.info(f"  ONNX inputs: {[i.name for i in inputs]}")
    logging.info(f"  ONNX outputs: {[o.name for o in outputs]}")

    return (
        session,
        tokenizer,
        {"pii_id2label": pii_id2label, "coref_id2label": coref_id2label},
    )


def run_pytorch_inference(
    model: MultiTaskPIIDetectionModel,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run inference with PyTorch model."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model = model.to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
        pii_logits = outputs["pii_logits"].cpu().numpy()
        coref_logits = outputs["coref_logits"].cpu().numpy()
    inference_time = (time.perf_counter() - start_time) * 1000

    return pii_logits, coref_logits, inference_time


def run_onnx_inference(
    session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    text: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run inference with ONNX model."""
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512,
    )

    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }

    start_time = time.perf_counter()
    outputs = session.run(None, ort_inputs)
    inference_time = (time.perf_counter() - start_time) * 1000

    # Outputs are [pii_logits, coref_logits]
    pii_logits = outputs[0]
    coref_logits = outputs[1]

    return pii_logits, coref_logits, inference_time


def compare_outputs(
    pytorch_pii: np.ndarray,
    pytorch_coref: np.ndarray,
    onnx_pii: np.ndarray,
    onnx_coref: np.ndarray,
    pytorch_pii_id2label: dict,
    onnx_pii_id2label: dict,
    pytorch_coref_id2label: dict,
    onnx_coref_id2label: dict,
    tolerance: float,
    verbose: bool = False,
) -> dict:
    """Compare PyTorch and ONNX outputs."""
    results = {
        "pii_predictions_match": False,
        "coref_predictions_match": False,
        "pii_logits_close": False,
        "coref_logits_close": False,
        "pii_max_diff": 0.0,
        "coref_max_diff": 0.0,
        "pii_comparable": True,
        "coref_comparable": True,
    }

    # Check if PII dimensions match
    if pytorch_pii.shape[-1] != onnx_pii.shape[-1]:
        logging.warning(
            f"  PII label count mismatch: PyTorch={pytorch_pii.shape[-1]}, ONNX={onnx_pii.shape[-1]}"
        )
        results["pii_comparable"] = False

    # Check if coref dimensions match
    if pytorch_coref.shape[-1] != onnx_coref.shape[-1]:
        logging.warning(
            f"  Coref label count mismatch: PyTorch={pytorch_coref.shape[-1]}, ONNX={onnx_coref.shape[-1]}"
        )
        results["coref_comparable"] = False

    # Get predictions (argmax)
    pytorch_pii_preds = np.argmax(pytorch_pii, axis=-1)
    onnx_pii_preds = np.argmax(onnx_pii, axis=-1)
    pytorch_coref_preds = np.argmax(pytorch_coref, axis=-1)
    onnx_coref_preds = np.argmax(onnx_coref, axis=-1)

    # Compare PII predictions
    if results["pii_comparable"]:
        results["pii_predictions_match"] = np.array_equal(
            pytorch_pii_preds, onnx_pii_preds
        )
        pii_diff = np.abs(pytorch_pii - onnx_pii)
        results["pii_max_diff"] = float(np.max(pii_diff))
        results["pii_mean_diff"] = float(np.mean(pii_diff))
        results["pii_logits_close"] = results["pii_max_diff"] < tolerance
    else:
        # Compare by label name instead of index
        results["pii_predictions_match"] = True
        results["pii_max_diff"] = float("nan")
        results["pii_mean_diff"] = float("nan")
        for batch_idx in range(pytorch_pii_preds.shape[0]):
            for seq_idx in range(pytorch_pii_preds.shape[1]):
                pt_label = pytorch_pii_id2label.get(
                    int(pytorch_pii_preds[batch_idx, seq_idx]), "UNK"
                )
                onnx_label = onnx_pii_id2label.get(
                    int(onnx_pii_preds[batch_idx, seq_idx]), "UNK"
                )
                if pt_label != onnx_label:
                    results["pii_predictions_match"] = False
                    if verbose:
                        logging.info(
                            f"    PII diff at [{batch_idx}, {seq_idx}]: PyTorch={pt_label}, ONNX={onnx_label}"
                        )

    # Compare coref predictions
    if results["coref_comparable"]:
        results["coref_predictions_match"] = np.array_equal(
            pytorch_coref_preds, onnx_coref_preds
        )
        coref_diff = np.abs(pytorch_coref - onnx_coref)
        results["coref_max_diff"] = float(np.max(coref_diff))
        results["coref_mean_diff"] = float(np.mean(coref_diff))
        results["coref_logits_close"] = results["coref_max_diff"] < tolerance
    else:
        # Compare by label name instead of index
        results["coref_predictions_match"] = True
        results["coref_max_diff"] = float("nan")
        results["coref_mean_diff"] = float("nan")
        for batch_idx in range(pytorch_coref_preds.shape[0]):
            for seq_idx in range(pytorch_coref_preds.shape[1]):
                pt_label = pytorch_coref_id2label.get(
                    int(pytorch_coref_preds[batch_idx, seq_idx]), "UNK"
                )
                onnx_label = onnx_coref_id2label.get(
                    int(onnx_coref_preds[batch_idx, seq_idx]), "UNK"
                )
                if pt_label != onnx_label:
                    results["coref_predictions_match"] = False
                    if verbose:
                        logging.info(
                            f"    Coref diff at [{batch_idx}, {seq_idx}]: PyTorch={pt_label}, ONNX={onnx_label}"
                        )

    if verbose and results["pii_comparable"] and not results["pii_predictions_match"]:
        diff_indices = np.where(pytorch_pii_preds != onnx_pii_preds)
        for batch_idx, seq_idx in zip(diff_indices[0], diff_indices[1], strict=True):
            pt_label = pytorch_pii_id2label.get(
                int(pytorch_pii_preds[batch_idx, seq_idx]), "UNK"
            )
            onnx_label = onnx_pii_id2label.get(
                int(onnx_pii_preds[batch_idx, seq_idx]), "UNK"
            )
            logging.info(
                f"    PII diff at [{batch_idx}, {seq_idx}]: PyTorch={pt_label}, ONNX={onnx_label}"
            )

    if (
        verbose
        and results["coref_comparable"]
        and not results["coref_predictions_match"]
    ):
        diff_indices = np.where(pytorch_coref_preds != onnx_coref_preds)
        for batch_idx, seq_idx in zip(diff_indices[0], diff_indices[1], strict=True):
            pt_label = pytorch_coref_id2label.get(
                int(pytorch_coref_preds[batch_idx, seq_idx]), "UNK"
            )
            onnx_label = onnx_coref_id2label.get(
                int(onnx_coref_preds[batch_idx, seq_idx]), "UNK"
            )
            logging.info(
                f"    Coref diff at [{batch_idx}, {seq_idx}]: PyTorch={pt_label}, ONNX={onnx_label}"
            )

    return results


def extract_entities(
    logits: np.ndarray,
    tokenizer: AutoTokenizer,
    text: str,
    id2label: dict,
) -> list[tuple[str, str, int, int]]:
    """Extract entities from logits."""
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    predictions = np.argmax(logits, axis=-1)[0]
    offset_mapping = inputs["offset_mapping"][0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    entities = []
    current_entity = None
    current_label = None
    current_start = None
    current_end = None

    for token, pred, offset in zip(tokens, predictions, offset_mapping, strict=True):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue

        label = id2label.get(int(pred), "O")

        if label.startswith("B-"):
            if current_entity is not None:
                entities.append(
                    (
                        text[current_start:current_end],
                        current_label,
                        current_start,
                        current_end,
                    )
                )
            current_label = label[2:]
            current_start = int(offset[0])
            current_end = int(offset[1])
            current_entity = token
        elif (
            label.startswith("I-")
            and current_entity is not None
            and current_label == label[2:]
        ):
            current_end = int(offset[1])
        elif current_entity is not None:
            entities.append(
                (
                    text[current_start:current_end],
                    current_label,
                    current_start,
                    current_end,
                )
            )
            current_entity = None
            current_label = None

    if current_entity is not None:
        entities.append(
            (text[current_start:current_end], current_label, current_start, current_end)
        )

    return entities


def main(argv):
    """Main function to compare models."""
    del argv  # Unused

    logging.set_verbosity(logging.INFO)

    logging.info("=" * 80)
    logging.info("Model Comparison: PyTorch vs ONNX")
    logging.info("=" * 80)

    # Check paths
    trained_path = Path(FLAGS.trained_model_path)
    onnx_path = Path(FLAGS.onnx_model_path)

    if not trained_path.exists():
        logging.error(f"Trained model path not found: {trained_path}")
        sys.exit(1)
    if not onnx_path.exists():
        logging.error(f"ONNX model path not found: {onnx_path}")
        sys.exit(1)

    # Load models
    logging.info("\nLoading models...")
    pytorch_model, pytorch_tokenizer, pytorch_labels = load_pytorch_model(
        str(trained_path)
    )
    onnx_session, onnx_tokenizer, onnx_labels = load_onnx_model(str(onnx_path))

    device = get_device()
    logging.info(f"\nPyTorch device: {device}")
    logging.info(f"Tolerance for comparison: {FLAGS.tolerance}")

    # Run comparison
    logging.info(f"\nRunning comparison on {len(TEST_CASES)} test cases...")

    all_results = []
    pytorch_times = []
    onnx_times = []

    for i, text in enumerate(TEST_CASES, 1):
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Test Case {i}: {text[:50]}...")

        # Run inference
        pytorch_pii, pytorch_coref, pytorch_time = run_pytorch_inference(
            pytorch_model, pytorch_tokenizer, text, device
        )
        onnx_pii, onnx_coref, onnx_time = run_onnx_inference(
            onnx_session, onnx_tokenizer, text
        )

        pytorch_times.append(pytorch_time)
        onnx_times.append(onnx_time)

        # Compare outputs
        results = compare_outputs(
            pytorch_pii,
            pytorch_coref,
            onnx_pii,
            onnx_coref,
            pytorch_labels["pii_id2label"],
            onnx_labels["pii_id2label"],
            pytorch_labels["coref_id2label"],
            onnx_labels["coref_id2label"],
            FLAGS.tolerance,
            FLAGS.verbose,
        )
        all_results.append(results)

        # Display results
        pii_match = "MATCH" if results["pii_predictions_match"] else "DIFFER"
        coref_match = "MATCH" if results["coref_predictions_match"] else "DIFFER"

        if results["pii_comparable"]:
            logging.info(
                f"  PII predictions: {pii_match} (max diff: {results['pii_max_diff']:.4f}, mean: {results['pii_mean_diff']:.4f})"
            )
        else:
            logging.info(
                f"  PII predictions: {pii_match} (label comparison only - dimensions differ)"
            )

        if results["coref_comparable"]:
            logging.info(
                f"  Coref predictions: {coref_match} (max diff: {results['coref_max_diff']:.4f}, mean: {results['coref_mean_diff']:.4f})"
            )
        else:
            logging.info(
                f"  Coref predictions: {coref_match} (label comparison only - dimensions differ)"
            )
        logging.info(
            f"  Inference time: PyTorch={pytorch_time:.2f}ms, ONNX={onnx_time:.2f}ms"
        )

        if FLAGS.verbose:
            # Extract and display entities
            pytorch_entities = extract_entities(
                pytorch_pii, pytorch_tokenizer, text, pytorch_labels["pii_id2label"]
            )
            onnx_entities = extract_entities(
                onnx_pii, onnx_tokenizer, text, onnx_labels["pii_id2label"]
            )

            logging.info("\n  PyTorch entities:")
            for entity_text, label, start, end in pytorch_entities:
                logging.info(f"    [{label}] '{entity_text}' ({start}-{end})")

            logging.info("\n  ONNX entities:")
            for entity_text, label, start, end in onnx_entities:
                logging.info(f"    [{label}] '{entity_text}' ({start}-{end})")

    # Summary
    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY")
    logging.info("=" * 80)

    pii_match_count = sum(1 for r in all_results if r["pii_predictions_match"])
    coref_match_count = sum(1 for r in all_results if r["coref_predictions_match"])
    total_cases = len(all_results)

    logging.info("\nPrediction Matching:")
    logging.info(
        f"  PII predictions match: {pii_match_count}/{total_cases} ({100 * pii_match_count / total_cases:.1f}%)"
    )
    logging.info(
        f"  Coref predictions match: {coref_match_count}/{total_cases} ({100 * coref_match_count / total_cases:.1f}%)"
    )

    # Check if logit comparison is possible
    pii_comparable = all(r["pii_comparable"] for r in all_results)
    coref_comparable = all(r["coref_comparable"] for r in all_results)

    logging.info("\nLogit Differences:")
    if pii_comparable:
        pii_diffs = [r["pii_max_diff"] for r in all_results]
        avg_pii_diff = np.mean(pii_diffs)
        max_pii_diff = max(pii_diffs)
        logging.info(
            f"  PII - avg max diff: {avg_pii_diff:.4f}, overall max: {max_pii_diff:.4f}"
        )
    else:
        max_pii_diff = float("inf")
        logging.info("  PII - N/A (different label counts between models)")

    if coref_comparable:
        coref_diffs = [r["coref_max_diff"] for r in all_results]
        avg_coref_diff = np.mean(coref_diffs)
        max_coref_diff = max(coref_diffs)
        logging.info(
            f"  Coref - avg max diff: {avg_coref_diff:.4f}, overall max: {max_coref_diff:.4f}"
        )
    else:
        max_coref_diff = float("inf")
        logging.info("  Coref - N/A (different label counts between models)")

    logging.info("\nInference Performance:")
    logging.info(
        f"  PyTorch - avg: {np.mean(pytorch_times):.2f}ms, min: {np.min(pytorch_times):.2f}ms, max: {np.max(pytorch_times):.2f}ms"
    )
    logging.info(
        f"  ONNX - avg: {np.mean(onnx_times):.2f}ms, min: {np.min(onnx_times):.2f}ms, max: {np.max(onnx_times):.2f}ms"
    )
    logging.info(f"  Speedup: {np.mean(pytorch_times) / np.mean(onnx_times):.2f}x")

    # Overall status
    all_pii_match = pii_match_count == total_cases
    all_coref_match = coref_match_count == total_cases

    logging.info("\n" + "=" * 80)

    # Warn if models have different architectures
    if not pii_comparable or not coref_comparable:
        logging.warning("WARNING: Models have different output dimensions!")
        logging.warning(
            "  The ONNX model was likely exported from a different training run."
        )
        logging.warning("  Re-export with: uv run python -m model.src.quantitize")

    if all_pii_match and all_coref_match:
        logging.info("RESULT: All predictions match between PyTorch and ONNX models")
    else:
        logging.info("RESULT: Some predictions differ (may be due to quantization)")
        if pii_comparable and coref_comparable:
            if max_pii_diff < FLAGS.tolerance and max_coref_diff < FLAGS.tolerance:
                logging.info(
                    f"  However, all logit differences are within tolerance ({FLAGS.tolerance})"
                )
    logging.info("=" * 80)


if __name__ == "__main__":
    app.run(main)
