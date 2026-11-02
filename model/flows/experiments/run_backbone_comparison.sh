#!/bin/bash
# Train the multilingual and the current English backbone on the same data and
# settings, and keep each ONNX export apart for evaluation.
#
# Usage (from anywhere in the repo):
#   ./model/flows/experiments/run_backbone_comparison.sh                   # both
#   ./model/flows/experiments/run_backbone_comparison.sh mdeberta_v3_base  # one
set -euo pipefail

cd "$(dirname "$0")/../../.."

EXP_DIR="model/flows/experiments"
RESULTS_DIR="model/trained/backbone_comparison"
SAMPLES_DIR="model/dataset/data_samples/training_samples"

if [ $# -gt 0 ]; then
    RUNS=("$@")
else
    RUNS=(mdeberta_v3_base deberta_v3_base_baseline)
fi

mkdir -p "$RESULTS_DIR"

if [ ! -d "$SAMPLES_DIR" ] || [ -z "$(ls -A "$SAMPLES_DIR")" ]; then
    echo "Downloading training data to $SAMPLES_DIR ..."
    uv run python model/dataset/huggingface/download_dataset_from_hf.py \
        --repo-id "DataikuNLP/kiji-pii-training-data"
fi

for name in "${RUNS[@]}"; do
    config="$PWD/$EXP_DIR/$name.toml"
    if [ ! -f "$config" ]; then
        echo "No config $config" >&2
        exit 1
    fi

    echo "========================================"
    echo "Training $name"
    echo "========================================"
    uv run --extra training --extra quantization --extra signing \
        python model/flows/training_pipeline.py --config config-file "$config" run \
        2>&1 | tee "$RESULTS_DIR/$name.log"

    # The export step always writes to model/quantized, so copy each export
    # out before the next run overwrites it.
    rm -rf "$RESULTS_DIR/${name}_onnx"
    cp -R model/quantized "$RESULTS_DIR/${name}_onnx"
    echo "ONNX export copied to $RESULTS_DIR/${name}_onnx"
done

echo ""
echo "Done. Note: model/quantized now holds the last run's export, not the"
echo "shipped model; 'git restore model/quantized' puts the shipped one back."
