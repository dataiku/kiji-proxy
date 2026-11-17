#!/bin/bash
# Train mDeBERTa-v3-base on the German training mix, once per seed, and keep
# each ONNX export apart for evaluation.
#
# Usage (from anywhere in the repo, on the GPU machine):
#   ./model/flows/experiments/run_german_mix.sh            # seeds 42 13 7
#   ./model/flows/experiments/run_german_mix.sh 42         # one seed
#
# Expects the mix at model/dataset/data_samples/german_mix_v4/samples, either
# unpacked from german_mix_v4.tar.gz or rebuilt with:
#   uv run python -m model.dataset.build_training_mix
set -euo pipefail

cd "$(dirname "$0")/../../.."

NAME="mdeberta_german_mix_v4"
CONFIG="$PWD/model/flows/experiments/$NAME.toml"
SAMPLES_DIR="model/dataset/data_samples/german_mix_v4/samples"
RESULTS_DIR="model/trained/german_mix"

if [ $# -gt 0 ]; then
    SEEDS=("$@")
else
    SEEDS=(42 13 7)
fi

if [ ! -d "$SAMPLES_DIR" ] || [ -z "$(ls -A "$SAMPLES_DIR")" ]; then
    echo "No training samples in $SAMPLES_DIR" >&2
    exit 1
fi
echo "Training samples: $(ls "$SAMPLES_DIR" | wc -l)"

mkdir -p "$RESULTS_DIR"

for seed in "${SEEDS[@]}"; do
    run="${NAME}_seed${seed}"
    echo "========================================"
    echo "Training $run"
    echo "========================================"
    TRAINING_SEED="$seed" uv run --extra training --extra quantization --extra signing \
        python model/flows/training_pipeline.py --config config-file "$CONFIG" run \
        2>&1 | tee "$RESULTS_DIR/$run.log"

    # The export step always writes to model/quantized; copy it out before the
    # next run overwrites it.
    rm -rf "$RESULTS_DIR/${run}_onnx"
    cp -R model/quantized "$RESULTS_DIR/${run}_onnx"
    echo "ONNX export copied to $RESULTS_DIR/${run}_onnx"
done

echo ""
echo "Done. model/quantized now holds the last run's export, not the shipped"
echo "model; 'git restore model/quantized' puts the shipped one back."
