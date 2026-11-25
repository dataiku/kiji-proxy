#!/bin/bash
set -e

echo "========================================"
echo "PII Detection Model Training Pipeline"
echo "========================================"
echo ""

# Default to local run
CONFIG_FLAG=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: ./run_training.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE                   Use custom config file (default: training_config.toml)"
            echo "  --num-samples N                 Limit local training samples (0 = all, default: 0)"
            echo "  --num-samples-ai4privacy N      Add N samples from ai4privacy/pii-masking-300k (0 = all)"
            echo "  -h, --help                      Show this help"
            echo ""
            echo "Examples:"
            echo "  ./run_training.sh                                    # Run with default config"
            echo "  ./run_training.sh --config prod.toml                 # Use custom config"
            echo "  ./run_training.sh --num-samples 5000                 # Use 5k local samples"
            echo "  ./run_training.sh --num-samples-ai4privacy 5000      # Add 5k ai4privacy samples"
            echo "  ./run_training.sh --num-samples-ai4privacy 0         # Add all ai4privacy samples"
            echo ""
            echo "Note: Edit training_config.toml to change epochs, batch size, etc."
            exit 0
            ;;
        --config)
            # Resolve to absolute path
            CONFIG_PATH="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
            CONFIG_FLAG="--config config-file $CONFIG_PATH"
            shift 2
            ;;
        --num-samples)
            export NUM_SAMPLES="$2"
            shift 2
            ;;
        --num-samples-ai4privacy)
            export NUM_AI4PRIVACY_SAMPLES="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "Running training pipeline..."
echo ""

# Run from project root
uv run --extra training --extra quantization --extra signing \
    python model/flows/training_pipeline.py $CONFIG_FLAG run $EXTRA_ARGS

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
