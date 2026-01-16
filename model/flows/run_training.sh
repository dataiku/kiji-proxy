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
            echo "  --config FILE   Use custom config file (default: training_config.toml)"
            echo "  -h, --help      Show this help"
            echo ""
            echo "Examples:"
            echo "  ./run_training.sh                       # Run with default config"
            echo "  ./run_training.sh --config prod.toml    # Use custom config"
            echo ""
            echo "Note: Edit training_config.toml to change epochs, batch size, etc."
            exit 0
            ;;
        --config)
            CONFIG_FLAG="--config config-file $2"
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
uv run --extra training --extra signing \
    python model/flows/training_pipeline.py $CONFIG_FLAG run $EXTRA_ARGS

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
