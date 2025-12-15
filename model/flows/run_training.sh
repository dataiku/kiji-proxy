#!/bin/bash
set -e

echo "========================================"
echo "PII Detection Model Training Pipeline"
echo "========================================"
echo ""
echo "Configuration is loaded from training_config.toml"
echo "Override with CLI parameters or --config flag"
echo ""

# Parse flags
SKIP_QUANT=""
SKIP_SIGN=""
QUICK_MODE=""
CUSTOM_CONFIG=""
EXTRA_PARAMS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: ./run_training.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick              Quick test (1 epoch, skip quant/sign)"
            echo "  --skip-quantization  Skip ONNX quantization"
            echo "  --skip-signing       Skip model signing"
            echo "  --config FILE        Use custom config file"
            echo "  --num-epochs N       Override number of epochs"
            echo "  --batch-size N       Override batch size"
            echo "  -h, --help           Show this help"
            echo ""
            echo "Examples:"
            echo "  ./run_training.sh                    # Use defaults from training_config.toml"
            echo "  ./run_training.sh --quick            # Quick test run"
            echo "  ./run_training.sh --num-epochs 10    # Override epochs"
            echo "  ./run_training.sh --config prod.toml # Use custom config"
            exit 0
            ;;
        --quick)
            QUICK_MODE="true"
            SKIP_QUANT="--skip-quantization true"
            SKIP_SIGN="--skip-signing true"
            EXTRA_PARAMS="$EXTRA_PARAMS --num-epochs 1"
            shift
            ;;
        --skip-quantization)
            SKIP_QUANT="--skip-quantization true"
            shift
            ;;
        --skip-signing)
            SKIP_SIGN="--skip-signing true"
            shift
            ;;
        --config)
            CUSTOM_CONFIG="$2"
            shift 2
            ;;
        --num-epochs)
            EXTRA_PARAMS="$EXTRA_PARAMS --num-epochs $2"
            shift 2
            ;;
        --batch-size)
            EXTRA_PARAMS="$EXTRA_PARAMS --batch-size $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Run mode: ${QUICK_MODE:+Quick Test}${QUICK_MODE:-Full Training}"
echo ""

# Build the command - flow is at project root
CMD="python training_pipeline.py --environment=fast-bakery"

# Config goes BEFORE run command
if [ -n "$CUSTOM_CONFIG" ]; then
    echo "Using custom config: $CUSTOM_CONFIG"
    CMD="$CMD --config config-file $CUSTOM_CONFIG"
fi

CMD="$CMD run"

# Parameters go AFTER run command
CMD="$CMD $SKIP_QUANT $SKIP_SIGN $EXTRA_PARAMS"

echo "Executing: $CMD"
echo ""

# Run the Metaflow pipeline
eval $CMD

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"

# Show the results
echo ""
echo "View latest run:"
echo "  python training_pipeline.py show latest"
echo ""
echo "View Metaflow Card:"
echo "  python training_pipeline.py card view latest"
