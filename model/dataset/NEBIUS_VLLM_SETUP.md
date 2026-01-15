# Running GPT-OSS-120b on Nebius with vLLM

This guide walks you through setting up and running OpenAI's GPT-OSS-120b model on Nebius cloud infrastructure using vLLM to generate training data for the Yaak Proxy model.

## Prerequisites

- Access to a Nebius GPU instance (recommended: multi-GPU setup with sufficient VRAM for 120B model)
- SSH access to your Nebius instance
- Python 3.12 or higher

## Step 1: Verify GPU Setup

First, verify that your GPU(s) are properly configured:

```bash
nvidia-smi
```

You should see your GPU(s) listed with available memory. The GPT-OSS-120b model requires significant VRAM (recommended: multiple A100 or H100 GPUs).

## Step 2: Install System Dependencies

Update your system and install required build tools:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-dev build-essential
```

## Step 3: Install vLLM

Install vLLM with the necessary wheels for GPU support. This uses the nightly PyTorch builds with CUDA 12.8 support:

```bash
uv pip install --pre vllm==0.13.0 \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match
```

> **Note:** The `--pre` flag allows pre-release versions, and the custom index URLs provide optimized builds for GPT-OSS models.

## Step 4: Serve the Model

You have two options for serving the model, depending on your use case:

### Option A: Basic Configuration (Lower Throughput)

For development or testing with lower concurrency:

```bash
uv run vllm serve openai/gpt-oss-120b \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 8
```

**Parameters:**
- `--max-model-len 4096`: Maximum sequence length (context window)
- `--gpu-memory-utilization 0.9`: Use 90% of available GPU memory
- `--max-num-seqs 8`: Process up to 8 sequences in parallel

### Option B: High Throughput Configuration (Recommended for Production)

For generating large training sets efficiently:

```bash
uv run --extra training vllm serve openai/gpt-oss-120b \
  --max-num-seqs 128 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --async-scheduling
```

**Parameters:**
- `--max-num-seqs 128`: Process up to 128 sequences in parallel (high throughput)
- `--async-scheduling`: Enable asynchronous scheduling for better performance
- `--extra training`: Include training-specific dependencies

The server will start on `http://localhost:8000` by default and provide an OpenAI-compatible API.

## Step 5: Generate Training Data

Once the model is running, generate training samples using the training set script.

### Generate a Fixed Number of Samples

To generate exactly 10,000 samples:

```bash
uv run model/dataset/training_set.py \
  --api_url=http://localhost:8000/v1/chat/completions \
  --max_workers=16 \
  --num_samples=10000
```

### Resume Training Set Generation

If you've already generated some samples and want to reach 10,000 total:

```bash
uv run model/dataset/training_set.py \
  --api_url=http://localhost:8000/v1/chat/completions \
  --max_workers=16 \
  --num_samples=$((10000 - $(find model/dataset/reviewed_samples/ -maxdepth 1 -type f | wc -l)))
```

**Parameters:**
- `--api_url`: The vLLM server endpoint (OpenAI-compatible)
- `--max_workers=16`: Number of parallel workers for concurrent generation
- `--num_samples`: Total number of samples to generate

## Tips for Optimal Performance

1. **Monitor GPU Usage**: Keep `nvidia-smi` running in a separate terminal to monitor GPU utilization
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Adjust Concurrency**: If you see OOM errors, reduce `--max-num-seqs` or `--gpu-memory-utilization`

3. **Network Latency**: Run the training set script on the same machine as the vLLM server to minimize latency

4. **Batch Size**: The `--max_workers` parameter should be tuned based on your `--max-num-seqs` setting and available system resources

## Output

Generated samples are saved to:
- `model/dataset/training_samples/` - Raw generated samples
- `model/dataset/reviewed_samples/` - Reviewed and validated samples

Each sample is saved as a JSON file with a timestamp and hash-based filename.

## Troubleshooting

### CUDA Out of Memory

- Reduce `--gpu-memory-utilization` (try 0.85 or 0.8)
- Reduce `--max-num-seqs`
- Reduce `--max-model-len` if you don't need the full 4096 context

### Slow Generation

- Ensure you're using the high throughput configuration
- Increase `--max-num-seqs` if you have available GPU memory
- Use `--async-scheduling` for better throughput

### Connection Refused

- Ensure the vLLM server is fully started (check for "Application startup complete" in logs)
- Verify the server is listening on the correct port (default: 8000)
- Check firewall rules if running on different machines

## Cost Considerations

Running a 120B model on Nebius can be expensive. Monitor your usage and consider:
- Stopping the instance when not in use
- Generating samples in batches rather than all at once

## Next Steps

After generating your training set:
1. Review samples in `model/dataset/reviewed_samples/`
2. Use Label Studio for annotation (see `model/dataset/labelstudio/README.md`)
3. Train your model using the generated dataset (see `model/flows/README.md`)
