# Pipeline Quick Reference Card

**TL;DR:** One command to generate synthetic PII training datasets!

```bash
python -m model.dataset.pipeline --command=start --num_samples=100
```

> **Note:** The default model is `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`. Use `--api_model="your-model"` to specify a different model.

## ⚡ Commands

```bash
# Start new pipeline (fully automated, auto_poll is True by default)
python -m model.dataset.pipeline --command=start --num_samples=100

# Check status
python -m model.dataset.pipeline --command=status

# Resume interrupted pipeline
python -m model.dataset.pipeline --command=resume

# Reset pipeline (start over)
python -m model.dataset.pipeline --command=reset

# Cancel and delete state
python -m model.dataset.pipeline --command=cancel
```

## 🔧 Common Options

```bash
# Small test (5 samples)
--num_samples=5

# Large production batch
--num_samples=5000

# Different model (default is Qwen/Qwen3-VL-235B-A22B-Instruct-FP8)
--api_model="your-model-name"

# Custom polling (check every 2 minutes)
--poll_interval=120

# More parallel workers
--max_workers=100

# Run without waiting (check status later)
--noauto_poll

# Debug mode
--log_level=DEBUG
```

> **Boolean Flag Syntax:** For boolean flags like `auto_poll`, absl.flags uses:
> - No flag = default value (True for auto_poll)
> - `--flagname` = explicitly True
> - `--noflagname` = explicitly False
> 
> Example: `--noauto_poll` disables automatic polling

## 📝 Setup

```bash
# 1. Set API key
export DOUBLEWORD_API_KEY="your-api-key-here"

# Or add to .env file:
echo "DOUBLEWORD_API_KEY=your-key" >> .env

# 2. Run pipeline (auto_poll defaults to True)
python -m model.dataset.pipeline --command=start --num_samples=100
```

## 📊 Output

```
model/dataset/
├── batch_requests_ner.jsonl           # Generated NER requests
├── batch_requests_coref.jsonl         # Generated coref requests
├── ner_results.jsonl                  # Downloaded NER results
├── coref_results.jsonl                # Downloaded coref results
├── annotation_samples/                # 📁 Final training samples (Label Studio format)
│   ├── final-coref-ner-request-0.json
│   ├── final-coref-ner-request-1.json
│   └── ...
└── .pipeline_state_*.json             # Pipeline state (for resumability)
```

## 🎯 Example Workflows

### Quick Test
```bash
python -m model.dataset.pipeline --command=start --num_samples=10
```

### Production Batch
```bash
python -m model.dataset.pipeline \
  --command=start \
  --num_samples=1000 \
  --max_workers=50 \
  --poll_interval=60
```

### Resume After Interrupt
```bash
# Start pipeline
python -m model.dataset.pipeline --command=start --num_samples=500

# Press Ctrl+C if needed

# Later, resume from where you left off
python -m model.dataset.pipeline --command=resume
```

### Background Job (No Auto-Poll)
```bash
# Submit batches without waiting
python -m model.dataset.pipeline \
  --command=start \
  --num_samples=5000 \
  --noauto_poll

# Check status later
python -m model.dataset.pipeline --command=status

# When ready, resume to download results
python -m model.dataset.pipeline --command=resume
```

## 🐛 Troubleshooting

### "API key not found"
```bash
export DOUBLEWORD_API_KEY="your-key"
```

### "Module not found"
```bash
# Run from project root
cd /path/to/yaak-proxy
python -m model.dataset.pipeline --command=start
```

### Check state file
```bash
cat model/dataset/.pipeline_state_*.json | jq '.'
```

### Debug mode
```bash
python -m model.dataset.pipeline --command=start --log_level=DEBUG
```

## 📖 Full Documentation

- **[PIPELINE_README.md](./PIPELINE_README.md)** - Complete documentation (commands, architecture, troubleshooting)
- **[MIGRATION.md](./MIGRATION.md)** - Migration guide from old scripts
- **[README.md](./README.md)** - Dataset generation overview

## 🎓 5-Minute Tutorial

**Step 1:** Set API key
```bash
export DOUBLEWORD_API_KEY="your-key"
```

**Step 2:** Run small test
```bash
python -m model.dataset.pipeline --command=start --num_samples=5
```

**Step 3:** Check output
```bash
ls -la model/dataset/annotation_samples/
```

**Step 4:** Scale up!
```bash
python -m model.dataset.pipeline --command=start --num_samples=500
```

That's it! ☕

## ⏱️ Time Estimates

| Samples | Time Estimate | Notes |
|---------|---------------|-------|
| 5-10 | ~2-5 minutes | Quick test |
| 50-100 | ~10-20 minutes | Small batch |
| 500-1000 | ~30-60 minutes | Medium batch |
| 5000+ | ~2-4 hours | Large batch |

*Times vary based on API load and model speed*

## 🎯 Common Issues

**"Unknown command line flag 'auto_poll'"**
- Use `--noauto_poll` instead of `--auto_poll=false`
- Boolean flags don't use `=value` syntax in absl.flags

**"Model 'X' has not been configured"**
- The default model is `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`
- Specify a different model with `--api_model="your-model"`
- Check with your API provider which models are available

## 🆘 Need Help?

1. Check [PIPELINE_README.md](./PIPELINE_README.md) for detailed docs
2. Run with `--log_level=DEBUG` to see what's happening
3. Check state: `python -m model.dataset.pipeline --command=status`
4. Look at state file: `cat model/dataset/.pipeline_state_*.json`

## ✅ Success Checklist

- [ ] API key set (`echo $DOUBLEWORD_API_KEY`)
- [ ] Running from project root (`pwd` shows yaak-proxy)
- [ ] Tested with small batch (`--num_samples=5`)
- [ ] Verified output (`ls model/dataset/annotation_samples/`)
- [ ] Ready for production! 🚀

---

**Last Updated:** 2024-01-15  
**Version:** 1.0.0