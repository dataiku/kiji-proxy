# Pipeline Quick Reference Card

**TL;DR:** One command to generate synthetic PII training datasets!

```bash
python -m model.dataset.doubleword.pipeline --command=start --num_samples=100
```

> **Note:** The default model is `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`. Use `--api_model="your-model"` to specify a different model.

> **Quality Enhancement:** Add `--enable_review` to include an optional review stage that validates and improves sample quality!

## ⚡ Commands

```bash
# Start new pipeline (fully automated, auto_poll is True by default)
python -m model.dataset.doubleword.pipeline --command=start --num_samples=100

# Check status
python -m model.dataset.doubleword.pipeline --command=status

# Resume interrupted pipeline
python -m model.dataset.doubleword.pipeline --command=resume

# Reset pipeline (start over)
python -m model.dataset.doubleword.pipeline --command=reset

# Cancel and delete state
python -m model.dataset.doubleword.pipeline --command=cancel
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

# Enable optional review stage for quality improvement
--enable_review
```

> **Boolean Flag Syntax:** For boolean flags like `auto_poll` and `enable_review`, absl.flags uses:
> - No flag = default value (True for auto_poll, False for enable_review)
> - `--flagname` = explicitly True
> - `--noflagname` = explicitly False
> 
> Example: `--noauto_poll` disables automatic polling, `--enable_review` enables quality review

## 📝 Setup

```bash
# 1. Set API key
export DOUBLEWORD_API_KEY="your-api-key-here"

# Or add to .env file:
echo "DOUBLEWORD_API_KEY=your-key" >> .env

# 2. Run pipeline (auto_poll defaults to True)
python -m model.dataset.doubleword.pipeline --command=start --num_samples=100
```

## 📊 Output

```
model/dataset/
├── doubleword/
│   ├── pipeline.py                    # Main pipeline orchestrator
│   ├── batch_generator.py             # Batch request generation
│   ├── batch_monitor.py               # Batch job monitoring
│   ├── doubleword_client.py           # API client
│   ├── pipeline_state.py              # State management
│   ├── result_processor.py            # Result processing
│   └── temp/                          # Temporary files
│       ├── batch_requests_ner.jsonl   # Generated NER requests
│       ├── batch_requests_coref.jsonl # Generated coref requests
│       ├── ner_results.jsonl          # Downloaded NER results
│       ├── coref_results.jsonl        # Downloaded coref results
│       └── .pipeline_state_*.json     # Pipeline state (for resumability)
└── data_samples/
    └── annotation_samples/            # Final training samples (Label Studio format)
        ├── final-coref-ner-request-0.json
        ├── final-coref-ner-request-1.json
        └── ...
```

## 🎯 Example Workflows

### Quick Test
```bash
python -m model.dataset.doubleword.pipeline --command=start --num_samples=10
```

### Production Batch
```bash
python -m model.dataset.doubleword.pipeline \
  --command=start \
  --num_samples=1000 \
  --max_workers=50 \
  --poll_interval=60
```

### High-Quality Production Batch (with Review)
```bash
python -m model.dataset.doubleword.pipeline \
  --command=start \
  --num_samples=1000 \
  --enable_review \
  --max_workers=50
```

### Resume After Interrupt
```bash
# Start pipeline
python -m model.dataset.doubleword.pipeline --command=start --num_samples=500

# Press Ctrl+C if needed

# Later, resume from where you left off
python -m model.dataset.doubleword.pipeline --command=resume
```

### Background Job (No Auto-Poll)
```bash
# Submit batches without waiting
python -m model.dataset.doubleword.pipeline \
  --command=start \
  --num_samples=5000 \
  --noauto_poll

# Check status later
python -m model.dataset.doubleword.pipeline --command=status

# When ready, resume to download results
python -m model.dataset.doubleword.pipeline --command=resume
```

## 🐛 Troubleshooting

### "API key not found"
```bash
export DOUBLEWORD_API_KEY="your-key"
```

### "Module not found"
```bash
# Run from project root
cd /path/to/kiji-proxy
python -m model.dataset.doubleword.pipeline --command=start
```

### Check state file
```bash
cat model/dataset/doubleword/temp/.pipeline_state_*.json | jq '.'
```

### Debug mode
```bash
python -m model.dataset.doubleword.pipeline --command=start --log_level=DEBUG
```

## 📖 Full Documentation

- **[PIPELINE_README.md](./PIPELINE_README.md)** - Complete documentation (commands, architecture, troubleshooting)
- **[README.md](./README.md)** - Dataset generation overview

## 🎓 5-Minute Tutorial

**Step 1:** Set API key
```bash
export DOUBLEWORD_API_KEY="your-key"
```

**Step 2:** Run small test
```bash
python -m model.dataset.doubleword.pipeline --command=start --num_samples=5
```

**Step 3:** Check output
```bash
ls -la model/dataset/data_samples/annotation_samples/
```

**Step 4:** Scale up!
```bash
python -m model.dataset.doubleword.pipeline --command=start --num_samples=500
```

That's it!

## ⏱️ Time Estimates

| Samples | Time Estimate (No Review) | Time Estimate (With Review) | Notes |
|---------|---------------------------|----------------------------|-------|
| 5-10 | ~2-5 minutes | ~3-7 minutes | Quick test |
| 50-100 | ~10-20 minutes | ~15-30 minutes | Small batch |
| 500-1000 | ~30-60 minutes | ~45-90 minutes | Medium batch |
| 5000+ | ~2-4 hours | ~3-6 hours | Large batch |

*Times vary based on API load and model speed. Review adds ~30-50% more time.*

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
3. Check state: `python -m model.dataset.doubleword.pipeline --command=status`
4. Look at state file: `cat model/dataset/doubleword/temp/.pipeline_state_*.json`

## ✅ Success Checklist

- [ ] API key set (`echo $DOUBLEWORD_API_KEY`)
- [ ] Running from project root (`pwd` shows kiji-proxy)
- [ ] Tested with small batch (`--num_samples=5`)
- [ ] Verified output (`ls model/dataset/data_samples/annotation_samples/`)
- [ ] Ready for production!

---

## 🔍 Optional: Review & Quality Improvement

The pipeline generates high-quality samples, but you can optionally add a **review stage** to validate and improve them:

### What is the Review Stage?

When you add `--enable_review`, the pipeline includes two additional stages:
1. **Review Submission**: Creates review prompts to validate/correct annotations
2. **Review Completion**: Processes reviewed and corrected samples

This adds quality control directly into the pipeline flow:

```
NER -> Coref -> Review (optional) -> Final Output
```

### When to Use It?

- **Production datasets** where quality is critical
- When you notice annotation errors in samples
- To improve label consistency across the dataset
- For datasets that will be used for model evaluation

### How to Use It

Simply add the `--enable_review` flag:

```bash
# Standard pipeline (no review)
python -m model.dataset.doubleword.pipeline --command=start --num_samples=100

# With review for higher quality
python -m model.dataset.doubleword.pipeline \
  --command=start \
  --num_samples=100 \
  --enable_review
```

### Pipeline Flow with Review

```
Stage 1: NER Generation
Stage 2: NER Completion
Stage 3: Coref Generation
Stage 4: Coref Completion
Stage 5: Review Generation (if --enable_review)
Stage 6: Review Completion (if --enable_review)
Stage 7: Final Processing
```

### Time Estimate

Review adds approximately 30-50% more time to the pipeline:
- Without review: ~30 min for 100 samples
- With review: ~45 min for 100 samples

**Note:** The review stage is **optional**. Most users can skip it for testing and prototyping.

---

**Last Updated:** 2024-01-15  
**Version:** 1.0.0