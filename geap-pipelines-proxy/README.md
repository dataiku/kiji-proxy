# geap-pipelines-proxy

KFP pipeline definitions that run the Kiji PII model trainer on Vertex AI
Pipelines. Standalone uv project — deliberately **not** a workspace member of
the root `kiji-proxy` project, because the two resolve incompatible `protobuf`
majors (this project needs 6.x for `kfp`/`google-cloud-aiplatform`; the trainer
pins 3.x/4.x alongside `torch` and `transformers`).

## What it produces

`kiji_training_pipeline.yaml` — the pipeline spec Vertex AI Pipelines consumes.

```bash
uv run geap-pipelines-proxy                       # -> kiji_training_pipeline.yaml
uv run geap-pipelines-proxy -o /tmp/pipeline.yaml
```

## Shape of the pipeline

Two steps:

1. **`render-training-config`** — a lightweight component that renders
   `training_config.toml` from the pipeline parameters and emits it as an
   artifact, so the exact hyperparameters of a run land in Vertex lineage.
2. **`train-pii-model`** — the `575lab/kiji-proxy:dev` image (built from
   `model/Dockerfile`), running the Metaflow flow
   `model/flows/training_pipeline.py`.

The flow is monolithic — export → preprocess → train → evaluate → ONNX export →
sign all happen in one process — so it is one KFP step, not one step per stage.
Splitting it into separate Vertex steps would mean restructuring the Metaflow
flow, not the pipeline definition.

The step overrides the image ENTRYPOINT with a small `sh` wrapper that:

- invokes the flow with `--config config-file <artifact path>` (Metaflow's
  syntax for supplying a value to `Config("config-file", ...)`, same as
  `model/flows/run_training.sh`);
- copies `/workspace/model/trained` and `/workspace/model/quantized` to the
  output artifact paths once training finishes.

Two reasons for the copy rather than writing straight to GCS:

- `model/quantized` is hardcoded in `training_pipeline.py` (`exported_output =
  "model/quantized"`), so unlike `paths.output_dir` it cannot be redirected
  through the config file.
- Training against the gcsfuse mount is slow and trips over the checkpoint
  rename HuggingFace `Trainer` performs between epochs. Local disk, then one
  copy, is the safer pattern.

## Submitting

```bash
uv run geap-pipelines-submit \
  --project my-gcp-project \
  --location us-central1 \
  --pipeline-root gs://my-bucket/kiji-pipelines \
  --training-samples-uri gs://my-bucket/data/training_samples \
  --service-account vertex-trainer@my-gcp-project.iam.gserviceaccount.com \
  --subsample-count 500        # smoke run
```

`gs://` inputs are rewritten to the `/gcs/` gcsfuse mount that Vertex exposes
inside the container.

## Parameters

Defaults mirror `model/flows/training_config.toml` as committed —
`microsoft/deberta-v3-base`, 30 epochs, batch 128, lr 2e-5, bf16 on, early
stopping at patience 3. `training_samples_uri` is the only required parameter.

`skip_export` is forced to `true`: Label Studio is not reachable from a Vertex
worker pool, so datasets are staged to GCS ahead of the run.

## Before the first real run

- **Image reachability.** `575lab/kiji-proxy:dev` must be public on Docker Hub,
  or mirrored into Artifact Registry and `TRAINER_IMAGE` in `pipeline.py`
  repointed at `<region>-docker.pkg.dev/<project>/<repo>/kiji-proxy:dev`.
  Artifact Registry is the better default — unauthenticated Docker Hub pulls
  from GCP ranges hit rate limits and fail intermittently.
- **Container user vs. gcsfuse.** `model/Dockerfile` ends with `USER trainer`
  (uid 1000). Vertex's `/gcs` fuse mount is owned by root, and a non-root
  process may not be able to write the output artifacts. If the copy step fails
  with `Permission denied`, that is the cause — either drop `USER trainer` for
  the Vertex variant of the image, or mount with `allow_other`-equivalent
  permissions. This is the most likely first-run failure.
- **Accelerator.** Pinned to one A100 because the committed config sets
  `bf16 = true`, which needs Ampere or newer. Dropping to a T4 requires setting
  `bf16 = false` too.
- **Boot disk.** The image is multi-GB and the HF cache plus checkpoints add
  more; Vertex's default 100 GB boot disk is adequate but not generous.
- **Signing.** `skip_signing` defaults to `false`, and
  `MODEL_SIGNING_KEY_PATH` is not wired up, so runs sign hash-only. Mount a key
  via Secret Manager if you need private-key signatures.
