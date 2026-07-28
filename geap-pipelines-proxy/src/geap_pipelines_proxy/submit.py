"""Submit the compiled pipeline to Vertex AI Pipelines."""

from __future__ import annotations

import argparse
from typing import Any

from geap_pipelines_proxy.pipeline import compile_pipeline


def submit(
    project: str,
    location: str,
    pipeline_root: str,
    training_samples_uri: str,
    template_path: str | None = None,
    display_name: str = "kiji-pii-model-training",
    service_account: str | None = None,
    parameter_values: dict[str, Any] | None = None,
    enable_caching: bool = False,
) -> Any:
    """Compile (if needed) and launch a PipelineJob.

    ``pipeline_root`` is the gs:// prefix Vertex uses for artifacts; it must be
    writable by the runtime service account.
    """
    from google.cloud import aiplatform

    if template_path is None:
        template_path = compile_pipeline()

    params: dict[str, Any] = {"training_samples_uri": training_samples_uri}
    params.update(parameter_values or {})

    aiplatform.init(project=project, location=location)
    job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=template_path,
        pipeline_root=pipeline_root,
        parameter_values=params,
        enable_caching=enable_caching,
    )
    job.submit(service_account=service_account)
    return job


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="geap-pipelines-submit",
        description="Submit the Kiji training pipeline to Vertex AI Pipelines.",
    )
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="Vertex region")
    parser.add_argument(
        "--pipeline-root",
        required=True,
        help="gs:// prefix for pipeline artifacts",
    )
    parser.add_argument(
        "--training-samples-uri",
        required=True,
        help="gs:// path to the training_samples directory",
    )
    parser.add_argument(
        "--template-path",
        default=None,
        help="Existing pipeline YAML; compiled fresh when omitted",
    )
    parser.add_argument("--service-account", default=None)
    parser.add_argument(
        "--subsample-count",
        type=int,
        default=None,
        help="Limit training samples for a smoke run (0 = use all)",
    )
    parser.add_argument("--num-epochs", type=int, default=None)
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.subsample_count is not None:
        overrides["subsample_count"] = args.subsample_count
    if args.num_epochs is not None:
        overrides["num_epochs"] = args.num_epochs

    job = submit(
        project=args.project,
        location=args.location,
        pipeline_root=args.pipeline_root,
        training_samples_uri=args.training_samples_uri,
        template_path=args.template_path,
        service_account=args.service_account,
        parameter_values=overrides,
    )
    print(f"Submitted: {job.resource_name}")
