"""Compile Vertex AI Pipelines definitions for the Kiji PII trainer."""

from __future__ import annotations

import argparse

from geap_pipelines_proxy.pipeline import compile_pipeline

__all__ = ["compile_pipeline", "main"]


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="geap-pipelines-proxy",
        description="Compile the Kiji training pipeline to Vertex AI YAML.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="kiji_training_pipeline.yaml",
        help="Path to write the compiled pipeline YAML (default: %(default)s)",
    )
    args = parser.parse_args()

    path = compile_pipeline(args.output)
    print(f"Compiled pipeline -> {path}")
