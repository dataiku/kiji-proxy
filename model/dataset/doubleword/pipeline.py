"""Main pipeline orchestrator for dataset generation."""

import sys
from pathlib import Path

from absl import app, flags, logging
from absl.flags import DuplicateFlagError
from dotenv import load_dotenv

# Add project root to sys.path for imports when running as a script
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from ..file_operations import FileManager
    from .batch_generator import BatchRequestGenerator
    from .batch_monitor import BatchMonitor
    from .doubleword_client import DoublewordClient
    from .pipeline_state import PipelineState
    from .result_processor import ResultProcessor
except ImportError:
    from batch_generator import BatchRequestGenerator
    from batch_monitor import BatchMonitor
    from doubleword_client import DoublewordClient
    from pipeline_state import PipelineState
    from result_processor import ResultProcessor

    from model.dataset.file_operations import FileManager

# Load .env file from root directory
root_dir = Path(__file__).parent.parent.parent.parent
env_path = root_dir / ".env"
load_dotenv(env_path)

# Define absl flags
FLAGS = flags.FLAGS


# Define flags - use individual try/except to handle duplicates gracefully
def _define_flag(define_func, *args, **kwargs):
    """Helper to define a flag with duplicate handling."""
    try:
        define_func(*args, **kwargs)
    except DuplicateFlagError:
        pass


_define_flag(
    flags.DEFINE_enum,
    "command",
    "start",
    ["start", "status", "resume", "reset", "cancel"],
    "Command to execute: start (new pipeline), status (check status), "
    "resume (continue pipeline), reset (clear state), cancel (delete state)",
)
_define_flag(flags.DEFINE_integer, "num_samples", 100, "Number of samples to generate")
_define_flag(
    flags.DEFINE_string,
    "api_model",
    "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "Model name to use for generation",
)
_define_flag(
    flags.DEFINE_string,
    "output_dir",
    "model/dataset/doubleword/temp",
    "Output directory for temporary batch files and pipeline state",
)
_define_flag(
    flags.DEFINE_string,
    "log_level",
    "INFO",
    "Logging level (DEBUG, INFO, WARNING, ERROR)",
)
_define_flag(
    flags.DEFINE_string,
    "doubleword_api_key",
    None,
    "Doubleword API key (if not set, will look for DOUBLEWORD_API_KEY env var)",
)
_define_flag(
    flags.DEFINE_integer,
    "poll_interval",
    60,
    "Seconds between batch status checks (default: 60)",
)
_define_flag(
    flags.DEFINE_integer,
    "timeout",
    None,
    "Maximum seconds to wait for batch completion (None for no timeout)",
)
_define_flag(
    flags.DEFINE_boolean,
    "auto_poll",
    True,
    "Automatically poll and wait for batch completion",
)
_define_flag(
    flags.DEFINE_integer,
    "max_workers",
    None,
    "Maximum number of parallel workers (default: min(32, num_samples + 4))",
)
_define_flag(
    flags.DEFINE_string,
    "pipeline_id",
    None,
    "Pipeline ID to resume (if not provided, uses most recent)",
)
_define_flag(
    flags.DEFINE_boolean,
    "enable_review",
    False,
    "Enable optional review stage for quality improvement",
)


class DatasetPipeline:
    """Orchestrates the full dataset generation pipeline."""

    def __init__(
        self,
        api_key: str,
        api_model: str,
        output_dir: str,
        num_samples: int,
        max_workers: int | None = None,
        pipeline_id: str | None = None,
        enable_review: bool = False,
    ):
        """
        Initialize dataset pipeline.

        Args:
            api_key: Doubleword API key
            api_model: Model name to use for generation
            output_dir: Output directory for all files
            num_samples: Number of samples to generate
            max_workers: Maximum number of parallel workers
            pipeline_id: Optional pipeline ID to resume
            enable_review: Whether to enable optional review stage
        """
        self.api_key = api_key
        self.api_model = api_model
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.max_workers = max_workers
        self.enable_review = enable_review

        # Initialize components
        self.client = DoublewordClient(api_key)
        self.generator = BatchRequestGenerator(
            api_model, num_samples, str(output_dir), max_workers
        )
        self.monitor = BatchMonitor(self.client)
        self.processor = ResultProcessor(FileManager(str(output_dir)))
        self.state = PipelineState(str(output_dir), pipeline_id)

    def run(
        self,
        auto_poll: bool = True,
        poll_interval: int = 60,
        timeout: int | None = None,
    ) -> list[str]:
        """
        Run the complete pipeline.

        Args:
            auto_poll: Whether to automatically poll for results
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait for batch completion

        Returns:
            List of generated training sample files
        """
        logging.info(f"Starting pipeline: {self.state.pipeline_id}")
        logging.info(f"Number of samples: {self.num_samples}")

        # Stage 1: Generate and submit NER batch
        if not self.state.is_stage_complete("ner_submission"):
            logging.info("=" * 80)
            logging.info("Stage 1: Generating NER batch requests...")
            logging.info("=" * 80)

            is_testing = self.num_samples <= 3
            ner_batch_path = self.generator.generate_ner_batch_requests(
                output_file="batch_requests_ner.jsonl",
                is_testing=is_testing,
                language_count=5,
            )

            logging.info("Submitting NER batch to Doubleword...")
            file_id, batch_id = self.client.submit_batch(str(ner_batch_path))

            self.state.save_stage(
                "ner_submission",
                {
                    "batch_id": batch_id,
                    "file_id": file_id,
                    "batch_file": str(ner_batch_path),
                },
            )

            logging.info(f"✓ NER batch submitted: {batch_id}")

        # Stage 2: Wait for NER completion and download
        if not self.state.is_stage_complete("ner_completion"):
            ner_info = self.state.get_stage("ner_submission")
            if not ner_info:
                raise RuntimeError(
                    "NER submission stage not found. Run with --command=start first."
                )
            batch_id = ner_info["batch_id"]

            if auto_poll:
                logging.info("=" * 80)
                logging.info("Stage 2: Waiting for NER batch completion...")
                logging.info("=" * 80)
                logging.info(f"Batch ID: {batch_id}")
                logging.info(f"Poll interval: {poll_interval}s")

                try:
                    ner_content = self.monitor.wait_for_completion(
                        batch_id, poll_interval, timeout
                    )
                except KeyboardInterrupt:
                    logging.warning("Interrupted by user. Progress saved.")
                    logging.info("Run with --command=resume to continue later.")
                    return []

                # Save results
                ner_results_path = self.output_dir / "ner_results.jsonl"
                ner_results_path.write_text(ner_content)

                self.state.save_stage(
                    "ner_completion", {"results_path": str(ner_results_path)}
                )

                logging.info(f"✓ NER results downloaded: {ner_results_path}")
            else:
                logging.info("=" * 80)
                logging.info("Stage 2: NER batch pending")
                logging.info("=" * 80)
                logging.info(f"NER Batch ID: {batch_id}")
                logging.info("Run with --command=resume when batch is complete")
                return []

        # Stage 3: Generate and submit coreference batch
        if not self.state.is_stage_complete("coref_submission"):
            logging.info("=" * 80)
            logging.info("Stage 3: Generating coreference batch requests...")
            logging.info("=" * 80)

            ner_completion = self.state.get_stage("ner_completion")
            if not ner_completion:
                raise RuntimeError(
                    "NER completion stage not found. Ensure NER stage completed."
                )
            ner_results_path = ner_completion["results_path"]

            # Parse NER results
            with open(ner_results_path) as f:
                ner_content = f.read()

            ner_samples = self.processor.parse_ner_results(ner_content)
            logging.info(f"Parsed {len(ner_samples)} NER samples")

            # Generate coreference batch
            coref_batch_path = self.generator.generate_coref_batch_requests(
                ner_samples, output_file="batch_requests_coref.jsonl"
            )

            logging.info("Submitting coreference batch to Doubleword...")
            file_id, batch_id = self.client.submit_batch(str(coref_batch_path))

            self.state.save_stage(
                "coref_submission",
                {
                    "batch_id": batch_id,
                    "file_id": file_id,
                    "batch_file": str(coref_batch_path),
                },
            )

            logging.info(f"✓ Coreference batch submitted: {batch_id}")

        # Stage 4: Wait for coref completion and download
        if not self.state.is_stage_complete("coref_completion"):
            coref_info = self.state.get_stage("coref_submission")
            if not coref_info:
                raise RuntimeError("Coref submission stage not found.")
            batch_id = coref_info["batch_id"]

            if auto_poll:
                logging.info("=" * 80)
                logging.info("Stage 4: Waiting for coreference batch completion...")
                logging.info("=" * 80)
                logging.info(f"Batch ID: {batch_id}")
                logging.info(f"Poll interval: {poll_interval}s")

                try:
                    coref_content = self.monitor.wait_for_completion(
                        batch_id, poll_interval, timeout
                    )
                except KeyboardInterrupt:
                    logging.warning("Interrupted by user. Progress saved.")
                    logging.info("Run with --command=resume to continue later.")
                    return []

                # Save results
                coref_results_path = self.output_dir / "coref_results.jsonl"
                coref_results_path.write_text(coref_content)

                self.state.save_stage(
                    "coref_completion", {"results_path": str(coref_results_path)}
                )

                logging.info(f"✓ Coreference results downloaded: {coref_results_path}")
            else:
                logging.info("=" * 80)
                logging.info("Stage 4: Coreference batch pending")
                logging.info("=" * 80)
                logging.info(f"Coreference Batch ID: {batch_id}")
                logging.info("Run with --command=resume when batch is complete")
                return []

        # Stage 5: Optional Review (if enabled)
        if self.enable_review:
            if not self.state.is_stage_complete("review_submission"):
                logging.info("=" * 80)
                logging.info("Stage 5: Generating review batch requests (OPTIONAL)...")
                logging.info("=" * 80)

                coref_completion = self.state.get_stage("coref_completion")
                if not coref_completion:
                    raise RuntimeError("Coref completion stage not found.")
                coref_results_path = coref_completion["results_path"]

                # Parse coref results for review
                with open(coref_results_path) as f:
                    coref_content = f.read()

                samples_for_review = self.processor.parse_samples_for_review(
                    coref_content
                )
                logging.info(f"Parsed {len(samples_for_review)} samples for review")

                # Generate review batch
                review_batch_path = self.generator.generate_review_batch_requests(
                    samples_for_review, output_file="batch_requests_review.jsonl"
                )

                logging.info("Submitting review batch to Doubleword...")
                file_id, batch_id = self.client.submit_batch(str(review_batch_path))

                self.state.save_stage(
                    "review_submission",
                    {
                        "batch_id": batch_id,
                        "file_id": file_id,
                        "batch_file": str(review_batch_path),
                    },
                )

                logging.info(f"✓ Review batch submitted: {batch_id}")

            # Stage 6: Wait for review completion and download
            if not self.state.is_stage_complete("review_completion"):
                review_info = self.state.get_stage("review_submission")
                if not review_info:
                    raise RuntimeError("Review submission stage not found.")
                batch_id = review_info["batch_id"]

                if auto_poll:
                    logging.info("=" * 80)
                    logging.info("Stage 6: Waiting for review batch completion...")
                    logging.info("=" * 80)
                    logging.info(f"Batch ID: {batch_id}")
                    logging.info(f"Poll interval: {poll_interval}s")

                    try:
                        review_content = self.monitor.wait_for_completion(
                            batch_id, poll_interval, timeout
                        )
                    except KeyboardInterrupt:
                        logging.warning("Interrupted by user. Progress saved.")
                        logging.info("Run with --command=resume to continue later.")
                        return []

                    # Save results
                    review_results_path = self.output_dir / "review_results.jsonl"
                    review_results_path.write_text(review_content)

                    self.state.save_stage(
                        "review_completion", {"results_path": str(review_results_path)}
                    )

                    logging.info(f"✓ Review results downloaded: {review_results_path}")
                else:
                    logging.info("=" * 80)
                    logging.info("Stage 6: Review batch pending")
                    logging.info("=" * 80)
                    logging.info(f"Review Batch ID: {batch_id}")
                    logging.info("Run with --command=resume when batch is complete")
                    return []

        # Stage 7 (or 5 if no review): Process final results
        final_stage = "Stage 7" if self.enable_review else "Stage 5"
        if not self.state.is_stage_complete("final_processing"):
            logging.info("=" * 80)
            logging.info(f"{final_stage}: Processing final results...")
            logging.info("=" * 80)

            # Use review results if available, otherwise use coref results
            if self.enable_review and self.state.is_stage_complete("review_completion"):
                review_completion = self.state.get_stage("review_completion")
                if not review_completion:
                    raise RuntimeError("Review completion stage not found.")
                results_path = review_completion["results_path"]
                source = "review"
            else:
                coref_completion = self.state.get_stage("coref_completion")
                if not coref_completion:
                    raise RuntimeError("Coref completion stage not found.")
                results_path = coref_completion["results_path"]
                source = "coref"

            # Process and save results
            with open(results_path) as f:
                content = f.read()

            results = self.processor.process_batch_content(content, file_id="final")

            saved_files = [file_name for _, file_name in results]

            self.state.save_stage(
                "final_processing",
                {
                    "num_samples": len(saved_files),
                    "output_dir": str(
                        self.output_dir / "data_samples/annotation_samples"
                    ),
                    "source": source,
                },
            )

            logging.info(f"✓ Processed {len(saved_files)} training samples")

        # Pipeline complete
        logging.info("=" * 80)
        logging.info("🎉 Pipeline Complete!")
        logging.info("=" * 80)
        final_info = self.state.get_stage("final_processing")
        if final_info:
            logging.info(f"Generated {final_info['num_samples']} training samples")
            logging.info(f"Output directory: {final_info['output_dir']}")
            if self.enable_review:
                source = final_info.get("source", "unknown")
                logging.info(f"Quality: Reviewed and validated (source: {source})")
            else:
                logging.info("Quality: Standard (no review)")
        logging.info("=" * 80)

        return []

    def status(self) -> str:
        """
        Get pipeline status.

        Returns:
            Human-readable status string
        """
        return self.state.get_summary()

    def reset(self):
        """Reset pipeline state."""
        self.state.reset()
        logging.info("Pipeline state reset")

    def cancel(self):
        """Cancel pipeline and delete state."""
        self.state.delete()
        logging.info("Pipeline cancelled and state deleted")


def main(argv):
    """Main function for pipeline CLI."""
    del argv  # Unused

    # Set log level
    logging.set_verbosity(getattr(logging, FLAGS.log_level.upper(), logging.INFO))

    # Get API key
    api_key = FLAGS.doubleword_api_key or None
    if not api_key:
        try:
            import os

            api_key = os.getenv("DOUBLEWORD_API_KEY")
        except Exception:
            pass

    if not api_key and FLAGS.command in ["start", "resume"]:
        logging.error(
            "Doubleword API key not found. Please set --doubleword_api_key flag "
            "or DOUBLEWORD_API_KEY environment variable."
        )
        return 1

    # Handle different commands
    if FLAGS.command == "status":
        # Show status of existing pipeline (API key not needed)
        try:
            pipeline = DatasetPipeline(
                api_key="dummy",  # Not used for status check
                api_model=FLAGS.api_model,
                output_dir=FLAGS.output_dir,
                num_samples=FLAGS.num_samples,
                pipeline_id=FLAGS.pipeline_id,
            )
        except ValueError:
            # DoublewordClient validates API key, but we don't need it for status
            # Create state directly
            from pipeline_state import PipelineState

            state = PipelineState(FLAGS.output_dir, FLAGS.pipeline_id)
            print("\n" + state.get_summary() + "\n")
            return 0
        print("\n" + pipeline.status() + "\n")
        return 0

    elif FLAGS.command == "reset":
        # Reset pipeline state
        pipeline = DatasetPipeline(
            api_key="dummy",
            api_model=FLAGS.api_model,
            output_dir=FLAGS.output_dir,
            num_samples=FLAGS.num_samples,
            pipeline_id=FLAGS.pipeline_id,
        )
        pipeline.reset()
        return 0

    elif FLAGS.command == "cancel":
        # Cancel and delete pipeline state
        pipeline = DatasetPipeline(
            api_key="dummy",
            api_model=FLAGS.api_model,
            output_dir=FLAGS.output_dir,
            num_samples=FLAGS.num_samples,
            pipeline_id=FLAGS.pipeline_id,
        )
        pipeline.cancel()
        return 0

    elif FLAGS.command in ["start", "resume"]:
        # Run or resume pipeline
        if not api_key:
            logging.error("API key required for start/resume commands")
            return 1

        pipeline = DatasetPipeline(
            api_key=api_key,
            api_model=FLAGS.api_model,
            output_dir=FLAGS.output_dir,
            num_samples=FLAGS.num_samples,
            max_workers=FLAGS.max_workers,
            pipeline_id=FLAGS.pipeline_id if FLAGS.command == "resume" else None,
            enable_review=FLAGS.enable_review,
        )

        try:
            pipeline.run(
                auto_poll=FLAGS.auto_poll,
                poll_interval=FLAGS.poll_interval,
                timeout=FLAGS.timeout,
            )
            return 0
        except KeyboardInterrupt:
            logging.info("\nPipeline interrupted. Run --command=resume to continue.")
            return 130
        except Exception as e:
            logging.error(f"Pipeline failed: {e}", exc_info=True)
            return 1

    else:
        logging.error(f"Unknown command: {FLAGS.command}")
        return 1


if __name__ == "__main__":
    app.run(main)
