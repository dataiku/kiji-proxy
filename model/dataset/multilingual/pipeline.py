"""Multilingual PII dataset pipeline: an LLM writes, a decision model labels.

Usage::

    uv run python -m model.dataset.multilingual.pipeline --num_samples=100 \\
        --generator_backend=workers-ai --decision_backend=clef

    # Local Qwen through vLLM and Laya through laya-serve
    uv run python -m model.dataset.multilingual.pipeline --num_samples=100 \\
        --generator_backend=openai-compatible --decision_backend=laya

    # Doubleword for generation, no credentials needed for a dry run with fake
    uv run python -m model.dataset.multilingual.pipeline --generator_backend=fake \\
        --decision_backend=fake

Output (under ``--output_dir``)::

    samples/<index>_<lang>_<text_type>.json   one labeled sample per file (resumable)
    labelstudio_tasks.json                    import this file into Label Studio
    labeling_config.xml                       matching labeling interface (symbol granularity)
    summary.json                              per-language / per-type counts and agreement
    progress.json                             live progress of the last run
"""

import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from absl import app, flags, logging
from absl.flags import DuplicateFlagError
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to sys.path for imports when running as a script
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.dataset.label_utils import LabelUtils  # noqa: E402
from model.dataset.multilingual.backends import (  # noqa: E402
    DECISION_BACKENDS,
    GENERATOR_BACKENDS,
    TextGenerator,
    build_decision_model,
    build_generator,
)
from model.dataset.multilingual.config import (  # noqa: E402
    DEFAULT_DECISION_BACKEND,
    DEFAULT_GENERATOR_BACKEND,
    DEFAULT_OUTPUT_DIR,
    GENERATION_LANGUAGES,
    TEXT_TYPES,
    PipelineConfig,
)
from model.dataset.multilingual.entity_labeler import EntityLabeler  # noqa: E402
from model.dataset.multilingual.labelstudio_tasks import (  # noqa: E402
    build_labelstudio_task,
    write_labeling_config,
    write_labelstudio_import,
)
from model.dataset.multilingual.progress import ProgressTracker  # noqa: E402
from model.dataset.multilingual.prompts import (  # noqa: E402
    build_generation_prompt,
    get_generation_schema,
)

load_dotenv(project_root / ".env")

FLAGS = flags.FLAGS


def _define_flag(define_func, *args, **kwargs):
    """Define a flag, ignoring duplicates when the module is imported twice."""
    try:
        define_func(*args, **kwargs)
    except DuplicateFlagError:
        pass


_define_flag(flags.DEFINE_integer, "num_samples", 10, "Number of samples to generate")
_define_flag(
    flags.DEFINE_string,
    "output_dir",
    DEFAULT_OUTPUT_DIR,
    "Output directory for samples and the Label Studio import file",
)
_define_flag(
    flags.DEFINE_enum,
    "generator_backend",
    DEFAULT_GENERATOR_BACKEND,
    list(GENERATOR_BACKENDS),
    "Backend that writes the texts",
)
_define_flag(
    flags.DEFINE_string,
    "generator_model",
    None,
    "Generation model name (default: the backend's default)",
)
_define_flag(
    flags.DEFINE_string,
    "generator_api_url",
    None,
    "Generation endpoint override (e.g. a vLLM chat completions URL)",
)
_define_flag(
    flags.DEFINE_enum,
    "decision_backend",
    DEFAULT_DECISION_BACKEND,
    list(DECISION_BACKENDS),
    "Backend that labels the candidate spans",
)
_define_flag(
    flags.DEFINE_string,
    "decision_model",
    None,
    "Decision model name (default: the backend's default)",
)
_define_flag(
    flags.DEFINE_string,
    "decision_api_url",
    None,
    "Decision endpoint override (e.g. a laya-serve or clef_server.py URL)",
)
_define_flag(
    flags.DEFINE_list,
    "languages",
    list(GENERATION_LANGUAGES.keys()),
    "Comma-separated languages to generate (default: all ten)",
)
_define_flag(
    flags.DEFINE_list,
    "text_types",
    list(TEXT_TYPES.keys()),
    "Comma-separated text types to rotate through (default: all)",
)
_define_flag(
    flags.DEFINE_integer,
    "min_labels_per_sample",
    2,
    "Fewest PII label types requested per text (the count is random per sample)",
)
_define_flag(
    flags.DEFINE_integer,
    "max_labels_per_sample",
    6,
    "Most PII label types requested per text (the count is random per sample)",
)
_define_flag(flags.DEFINE_integer, "seed", 0, "Seed for language/type/label planning")
_define_flag(
    flags.DEFINE_float,
    "min_score",
    0.5,
    "Minimum decision probability for an entity to appear in the predictions",
)
_define_flag(
    flags.DEFINE_integer,
    "max_workers",
    None,
    "Parallel workers (default: min(8, num_samples))",
)
_define_flag(
    flags.DEFINE_boolean, "overwrite", False, "Regenerate samples that already exist"
)
_define_flag(
    flags.DEFINE_string, "log_level", "INFO", "Logging level (DEBUG, INFO, WARNING)"
)


@dataclass(frozen=True)
class SamplePlan:
    """What one sample should look like."""

    index: int
    language: str
    country: str
    text_type: str
    labels: tuple[str, ...]

    @property
    def language_code(self) -> str:
        return GENERATION_LANGUAGES[self.language]["code"]

    @property
    def script(self) -> str:
        return GENERATION_LANGUAGES[self.language]["script"]

    @property
    def file_name(self) -> str:
        return f"{self.index:05d}_{self.language_code}_{self.text_type}.json"


class GenerationPipeline:
    """Generates, labels and exports multilingual PII samples."""

    def __init__(
        self,
        config: PipelineConfig,
        generator: TextGenerator,
        labeler: EntityLabeler,
    ):
        self.config = config
        self.generator = generator
        self.labeler = labeler
        self.output_dir = Path(config.output_dir)
        self.samples_dir = self.output_dir / "samples"

    # ---------------------------------------------------------------- planning
    def plan(self, index: int) -> SamplePlan:
        """Deterministically assign language, country, text type and labels.

        Languages rotate fastest so every language gets an equal share; text
        types rotate once per full language cycle. The number of PII label
        types is drawn per sample from the configured range, so texts vary
        between sparse and dense; everything derives from the seed and index.
        """
        languages = self.config.languages
        text_types = self.config.text_types
        language = languages[index % len(languages)]
        text_type = text_types[(index // len(languages)) % len(text_types)]
        rng = random.Random(self.config.seed * 1_000_003 + index)
        country = rng.choice(GENERATION_LANGUAGES[language]["countries"])
        label_count = rng.randint(
            self.config.min_labels_per_sample,
            min(self.config.max_labels_per_sample, len(LabelUtils.STANDARD_PII_LABELS)),
        )
        labels = LabelUtils.get_pii_labels(
            return_count=label_count, seed=self.config.seed * 1_000_003 + index
        )
        return SamplePlan(
            index=index,
            language=language,
            country=country,
            text_type=text_type,
            labels=tuple(sorted(labels)),
        )

    # -------------------------------------------------------------- one sample
    def generate_sample(self, plan: SamplePlan) -> dict[str, Any]:
        """Generate text, then label its candidates with the decision model."""
        labels = {code: LabelUtils.LABEL_DESCRIPTIONS[code] for code in plan.labels}
        prompt = build_generation_prompt(
            plan.language, plan.country, plan.text_type, labels, plan.index
        )
        generated = self.generator.generate(prompt, get_generation_schema())
        text = str(generated.get("text", "")).strip()
        if not text:
            raise ValueError("Generator returned an empty text")
        candidates = generated.get("candidates") or []

        result = self.labeler.label(text, plan.language, plan.country, candidates)

        return {
            "text": text,
            "language": plan.language,
            "language_code": plan.language_code,
            "script": plan.script,
            "country": plan.country,
            "text_type": plan.text_type,
            "requested_labels": list(plan.labels),
            "generator_backend": self.config.generator_backend,
            "generator_model": self.generator.model,
            "labeler_backend": self.config.decision_backend,
            "labeler_model": self.labeler.model,
            "sample_index": plan.index,
            "candidates": candidates,
            "privacy_mask": [span.to_dict() for span in result.pii_spans],
            "rejected_candidates": [span.to_dict() for span in result.rejected_spans],
            "dropped_candidates": result.dropped_candidates,
            "missing_pii_probability": result.missing_pii_probability,
        }

    def process(self, index: int) -> tuple[SamplePlan, Path, bool]:
        """Generate one sample unless its file exists. Returns (plan, path, created)."""
        plan = self.plan(index)
        path = self.samples_dir / plan.file_name
        if path.exists() and not self.config.overwrite:
            return plan, path, False
        sample = self.generate_sample(plan)
        sample["file_name"] = plan.file_name
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        return plan, path, True

    # ------------------------------------------------------------------- batch
    def run(
        self, tracker: ProgressTracker | None = None, show_progress_bar: bool = True
    ) -> list[dict[str, Any]]:
        """Generate all samples, then write the Label Studio import file.

        ``tracker`` receives one outcome per sample and can request a cancel;
        samples not started yet are then skipped and the export still runs on
        what exists.
        """
        tracker = tracker or ProgressTracker()
        indices = list(range(self.config.num_samples))
        max_workers = self.config.max_workers or min(8, max(1, len(indices)))

        logging.info(
            f"Generating {len(indices)} samples across {len(self.config.languages)} "
            f"languages and {len(self.config.text_types)} text types "
            f"with {max_workers} workers ({self.config.generator_backend} + "
            f"{self.config.decision_backend})"
        )
        tracker.start(len(indices), str(self.output_dir))

        def guarded(index: int) -> tuple[SamplePlan, Path | None, str]:
            plan = self.plan(index)
            if tracker.cancelled:
                return plan, None, "cancelled"
            plan, path, created = self.process(index)
            return plan, path, "created" if created else "skipped"

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(guarded, i): i for i in indices}
                with tqdm(
                    total=len(indices),
                    desc="Generating samples",
                    disable=not show_progress_bar,
                ) as bar:
                    for future in as_completed(futures):
                        index = futures[future]
                        plan = self.plan(index)
                        try:
                            plan, _, outcome = future.result()
                            tracker.record(outcome, plan.language, plan.text_type)
                        except Exception as exc:  # noqa: BLE001 - keep the batch running
                            logging.error(f"Sample {index} failed: {exc}")
                            tracker.record(
                                "failed", plan.language, plan.text_type, str(exc), index
                            )
                        bar.update(1)
            snapshot = tracker.snapshot()
            if snapshot["failed"]:
                logging.warning(
                    f"{snapshot['failed']} of {len(indices)} samples failed; "
                    "re-run the same command to retry only the missing ones"
                )
            logging.info(f"Generated {snapshot['created']} new samples")
            tasks = self.export()
            tracker.finish(
                self.load_summary(), str(self.output_dir / "labelstudio_tasks.json")
            )
            tracker.write(self.output_dir / "progress.json")
            return tasks
        except Exception as exc:
            tracker.fail(str(exc))
            tracker.write(self.output_dir / "progress.json")
            raise

    def load_samples(self) -> list[dict[str, Any]]:
        """Load every sample file in the samples directory."""
        samples = []
        for path in sorted(self.samples_dir.glob("*.json")):
            with open(path, encoding="utf-8") as f:
                samples.append(json.load(f))
        return samples

    def load_summary(self) -> dict[str, Any]:
        path = self.output_dir / "summary.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def export(self) -> list[dict[str, Any]]:
        """Build the Label Studio import, labeling config and summary from disk."""
        samples = self.load_samples()
        tasks = [
            build_labelstudio_task(sample, min_score=self.config.min_score)
            for sample in samples
        ]
        tasks_path = write_labelstudio_import(
            tasks, self.output_dir / "labelstudio_tasks.json"
        )
        write_labeling_config(self.output_dir / "labeling_config.xml")
        summary = self.summarize(samples, tasks)
        with open(self.output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logging.info(f"Wrote {len(tasks)} Label Studio tasks to {tasks_path}")
        logging.info(f"Summary: {json.dumps(summary, ensure_ascii=False)}")
        return tasks

    @staticmethod
    def summarize(
        samples: list[dict[str, Any]], tasks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Counts per language, text type and label, plus author/labeler agreement."""
        languages = Counter(s["language"] for s in samples)
        text_types = Counter(s["text_type"] for s in samples)
        labels = Counter(
            region["value"]["labels"][0]
            for task in tasks
            for region in task["predictions"][0]["result"]
        )
        decided = [
            item
            for s in samples
            for item in s.get("privacy_mask", []) + s.get("rejected_candidates", [])
            if item.get("author_label")
        ]
        agreement = (
            round(
                sum(1 for i in decided if i["author_label"] == i["label"])
                / len(decided),
                4,
            )
            if decided
            else None
        )
        return {
            "samples": len(samples),
            "tasks": len(tasks),
            "entities": sum(labels.values()),
            "languages": dict(sorted(languages.items())),
            "text_types": dict(sorted(text_types.items())),
            "labels": dict(sorted(labels.items())),
            "author_labeler_agreement": agreement,
        }


def build_pipeline(config: PipelineConfig) -> GenerationPipeline:
    """Wire the configured backends into a pipeline."""
    generator = build_generator(
        config.generator_backend, config.generator_model, config.generator_api_url
    )
    decision_model = build_decision_model(
        config.decision_backend, config.decision_model, config.decision_api_url
    )
    return GenerationPipeline(config, generator, EntityLabeler(decision_model))


def main(argv):
    del argv  # Unused
    logging.set_verbosity(getattr(logging, FLAGS.log_level.upper(), logging.INFO))

    config = PipelineConfig(
        num_samples=FLAGS.num_samples,
        output_dir=FLAGS.output_dir,
        generator_backend=FLAGS.generator_backend,
        generator_model=FLAGS.generator_model,
        generator_api_url=FLAGS.generator_api_url,
        decision_backend=FLAGS.decision_backend,
        decision_model=FLAGS.decision_model,
        decision_api_url=FLAGS.decision_api_url,
        languages=[lang.strip() for lang in FLAGS.languages],
        text_types=[t.strip() for t in FLAGS.text_types],
        min_labels_per_sample=FLAGS.min_labels_per_sample,
        max_labels_per_sample=FLAGS.max_labels_per_sample,
        seed=FLAGS.seed,
        min_score=FLAGS.min_score,
        max_workers=FLAGS.max_workers,
        overwrite=FLAGS.overwrite,
    )
    tasks = build_pipeline(config).run()
    print(
        f"Label Studio import ready: "
        f"{Path(config.output_dir) / 'labelstudio_tasks.json'} ({len(tasks)} tasks)"
    )


if __name__ == "__main__":
    app.run(main)
