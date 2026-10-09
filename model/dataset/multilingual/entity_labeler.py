"""Entity labeling with a decision model (Clef, Laya, ...).

A decision model does not generate text; it answers typed questions about a
state with a probability for every allowed option. The labeler turns each
candidate span into a ``choice`` question whose options are the PII labels
plus ``NONE``, and adds a ``noul`` question that estimates whether the
candidate list missed personal data.

Models with a small context (``max_state_chars`` set, e.g. Laya's 512-token
encoder) receive one span per request with a text window around the span.
"""

from dataclasses import dataclass, field
from typing import Any

from absl import logging

from ..label_utils import LabelUtils
from .backends.protocols import DecisionModel
from .config import MAX_QUESTIONS_PER_REQUEST, NONE_LABEL

MISSING_PII_QUESTION_ID = "missing_pii"

NONE_DESCRIPTION = (
    "Not personal data: a common word, a product, a relative date, or a generic "
    "organisation or number that does not identify a person"
)


@dataclass
class LabeledSpan:
    """A candidate span with the decision model's verdict."""

    value: str
    label: str
    score: float
    author_label: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def is_pii(self) -> bool:
        return self.label != NONE_LABEL

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "label": self.label,
            "score": round(self.score, 4),
            "author_label": self.author_label,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


@dataclass
class LabelingResult:
    """The decision model's verdicts for one text."""

    spans: list[LabeledSpan]
    missing_pii_probability: float | None = None
    dropped_candidates: list[str] = field(default_factory=list)

    @property
    def pii_spans(self) -> list[LabeledSpan]:
        return [span for span in self.spans if span.is_pii]

    @property
    def rejected_spans(self) -> list[LabeledSpan]:
        return [span for span in self.spans if not span.is_pii]


def validate_candidates(
    text: str, candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep candidates whose value appears verbatim in ``text``; dedupe by value."""
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        value = str(candidate.get("value", "")).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        if value not in text:
            dropped.append(value)
            continue
        kept.append({"value": value, "author_label": candidate.get("author_label")})
    return kept, dropped


def text_window(text: str, value: str, max_chars: int) -> str:
    """A window of at most ``max_chars`` characters around the first ``value``."""
    if len(text) <= max_chars:
        return text
    pos = text.find(value)
    if pos == -1:
        return text[:max_chars]
    half = max(0, (max_chars - len(value)) // 2)
    start = max(0, pos - half)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    return text[start:end]


class EntityLabeler:
    """Decides the PII label of candidate spans with a decision model."""

    def __init__(
        self,
        decision_model: DecisionModel,
        labels: list[str] | None = None,
        max_questions_per_request: int | None = None,
        ask_missing_pii: bool = True,
    ):
        self.decision_model = decision_model
        self.labels = list(labels or LabelUtils.STANDARD_PII_LABELS)
        limit = max_questions_per_request or min(
            MAX_QUESTIONS_PER_REQUEST,
            getattr(
                decision_model, "max_questions_per_request", MAX_QUESTIONS_PER_REQUEST
            ),
        )
        if limit < 2:
            raise ValueError("max_questions_per_request must be at least 2")
        self.max_questions_per_request = limit
        self.ask_missing_pii = ask_missing_pii
        self._options = self._build_options()
        self._option_lookup = {opt.lower(): opt for opt in self._options}

    @property
    def model(self) -> str:
        return self.decision_model.model

    def _build_options(self) -> dict[str, str]:
        options = {
            label: LabelUtils.get_label_description(label) for label in self.labels
        }
        options[NONE_LABEL] = NONE_DESCRIPTION
        return options

    # ------------------------------------------------------------------ build
    def build_state(
        self,
        text: str,
        language: str,
        country: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """State shared by all questions of one request."""
        return {
            "task": (
                "Personal data (PII) annotation. The text was written in "
                f"{language} for {country}. Decide for each numbered candidate "
                "span which kind of personal data it is, as it is used in the text."
            ),
            "text": text,
            "candidates": [
                {"id": i + 1, "value": c["value"]} for i, c in enumerate(candidates)
            ],
        }

    def build_candidate_question(self, index: int, value: str) -> dict[str, Any]:
        """A ``choice`` question for candidate number ``index`` (1-based)."""
        return {
            "type": "choice",
            "instructions": (
                f'Candidate {index} is the span "{value}". Which kind of personal '
                f"data is it in this text? Answer {NONE_LABEL} if it is not personal data."
            ),
            "criteria": dict(self._options),
        }

    @staticmethod
    def build_missing_pii_question() -> dict[str, Any]:
        """A ``noul`` question estimating whether the candidate list is incomplete."""
        return {
            "type": "noul",
            "instructions": (
                "Does the text contain personal data (names, contact details, "
                "addresses, identifiers) that is not covered by any candidate span?"
            ),
        }

    def _chunks(
        self, candidates: list[dict[str, Any]]
    ) -> list[list[tuple[int, dict[str, Any]]]]:
        numbered = list(enumerate(candidates, start=1))
        first_size = self.max_questions_per_request - (1 if self.ask_missing_pii else 0)
        chunks = [numbered[:first_size]]
        rest = numbered[first_size:]
        size = self.max_questions_per_request
        for start in range(0, len(rest), size):
            chunks.append(rest[start : start + size])
        return [chunk for chunk in chunks if chunk] or [[]]

    # ------------------------------------------------------------------ parse
    def _normalise_option(self, option: Any) -> str | None:
        if option is None:
            return None
        return self._option_lookup.get(str(option).strip().lower())

    def parse_choice(
        self, answer: dict[str, Any]
    ) -> tuple[str, float, dict[str, float]]:
        """Return ``(label, probability, probabilities)`` from a choice answer."""
        raw_probs = answer.get("probabilities") or answer.get("options") or {}
        probabilities: dict[str, float] = {}
        if isinstance(raw_probs, dict):
            for option, prob in raw_probs.items():
                normalised = self._normalise_option(option)
                if normalised is not None and isinstance(prob, int | float):
                    probabilities[normalised] = float(prob)
        elif isinstance(raw_probs, list):
            for item in raw_probs:
                if not isinstance(item, dict):
                    continue
                normalised = self._normalise_option(
                    item.get("option") or item.get("id") or item.get("value")
                )
                prob = item.get("probability", item.get("p"))
                if normalised is not None and isinstance(prob, int | float):
                    probabilities[normalised] = float(prob)

        label = self._normalise_option(
            answer.get("choice") or answer.get("answer") or answer.get("value")
        )
        if label is None and probabilities:
            label = max(probabilities, key=probabilities.get)
        if label is None:
            raise ValueError(f"Could not read a choice from the answer: {answer}")

        if label in probabilities:
            score = probabilities[label]
        elif isinstance(answer.get("confidence"), int | float):
            score = float(answer["confidence"])
        else:
            score = 1.0
        return label, score, probabilities

    @staticmethod
    def parse_noul(answer: dict[str, Any]) -> float | None:
        """Return the probability that a ``noul`` answer is true."""
        for key in ("probability", "noul", "true", "yes", "p", "value"):
            value = answer.get(key)
            if isinstance(value, int | float):
                return float(value)
        probabilities = answer.get("probabilities")
        if isinstance(probabilities, dict):
            for key in ("true", "yes"):
                value = probabilities.get(key)
                if isinstance(value, int | float):
                    return float(value)
        return None

    def _span_from_answer(
        self, candidate: dict[str, Any], answer: dict[str, Any]
    ) -> LabeledSpan:
        label, score, probabilities = self.parse_choice(answer)
        return LabeledSpan(
            value=candidate["value"],
            label=label,
            score=score,
            author_label=candidate.get("author_label"),
            probabilities=probabilities,
        )

    # -------------------------------------------------------------------- run
    def label(
        self,
        text: str,
        language: str,
        country: str,
        candidates: list[dict[str, Any]],
    ) -> LabelingResult:
        """Ask the decision model to label every candidate span of ``text``."""
        valid, dropped = validate_candidates(text, candidates)
        if dropped:
            logging.info(
                f"Dropped {len(dropped)} candidates not found verbatim in text"
            )
        if not valid:
            return LabelingResult(spans=[], dropped_candidates=dropped)

        limit = getattr(self.decision_model, "max_state_chars", None)
        if limit and len(text) > limit:
            result = self._label_windowed(text, language, country, valid, limit)
        else:
            result = self._label_batched(text, language, country, valid)
        result.dropped_candidates = dropped
        return result

    def _label_batched(
        self, text: str, language: str, country: str, valid: list[dict[str, Any]]
    ) -> LabelingResult:
        state = self.build_state(text, language, country, valid)
        spans: list[LabeledSpan] = []
        missing_pii: float | None = None

        for chunk_index, chunk in enumerate(self._chunks(valid)):
            questions: dict[str, dict[str, Any]] = {}
            for number, candidate in chunk:
                questions[f"c{number}"] = self.build_candidate_question(
                    number, candidate["value"]
                )
            ask_noul = chunk_index == 0 and self.ask_missing_pii
            if ask_noul:
                questions[MISSING_PII_QUESTION_ID] = self.build_missing_pii_question()

            answers = self.decision_model.decide(state, questions)

            for number, candidate in chunk:
                answer = answers.get(f"c{number}")
                if answer is None:
                    logging.warning(f"No answer for candidate {number}")
                    continue
                spans.append(self._span_from_answer(candidate, answer))
            if ask_noul and answers.get(MISSING_PII_QUESTION_ID) is not None:
                missing_pii = self.parse_noul(answers[MISSING_PII_QUESTION_ID])

        return LabelingResult(spans=spans, missing_pii_probability=missing_pii)

    def _label_windowed(
        self,
        text: str,
        language: str,
        country: str,
        valid: list[dict[str, Any]],
        max_chars: int,
    ) -> LabelingResult:
        """One request per span with a text window, for small-context models."""
        spans: list[LabeledSpan] = []
        for candidate in valid:
            window = text_window(text, candidate["value"], max_chars)
            state = self.build_state(window, language, country, [candidate])
            questions = {"c1": self.build_candidate_question(1, candidate["value"])}
            answers = self.decision_model.decide(state, questions)
            answer = answers.get("c1")
            if answer is None:
                logging.warning(f"No answer for span {candidate['value']!r}")
                continue
            spans.append(self._span_from_answer(candidate, answer))
        # The missing-PII question needs the whole text, which does not fit.
        return LabelingResult(spans=spans, missing_pii_probability=None)
