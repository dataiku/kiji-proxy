"""Interfaces every generator and decision backend implements."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TextGenerator(Protocol):
    """Writes a JSON object (text plus candidate spans) from a prompt."""

    model: str

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON object following ``json_schema``."""
        ...


@runtime_checkable
class DecisionModel(Protocol):
    """Answers typed questions (choice / noul / score) about a state.

    ``max_questions_per_request`` bounds how many questions one call may carry.
    ``max_state_chars`` is ``None`` for models with a large context, or the
    number of characters of text the model can reliably read (small encoders
    such as Laya); the labeler switches to per-span windows above that size.
    """

    model: str
    max_questions_per_request: int
    max_state_chars: int | None

    def decide(
        self, state: str | dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Return one answer dict per question id."""
        ...


def extract_answers(result: Any, question_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Locate the per-question answer map inside a decision-model response.

    Clef (Workers AI) and the Jev wire API used by Laya return
    ``{"<question id>": {...}}``, sometimes nested under a container key such
    as ``answers``. Every answer is normalised to a dict.
    """
    if not isinstance(result, dict):
        raise ValueError(f"Unexpected decision response type: {type(result)}")

    candidates = [result]
    for key in ("answers", "decisions", "questions", "results", "output", "result"):
        nested = result.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)

    for mapping in candidates:
        if question_ids & set(mapping):
            return {
                qid: (ans if isinstance(ans, dict) else {"value": ans})
                for qid, ans in mapping.items()
                if qid in question_ids
            }
    preview = str(result)[:300]
    raise ValueError(
        f"Decision response did not contain answers for "
        f"{sorted(question_ids)[:5]}: {preview}"
    )
