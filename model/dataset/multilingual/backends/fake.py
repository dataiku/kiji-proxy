"""Credential-free backends with canned output, for trying the pipeline and UI.

Select them with ``--generator_backend=fake --decision_backend=fake``. The
generator returns a fixed text per language and the decision model labels
the spans it knows; everything else is marked NONE.
"""

import re
import time
from typing import Any

from ..config import NONE_LABEL

FAKE_SAMPLES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "English": (
        "Hi Anna, Ann Lee from Nolan Logistics will call 087 123 4567 about the parcel to 14 Harbour Lane, Galway.",
        [
            ("Anna", "FIRSTNAME"),
            ("Ann", "FIRSTNAME"),
            ("Lee", "SURNAME"),
            ("Nolan Logistics", "COMPANYNAME"),
            ("087 123 4567", "PHONENUMBER"),
            ("14", "BUILDINGNUM"),
            ("Harbour Lane", "STREET"),
            ("Galway", "CITY"),
            ("parcel", NONE_LABEL),
        ],
    ),
    "German": (
        "Sehr geehrte Frau Brandt, Ihre Bestellung geht an die Lindenstraße 7, 80331 München. Rückfragen an lena.brandt@mail.de.",
        [
            ("Brandt", "SURNAME"),
            ("Lindenstraße", "STREET"),
            ("7", "BUILDINGNUM"),
            ("80331", "ZIP"),
            ("München", "CITY"),
            ("lena.brandt@mail.de", "EMAIL"),
        ],
    ),
    "French": (
        "Bonjour Camille Roux, votre colis sera livré au 12 rue des Lilas, 69003 Lyon. Téléphone : 06 12 34 56 78.",
        [
            ("Camille", "FIRSTNAME"),
            ("Roux", "SURNAME"),
            ("12", "BUILDINGNUM"),
            ("rue des Lilas", "STREET"),
            ("69003", "ZIP"),
            ("Lyon", "CITY"),
            ("06 12 34 56 78", "PHONENUMBER"),
        ],
    ),
    "Spanish": (
        "Hola Mateo, tu pedido de Ferretería Soler llega mañana a Calle Mayor 5, 28013 Madrid.",
        [
            ("Mateo", "FIRSTNAME"),
            ("Ferretería Soler", "COMPANYNAME"),
            ("Calle Mayor", "STREET"),
            ("5", "BUILDINGNUM"),
            ("28013", "ZIP"),
            ("Madrid", "CITY"),
            ("mañana", NONE_LABEL),
        ],
    ),
    "Italian": (
        "Gentile Giulia Conti, la fattura n. 42 è intestata a via Garibaldi 9, 40124 Bologna, codice fiscale CNTGLI85M41A944X.",
        [
            ("Giulia", "FIRSTNAME"),
            ("Conti", "SURNAME"),
            ("via Garibaldi", "STREET"),
            ("9", "BUILDINGNUM"),
            ("40124", "ZIP"),
            ("Bologna", "CITY"),
            ("CNTGLI85M41A944X", "TAXNUM"),
            ("42", NONE_LABEL),
        ],
    ),
    "Portuguese": (
        "Olá Rafael Nunes, o seu CPF 123.456.789-09 foi registado. Endereço: Rua das Flores 21, São Paulo.",
        [
            ("Rafael", "FIRSTNAME"),
            ("Nunes", "SURNAME"),
            ("123.456.789-09", "TAXNUM"),
            ("Rua das Flores", "STREET"),
            ("21", "BUILDINGNUM"),
            ("São Paulo", "CITY"),
        ],
    ),
    "Dutch": (
        "Beste Sanne de Vries, uw afspraak is op Keizersgracht 100, 1015 CV Amsterdam. Bel 06-12345678.",
        [
            ("Sanne", "FIRSTNAME"),
            ("de Vries", "SURNAME"),
            ("Keizersgracht", "STREET"),
            ("100", "BUILDINGNUM"),
            ("1015 CV", "ZIP"),
            ("Amsterdam", "CITY"),
            ("06-12345678", "PHONENUMBER"),
        ],
    ),
    "Japanese": (
        "田中太郎様、ご注文は〒150-0001 東京都渋谷区神宮前1-2-3へ発送します。連絡先 090-1234-5678。",
        [
            ("田中", "SURNAME"),
            ("太郎", "FIRSTNAME"),
            ("150-0001", "ZIP"),
            ("東京都", "STATE"),
            ("渋谷区", "CITY"),
            ("神宮前", "STREET"),
            ("1-2-3", "BUILDINGNUM"),
            ("090-1234-5678", "PHONENUMBER"),
            ("ご注文", NONE_LABEL),
        ],
    ),
    "Korean": (
        "김민준 고객님, 주문하신 상품은 서울특별시 강남구 테헤란로 123으로 배송됩니다. 연락처 010-9876-5432.",
        [
            ("김", "SURNAME"),
            ("민준", "FIRSTNAME"),
            ("서울특별시", "STATE"),
            ("강남구", "CITY"),
            ("테헤란로", "STREET"),
            ("123", "BUILDINGNUM"),
            ("010-9876-5432", "PHONENUMBER"),
            ("상품", NONE_LABEL),
        ],
    ),
    "Chinese (Simplified)": (
        "王伟先生，您的包裹将寄往北京市朝阳区建国路88号。联系电话 138 0013 8000。",
        [
            ("王", "SURNAME"),
            ("伟", "FIRSTNAME"),
            ("北京市", "CITY"),
            ("朝阳区", "CITY"),
            ("建国路", "STREET"),
            ("88", "BUILDINGNUM"),
            ("138 0013 8000", "PHONENUMBER"),
            ("包裹", NONE_LABEL),
        ],
    ),
}

FAKE_LABELS: dict[str, str] = {
    value: label for _, pairs in FAKE_SAMPLES.values() for value, label in pairs
}

_LANGUAGE_RE = re.compile(r"entirely in ([^,]+), that reads")
_TEXT_TYPE_RE = re.compile(r"feel natural for a ([a-z ]+)\.")


class FakeGenerator:
    """Returns a canned text for the language named in the prompt."""

    model = "fake-qwen"

    def __init__(self, delay: float = 0.0):
        self.delay = delay

    def generate(self, prompt: str, json_schema: dict[str, Any]) -> dict[str, Any]:
        if self.delay:
            time.sleep(self.delay)
        match = _LANGUAGE_RE.search(prompt)
        language = match.group(1).strip() if match else "English"
        text, pairs = FAKE_SAMPLES.get(language, FAKE_SAMPLES["English"])
        type_match = _TEXT_TYPE_RE.search(prompt)
        if type_match:
            text = f"[{type_match.group(1)}] {text}"
        return {
            "text": text,
            "candidates": [{"value": v, "author_label": label} for v, label in pairs],
        }


class FakeDecisionModel:
    """Labels spans from a fixed lookup table with a fixed confidence."""

    model = "fake-clef"
    max_questions_per_request = 64
    max_state_chars = None

    def __init__(self, delay: float = 0.0, confidence: float = 0.9):
        self.delay = delay
        self.confidence = confidence

    def decide(
        self, state: str | dict[str, Any], questions: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        if self.delay:
            time.sleep(self.delay)
        candidates = {}
        if isinstance(state, dict):
            candidates = {c["id"]: c["value"] for c in state.get("candidates", [])}
        answers: dict[str, dict[str, Any]] = {}
        for qid, question in questions.items():
            if question.get("type") == "noul":
                answers[qid] = {"probability": 0.1}
                continue
            value = candidates.get(int(qid[1:])) if qid[1:].isdigit() else None
            label = FAKE_LABELS.get(value, NONE_LABEL)
            options = question.get("criteria") or {}
            probabilities = dict.fromkeys(options, 0.0)
            probabilities[label] = self.confidence
            answers[qid] = {
                "choice": label,
                "confidence": self.confidence,
                "probabilities": probabilities,
            }
        return answers
