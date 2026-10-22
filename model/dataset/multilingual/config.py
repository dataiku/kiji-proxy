"""Configuration for the multilingual generation pipeline."""

from dataclasses import asdict, dataclass, field

# Backend names (see backends/__init__.py for the registry).
DEFAULT_GENERATOR_BACKEND = "workers-ai"
DEFAULT_DECISION_BACKEND = "clef"
DEFAULT_OUTPUT_DIR = "model/dataset/data_samples/multilingual"

# Option returned by the decision model when a span is not personal data.
NONE_LABEL = "NONE"

# Clef and the Jev wire API accept at most 64 questions per request.
MAX_QUESTIONS_PER_REQUEST = 64

# Ten target languages. Each entry carries the ISO 639-1 code, the script family
# (used to pick matching rules and the Label Studio granularity) and generation
# hints that make the PII realistic for the locale.
GENERATION_LANGUAGES: dict[str, dict] = {
    "English": {
        "code": "en",
        "script": "latin",
        "countries": ["United States", "United Kingdom", "Canada", "Australia"],
        "hints": [],
    },
    "German": {
        "code": "de",
        "script": "latin",
        "countries": ["Germany", "Austria", "Switzerland"],
        "hints": ["Use German postal codes (5 digits) and +49/+43/+41 phone formats."],
    },
    "French": {
        "code": "fr",
        "script": "latin",
        "countries": ["France", "Belgium", "Canada", "Switzerland"],
        "hints": [
            "Use French address conventions (number, street type, postal code, city)."
        ],
    },
    "Spanish": {
        "code": "es",
        "script": "latin",
        "countries": ["Spain", "Mexico", "Argentina", "Colombia", "Chile"],
        "hints": ["Use double surnames where natural for the country."],
    },
    "Italian": {
        "code": "it",
        "script": "latin",
        "countries": ["Italy", "Switzerland"],
        "hints": ["Use Italian CAP postal codes and codice fiscale for tax numbers."],
    },
    "Portuguese": {
        "code": "pt",
        "script": "latin",
        "countries": ["Brazil", "Portugal"],
        "hints": ["Use CPF for Brazilian tax numbers and NIF for Portuguese ones."],
    },
    "Dutch": {
        "code": "nl",
        "script": "latin",
        "countries": ["Netherlands", "Belgium"],
        "hints": ["Use Dutch postal codes (1234 AB) and BSN for national IDs."],
    },
    "Japanese": {
        "code": "ja",
        "script": "cjk",
        "countries": ["Japan"],
        "hints": [
            "Write names in kanji (family name first) and addresses in Japanese order: 〒postal code, prefecture, city, ward, chome-ban-go.",
            "Phone numbers use the 0X-XXXX-XXXX or 090-XXXX-XXXX format; My Number has 12 digits.",
            "Use half-width ASCII digits for numbers, emails and URLs.",
        ],
    },
    "Korean": {
        "code": "ko",
        "script": "cjk",
        "countries": ["South Korea"],
        "hints": [
            "Write names in Hangul (family name first) and road-name addresses (도로명 주소) with a 5-digit postal code.",
            "Phone numbers use the 010-XXXX-XXXX format; resident registration numbers use the XXXXXX-XXXXXXX format.",
            "Use half-width ASCII digits for numbers, emails and URLs.",
        ],
    },
    "Chinese (Simplified)": {
        "code": "zh",
        "script": "cjk",
        "countries": ["China", "Singapore"],
        "hints": [
            "Write in Simplified Chinese characters only. Names in Chinese characters (family name first); addresses in Chinese order: province, city, district, road, building number.",
            "Phone numbers are 11-digit mobile numbers (1XX XXXX XXXX); national IDs are 18 characters.",
            "Use half-width ASCII digits for numbers, emails and URLs.",
        ],
    },
}

# Text types the generator rotates through. The hint shapes the layout and tone.
TEXT_TYPES: dict[str, str] = {
    "business_email": "a formal business email with greeting, body and signature block",
    "chat_message": "a casual chat or instant message between two people",
    "support_ticket": "a customer support ticket with the customer's details and the issue",
    "official_form": "an official form or application with labeled fields and values",
    "news_article": "a short news article or blog post quoting people",
    "social_media_post": "a social media post or comment thread",
    "medical_note": "a clinical note or patient record entry",
    "legal_excerpt": "an excerpt of a contract or legal notice naming the parties",
    "invoice": "an invoice, receipt or order confirmation with billing details",
    "meeting_minutes": "meeting minutes listing attendees and action items",
    "job_application": "a job application or CV excerpt",
    "hr_record": "an HR onboarding note or employee record",
}


@dataclass
class PipelineConfig:
    """Runtime configuration for the pipeline."""

    num_samples: int = 10
    output_dir: str = DEFAULT_OUTPUT_DIR
    generator_backend: str = DEFAULT_GENERATOR_BACKEND
    generator_model: str | None = None  # backend default when None
    generator_api_url: str | None = None
    decision_backend: str = DEFAULT_DECISION_BACKEND
    decision_model: str | None = None  # backend default when None
    decision_api_url: str | None = None
    languages: list[str] = field(
        default_factory=lambda: list(GENERATION_LANGUAGES.keys())
    )
    text_types: list[str] = field(default_factory=lambda: list(TEXT_TYPES.keys()))
    min_labels_per_sample: int = 2  # label count is drawn per sample from this range
    max_labels_per_sample: int = 6
    seed: int = 0
    min_score: float = 0.5
    max_workers: int | None = None
    overwrite: bool = False

    def __post_init__(self) -> None:
        unknown_languages = [
            lang for lang in self.languages if lang not in GENERATION_LANGUAGES
        ]
        if unknown_languages:
            raise ValueError(
                f"Unknown languages: {unknown_languages}. "
                f"Choose from: {list(GENERATION_LANGUAGES.keys())}"
            )
        unknown_types = [t for t in self.text_types if t not in TEXT_TYPES]
        if unknown_types:
            raise ValueError(
                f"Unknown text types: {unknown_types}. "
                f"Choose from: {list(TEXT_TYPES.keys())}"
            )
        if not self.languages:
            raise ValueError("At least one language is required")
        if not self.text_types:
            raise ValueError("At least one text type is required")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        if self.num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        if self.min_labels_per_sample < 1:
            raise ValueError("min_labels_per_sample must be at least 1")
        if self.max_labels_per_sample < self.min_labels_per_sample:
            raise ValueError("max_labels_per_sample must be >= min_labels_per_sample")

    def to_dict(self) -> dict:
        """Plain dict (for job records and the UI)."""
        return asdict(self)
