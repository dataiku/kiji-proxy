"""Prompt builders for LLM interactions."""

import json
import random
from typing import Any


class PromptBuilder:
    """Builder for creating prompts for LLM interactions."""

    SAMPLE_DATA = [
        {
            "text": "Subject: Meeting Confirmation\n\nHi Sarah,\n\nThank you for your interest in our services. I'd like to confirm our meeting scheduled for next week.\n\nPlease find my contact details below:\nEmail: michael.chen@techcorp.com\nPhone: 555-987-6543\nAddress: 1234 Innovation Drive, San Francisco, CA 94105\n\nLooking forward to speaking with you.\n\nBest regards,\nMichael Chen\nSenior Manager at TechCorp\nWebsite: https://www.techcorp.com",
            "privacy_mask": [
                {"value": "Sarah", "label": "FIRSTNAME"},
                {"value": "michael.chen@techcorp.com", "label": "EMAIL"},
                {"value": "Michael", "label": "FIRSTNAME"},
                {"value": "Chen", "label": "SURNAME"},
                {"value": "555-987-6543", "label": "PHONENUMBER"},
                {"value": "1234", "label": "BUILDINGNUM"},
                {"value": "Innovation Drive", "label": "STREET"},
                {"value": "San Francisco", "label": "CITY"},
                {"value": "CA", "label": "STATE"},
                {"value": "94105", "label": "ZIP"},
                {"value": "TechCorp", "label": "COMPANYNAME"},
                {"value": "https://www.techcorp.com", "label": "URL"},
            ],
            "coreferences": [
                {
                    "cluster_id": 0,
                    "mentions": ["Michael Chen", "I", "my"],
                    "entity_type": "person",
                }
            ],
        },
        {
            "text": "The patient, Maria Santos, was admitted on 2024-03-15. Her date of birth is 1985-07-22. The medical record shows her SSN as 456-78-9012. Dr. Thompson noted that Ms. Santos should follow up at her current address: 789 Maple Boulevard, Austin, TX 78701.",
            "privacy_mask": [
                {"value": "Maria", "label": "FIRSTNAME"},
                {"value": "Santos", "label": "SURNAME"},
                {"value": "2024-03-15", "label": "DATEOFBIRTH"},
                {"value": "1985-07-22", "label": "DATEOFBIRTH"},
                {"value": "456-78-9012", "label": "SSN"},
                {"value": "789", "label": "BUILDINGNUM"},
                {"value": "Maple Boulevard", "label": "STREET"},
                {"value": "Austin", "label": "CITY"},
                {"value": "TX", "label": "STATE"},
                {"value": "78701", "label": "ZIP"},
            ],
            "coreferences": [
                {
                    "cluster_id": 0,
                    "mentions": ["Maria Santos", "Her", "Ms. Santos", "her"],
                    "entity_type": "person",
                }
            ],
        },
        {
            "text": "Customer ID: CUST-789456. Account holder: Kenji Yamamoto. Contact: kenji.y@email.jp. The account was opened on 12/05/2020. Tax identification number: JP-987654321. IBAN: GB82WEST12345698765432.",
            "privacy_mask": [
                {"value": "Kenji", "label": "FIRSTNAME"},
                {"value": "Yamamoto", "label": "SURNAME"},
                {"value": "kenji.y@email.jp", "label": "EMAIL"},
                {"value": "12/05/2020", "label": "DATEOFBIRTH"},
                {"value": "JP-987654321", "label": "TAXNUM"},
                {"value": "GB82WEST12345698765432", "label": "IBAN"},
            ],
            "coreferences": [],
        },
        {
            "text": "Just moved to a new place! My address is now 42 Rue de la Paix, Lyon, 69001, France. Phone changed too: +33-6-12-34-56-78. Email stays the same: sophie.martin@france.fr",
            "privacy_mask": [
                {"value": "42", "label": "BUILDINGNUM"},
                {"value": "Rue de la Paix", "label": "STREET"},
                {"value": "Lyon", "label": "CITY"},
                {"value": "69001", "label": "ZIP"},
                {"value": "France", "label": "COUNTRY"},
                {"value": "+33-6-12-34-56-78", "label": "PHONENUMBER"},
                {"value": "sophie.martin@france.fr", "label": "EMAIL"},
            ],
            "coreferences": [
                {
                    "cluster_id": 0,
                    "mentions": ["My", "I", "My"],
                    "entity_type": "person",
                }
            ],
        },
    ]

    @staticmethod
    def build_generation_prompt(
        labels: dict[str, str],
        languages_countries: list[tuple[str, str]],
        sample_index: int = 0,
    ) -> str:
        """
        Build a prompt for generating PII samples.

        Args:
            labels: Dictionary of PII labels to include
            languages: List of languages to use
            sample_index: Index of sample being generated (for variation)
        """
        # Use sample_index to select example (for more variation)
        example_index = sample_index % len(PromptBuilder.SAMPLE_DATA)
        random_sample = PromptBuilder.SAMPLE_DATA[example_index]
        example_string = f"""
        ```json\n{json.dumps(random_sample, indent=2)}\n```
        """

        # Pick one language randomly from the list
        selected_language_country = (
            random.choice(languages_countries)
            if languages_countries
            else ("English", "United States")
        )

        # Add variety in writing style and context
        writing_styles = [
            "formal business email",
            "casual personal message",
            "official document or form",
            "news article or blog post",
            "social media post",
            "customer service interaction",
            "medical record entry",
            "legal document excerpt",
            "academic paper citation",
            "job application form",
        ]
        style = writing_styles[sample_index % len(writing_styles)]

        # Vary sentence complexity
        complexity_hints = [
            "Use simple, straightforward sentences.",
            "Use varied sentence lengths and structures.",
            "Include some complex sentences with subordinate clauses.",
            "Mix short and long sentences for natural flow.",
            "Use a conversational tone with varied sentence patterns.",
        ]
        complexity = complexity_hints[sample_index % len(complexity_hints)]

        # Create instruction text with proper escaping for JSON examples
        incorrect_example = '{"value": "Ravi Patel", "label": "FIRSTNAME"}'
        correct_surname_only = '{"value": "Patel", "label": "SURNAME"}'
        correct_firstname_only = '{"value": "Ravi", "label": "FIRSTNAME"}'
        correct_both = '{"value": "Ravi", "label": "FIRSTNAME"} {"value": "Patel", "label": "SURNAME"}'

        language = selected_language_country[0]
        country = selected_language_country[1]
        labels_list = ", ".join(labels.values())
        return f"""
        Generate one text sample in the style of a {style} that contains the following PII types:
        {labels_list}

        Instructions:
            1. Generate the sample in `{language}` and make the text as well as the PII data as realistic to the geographic area of `{country}`.
            2. Use only the PII types listed above
            3. Write in the style of a {style} - make it realistic and contextually appropriate
            4. {complexity}
            5. Contain 2-5 sentences with multiple entities (vary the number of sentences)
            6. Include at least one coreference chain with 3+ mentions (e.g., "Mark Johnson likes soccer. He is member of a soccer club. His email is mark.johnson@example.com.")
            7. Use varied reference types: pronouns (he, she, they, it), definite descriptions ("the customer", "the patient"), proper names, and possessive forms
            8. Vary the position of PII in sentences - don't always put names at the beginning
            9. Return the text sample with the included PII data and the type of PII (see example below)
            10. Include coreference information: group all mentions that refer to the same entity into clusters (e.g., if "John Doe", "He", and "His" all refer to the same person, they should be in one cluster)
            11. This is incorrect: {incorrect_example}. If the required labels only contain "surname", only generate a sample with a surname {correct_surname_only}. If the required labels only contain "first name", only generate a sample with a first name {correct_firstname_only}. If first name and surname are required, this is correct: {correct_both}.
            12. Be creative with your first and last names (use diverse ethnic backgrounds, cultural origins, and avoid common names like "John Smith", "Jane Doe", "Sarah Johnson", "Mike Wilson", etc.).
            13. Use diverse street names and city names (use names from different countries, cultures, and avoid common ones like "Main Street", "Oak Avenue", "New York", "Los Angeles", "London", "Paris", etc.).
            14. Vary the format of dates, phone numbers, and addresses to reflect different countries and regions
            15. Make the text feel natural and contextually appropriate - don't just list PII items
            16. Review your work before returning the samples. DO NOT create new label names. You can only use the labels listed above.
            17. Return the text sample in the following JSON format:
            ```json
            [{{
                "text": "The text sample",
                "privacy_mask": [
                    {{
                        "value": "The value",
                        "label": "The label"
                    }}
                ],
                "coreferences": [
                    {{
                        "cluster_id": 0,
                        "mentions": ["John Doe", "He", "His"],
                        "entity_type": "person"
                    }}
                ]
            }}]
            ```

        Here is an example:
        {example_string}
        """.strip()

    @staticmethod
    def build_review_prompt(sample: dict[str, Any], expected_labels: str) -> str:
        """Build a prompt for reviewing a sample."""
        sample_json = json.dumps(sample, indent=2)
        return f"""You are reviewing a dataset for training a PII (Personally Identifiable Information) detection model. The dataset contains text samples with privacy mask annotations and coreference information.

**Dataset Purpose:**
This dataset is used to train a token classification model (BERT-based) to detect and label PII entities in text. The model needs to identify various types of PII including names, addresses, emails, phone numbers, IDs, etc.

**Expected PII Labels:**
{expected_labels}

**Dataset Structure:**
Each sample contains:
- `text`: The original text containing PII
- `privacy_mask`: Array of PII entities with their values and labels
- `coreferences`: Array of coreference clusters grouping mentions that refer to the same entity

**Sample to Review:**
```json
{sample_json}
```

**Your Task:**
Please review and correct the dataset example! Return the correct JSON if it needs correction. Just return the JSON structure, no explanation, no extra text."""
