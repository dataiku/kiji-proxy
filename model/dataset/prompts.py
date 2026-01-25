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
                    "entity_type": "person",
                    "mentions": [
                        {
                            "text": "Michael Chen",
                            "type": "name",
                            "privacy_mask_labels": ["FIRSTNAME", "SURNAME"],
                        },
                        {"text": "I", "type": "pronoun"},
                        {"text": "my", "type": "pronoun"},
                    ],
                },
                {
                    "cluster_id": 1,
                    "entity_type": "person",
                    "mentions": [
                        {
                            "text": "Sarah",
                            "type": "name",
                            "privacy_mask_labels": ["FIRSTNAME"],
                        },
                        {"text": "you", "type": "pronoun"},
                        {"text": "your", "type": "pronoun"},
                    ],
                },
            ],
            "language": "English",
            "country": "United States",
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
                    "entity_type": "person",
                    "mentions": [
                        {"text": "the patient", "type": "reference"},
                        {
                            "text": "Maria Santos",
                            "type": "name",
                            "privacy_mask_labels": ["FIRSTNAME", "SURNAME"],
                        },
                        {"text": "Her", "type": "pronoun"},
                        {"text": "Ms. Santos", "type": "reference"},
                        {"text": "her", "type": "pronoun"},
                    ],
                }
            ],
            "language": "English",
            "country": "United States",
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
            "language": "English",
            "country": "Japan",
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
                    "entity_type": "person",
                    "mentions": [
                        {"text": "My", "type": "pronoun"},
                        {"text": "I", "type": "pronoun"},
                    ],
                }
            ],
            "language": "English",
            "country": "France",
        },
    ]

    @staticmethod
    def build_ner_generation_prompt(
        labels: dict[str, dict],
        languages_countries: list[tuple[str, str]],
        sample_index: int = 0,
    ) -> tuple[str, str, str]:
        """
        Build a prompt for generating NER (Named Entity Recognition) samples with PII annotations.

        Args:
            labels: Dictionary of PII labels to include
            languages_countries: List of (language, country) tuples
            sample_index: Index of sample being generated (for variation)

        Returns:
            Tuple of (prompt, language, country)
        """
        # Use sample_index to select example (for more variation)
        example_index = sample_index % len(PromptBuilder.SAMPLE_DATA)
        random_sample = PromptBuilder.SAMPLE_DATA[example_index]

        # Extract only text and privacy_mask for NER example
        ner_example = {
            "text": random_sample["text"],
            "privacy_mask": random_sample["privacy_mask"],
            "language": random_sample["language"],
            "country": random_sample["country"],
        }
        example_string = f"```json\n{json.dumps(ner_example, indent=2)}\n```"

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

        language = selected_language_country[0]
        country = selected_language_country[1]
        labels_list = ", ".join(labels.keys())
        label_instructions = []
        for label_key, label_values in labels.items():
            if not label_values["hints"]:
                continue
            label_instructions += f"{label_key}:"
            label_instructions += label_values["hints"]
            label_instructions += "\n"
        label_instructions = "\n".join(label_instructions)

        return (
            f"""
        Generate one text sample in the style of a {style} that contains the following PII types:
        {labels_list}

        Instructions:
            1. Generate the sample in `{language}` and make the text as well as the PII data as realistic to the geographic area of `{country}`
            2. Use only the PII types listed above
            3. Write in the style of a {style} - make it realistic and contextually appropriate
            4. {complexity}
            5. Contain 2-5 sentences with multiple entities (vary the number of sentences)
            6. Vary the position of PII in sentences - don't always put names at the beginning
            7. Return the text sample with the included PII data and the type of PII (see example below)
            8. Make the text feel natural and contextually appropriate - don't just list PII items
            9. Review your work before returning the samples. DO NOT create new label names. You can only use the labels listed above

        Label-specific Instructions:
            {label_instructions}

        Schema Instructions:
            Return the text sample in the following JSON format:
            ```json
            [{{
                "text": "The text sample",
                "privacy_mask": [
                    {{
                        "value": "The value",
                        "label": "The label"
                    }}
                ],
                "language": "{language}",
                "country": "{country}"
            }}]
            ```

        Here is an example:
        {example_string}
        """.strip(),
            language,
            country,
        )

    @staticmethod
    def build_coreference_generation_prompt(
        ner_sample: dict[str, Any],
        language: str,
        country: str,
    ) -> str:
        """
        Build a prompt for generating coreference annotations for an existing NER sample.

        Args:
            ner_sample: Dictionary containing 'text' and 'privacy_mask' fields
            language: Language of the text
            country: Country/region for the text

        Returns:
            Prompt string for coreference generation
        """

        # Show example with coreferences
        example_index = 0
        example_sample = PromptBuilder.SAMPLE_DATA[example_index]
        example_string = f"```json\n{json.dumps(example_sample, indent=2)}\n```"

        current_sample_string = f"```json\n{json.dumps(ner_sample, indent=2)}\n```"

        return f"""
        You are provided with a text sample that already has PII (Personally Identifiable Information) annotations. Your task is to add coreference information to this sample.

        **Text Language:** {language}
        **Geographic Region:** {country}

        **Coreference Guidelines:**
        1. Identify all mentions that refer to the same real-world entity in third person (person, and organization)
        2. Group these mentions into coreference clusters
        3. Each cluster must have:
           - `cluster_id`: A unique integer identifier
           - `entity_type`: Type of entity ("person" or "organization")
           - `mentions`: Array of mention objects, each containing:
             * `text`: The exact string as it appears in the text (case-sensitive)
             * `type`: One of "name", "pronoun", or "reference"
             * `privacy_mask_labels`: (Optional) Array of privacy_mask labels this mention maps to (e.g., ["FIRSTNAME", "SURNAME"] for "John Doe")
        4. Include pronouns (he, she, they, it, his, her, their) or the language equivalents
        5. Include definite descriptions ("the customer", "the patient", "the employee")
        6. Include proper names and variations (e.g., "Maria Santos", "Ms. Santos")
        7. CRITICAL mention validation rules:
           - Each mention "text" must be an EXACT string that appears as a complete word in the text
           - Do NOT include substrings (e.g., do not include "her" if the text only contains "here" or "there")
           - Verify word boundaries - "her" should match "her" but not "where", "here", or "there"
           - If a mention appears multiple times in the text, include it once in the mentions array
           - For names that appear in privacy_mask, include the "privacy_mask_labels" field to indicate which labels correspond to this mention

        **Current Sample (needs coreference annotations):**
        {current_sample_string}

        **Example with Coreferences:**
        {example_string}

        **Your Task:**
        Add coreference annotations to the provided sample. Return the complete sample with the added "coreferences" field in JSON format. Just return the JSON structure, no explanation, no extra text.

        Expected output format:
        ```json
        {{
            "text": "...",
            "privacy_mask": [...],
            "coreferences": [
                {{
                    "cluster_id": 0,
                    "entity_type": "person",
                    "mentions": [
                        {{"text": "John Doe", "type": "name", "privacy_mask_labels": ["FIRSTNAME", "SURNAME"]}},
                        {{"text": "He", "type": "pronoun"}},
                        {{"text": "his", "type": "pronoun"}}
                    ]
                }}
            ],
            "language": "{language}",
            "country": "{country}"
        }}
        ```
        """.strip()

    @staticmethod
    def build_generation_prompt(
        labels: dict[str, dict],
        languages_countries: list[tuple[str, str]],
        sample_index: int = 0,
    ) -> tuple[str, str, str]:
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

        language = selected_language_country[0]
        country = selected_language_country[1]

        labels_list = ", ".join(labels.keys())
        label_instructions = []
        for label_key, label_values in labels.items():
            if not label_values["hints"]:
                continue
            label_instructions += f"{label_key}:"
            label_instructions += label_values["hints"]
            label_instructions += "\n"
        label_instructions = "\n".join(label_instructions)
        return (
            f"""
        Generate one text sample in the style of a {style} that contains the following PII types:
        {labels_list}

        General Instructions:
            1. Generate the sample in `{language}` and make the text as well as the PII data as realistic to the geographic area of `{country}`.
            2. Write in the style of a {style} - make it realistic and contextually appropriate
            3. {complexity}
            4. Contain 2-5 sentences with multiple entities (vary the number of sentences)
            5. Use varied reference types: pronouns (he, she, they, it), definite descriptions ("the customer", "the patient"), proper names, and possessive forms
            6. Include at least one coreference chain with 3+ mentions (e.g., "Mark Johnson likes soccer. He is member of a soccer club. His email is mark.johnson@example.com.")
            7. Vary the position of PII in sentences - don't always put names at the beginning
            8. Make the text feel natural and contextually appropriate - don't just list PII items
            9. Review your work before returning the samples. DO NOT create new label names. You can only use the labels listed above.
            10. Use only the PII types listed above

        Label-specific Instructions:
            {label_instructions}

        Schema Instructions:

            Return the text sample in the following JSON format:
            ```json
            [{{
                "text": "The text sample",
                "privacy_mask": [
                    {{
                        "value": "The value",
                        "label": "The label"
                    }}
                ],
                "coreferences": [],
                "language": "{language}",
                "country": "{country}"
            }}]
            ```

        Here is an example:
        {example_string}
        """.strip(),
            language,
            country,
        )

    @staticmethod
    def build_review_prompt(
        sample: dict[str, Any], expected_labels: str, language: str, country: str
    ) -> str:
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

**Coreference Review Guidelines:**
    1. All mentions within a cluster must refer to the same real-world entity
    2. Clusters must be correctly separated (different entities should not be in the same cluster)
    3. The text must be in `{language}` and realistic to the geographic area of `{country}`
    4. All relevant mentions must be included (pronouns, definite descriptions, proper names, possessive forms)
    5. The `entity_type` field must accurately describe the type of entity (options are "person", "organization")
    6. Coreference clusters must be meaningful and help identify relationships between PII mentions
    7. Mention structure requirements:
       - `text`: The exact string as it appears in the text (case-sensitive)
       - `type`: One of "name", "pronoun", or "reference"
       - `privacy_mask_labels`: (Optional) Array of privacy_mask labels this mention maps to (e.g., ["FIRSTNAME", "SURNAME"] for "John Doe")
    8. CRITICAL mention validation rules:
       - Each mention text must be an EXACT string that appears as a complete word in the text (case-sensitive)
       - Mentions must be complete words, not substrings (e.g., "her" should NOT be included if the text only contains "here", "there", or "where")
       - If a mention appears multiple times, include it once in the mentions array - the system will automatically find all occurrences
       - Use word boundaries to ensure accurate matching (e.g., "her" should match "her" but not "here")
       - For names that appear in privacy_mask, include the "privacy_mask_labels" field to help map split names (e.g., "John Doe" → ["FIRSTNAME", "SURNAME"])

**Sample to Review:**
```json
{sample_json}
```

**Your Task:**
Please review and correct the dataset example! Check both the privacy_mask annotations and the coreference clusters. Return the correct JSON if it needs correction. Just return the JSON structure, no explanation, no extra text."""
