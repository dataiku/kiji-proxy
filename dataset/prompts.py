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
        },
        {
            "text": "John Doe is a 30-year-old man who lives in Los Angeles. He lives at 3566 Main Street, Los Angeles, CA 90012. His phone number is 555-123-4567 and his email is john.doe@example.com.",
            "privacy_mask": [
                {"value": "John", "label": "FIRSTNAME"},
                {"value": "Doe", "label": "SURNAME"},
                {"value": "30", "label": "AGE"},
                {"value": "Los Angeles", "label": "CITY"},
                {"value": "Main Street", "label": "STREET"},
                {"value": "3566", "label": "BUILDINGNUM"},
                {"value": "90012", "label": "ZIP"},
                {"value": "CA", "label": "STATE"},
                {"value": "555-123-4567", "label": "PHONENUMBER"},
                {"value": "john.doe@example.com", "label": "EMAIL"},
            ],
        },
    ]

    @staticmethod
    def build_generation_prompt(
        labels: dict[str, str], languages: list[str]
    ) -> str:
        """Build a prompt for generating PII samples."""
        # Pick a random sample for the example
        random_sample = random.choice(PromptBuilder.SAMPLE_DATA)
        example_string = f"""
        ```json\n{json.dumps(random_sample, indent=2)}\n```
        """

        # Create instruction text with proper escaping for JSON examples
        incorrect_example = '{"value": "Ravi Patel", "label": "FIRSTNAME"}'
        correct_surname_only = '{"value": "Patel", "label": "SURNAME"}'
        correct_firstname_only = '{"value": "Ravi", "label": "FIRSTNAME"}'
        correct_both = (
            '{"value": "Ravi", "label": "FIRSTNAME"} {"value": "Patel", "label": "SURNAME"}'
        )

        return f"""
        Generate one text sample that contain the following PII types: \n {", \n".join(labels.values())}

        Instructions:
            1. Generate the samples in following languages: {", ".join(languages)}
            2. Use only the PII types listed above
            3. Contain 2-4 sentences with multiple entities
            4. Include at least one coreference chain with 3+ mentions (e.g., "Mark Johnson likes soccer. He is member of a soccer club. His email is mark.johnson@example.com.")
            5. Use varied reference types: pronouns, definite descriptions, proper names
            6. Return the text sample with the included PII data and the type of PII (see example below)
            7. Include coreference information: group all mentions that refer to the same entity into clusters (e.g., if "John Doe", "He", and "His" all refer to the same person, they should be in one cluster)
            8. This is incorrect: {incorrect_example}. If the required labels only contain "surname", only generate a sample with a surname {correct_surname_only}. If the required labels only contain "first name", only generate a sample with a first name {correct_firstname_only}. If first name and surname are required, this is correct: {correct_both}.
            9. Be creative with your first and last names (use different ethnic backgrounds and avoid using names like "John Smith", "Jane Doe", "Sarah Johnson", etc.). 
            10. Use different street names and city names (use different ethnic backgrounds and avoid using names like "Main Street", "New York", "Los Angeles", etc.).
            11. Review your work before returning the samples. DO NOT create new label names. You can only use the labels listed above.
            12. Return the text sample in the following JSON format:
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

