"""Label utilities for PII detection."""

import random
from pathlib import Path
from typing import ClassVar


class LabelUtils:
    """Utilities for managing PII labels and mappings."""

    incorrect_example = '{"value": "Ravi Patel", "label": "FIRSTNAME"}'
    correct_firstname = '{"value": "Ravi", "label": "FIRSTNAME"}'
    correct_surname = '{"value": "Patel", "label": "SURNAME"}'

    # Single source of truth: Label code to human-readable description mapping
    # This dictionary defines all available PII labels and their descriptions
    LABEL_DESCRIPTIONS: ClassVar[dict[str, dict]] = {
        "SURNAME": {
            "name": "Last name",
            "color": "#e74c3c",
            "chance": 0.5,
            "hints": [
                "Consider different name lengths and phonetic variety",
                "Geographic origins (East Asian, South Asian, African, European, Latin American, Middle Eastern, etc.)",
                "Common and uncommon names within each culture",
                "Gender presentation",
                f"Correct surname example: {correct_surname}",
            ],
        },
        "FIRSTNAME": {
            "name": "First name",
            "color": "#3498db",
            "chance": 0.5,
            "hints": [
                "Consider different name lengths and phonetic variety",
                "Geographic origins (East Asian, South Asian, African, European, Latin American, Middle Eastern, etc.)",
                "Common and uncommon names within each culture",
                "Gender presentation",
                f"Correct first name example: {correct_firstname}",
                f"Incorrect example: {incorrect_example}.",
            ],
        },
        "BUILDINGNUM": {
            "name": "Building number",
            "color": "#f39c12",
            "chance": 0.1,
            "hints": ["Consider different formats applicable to the geographic area"],
        },
        "DATEOFBIRTH": {
            "name": "Date of birth",
            "color": "#d35400",
            "chance": 0.1,
            "hints": ["Use the correct date format applicable in the geographic area"],
        },
        "EMAIL": {
            "name": "Email",
            "color": "#9b59b6",
            "chance": 0.1,
            "hints": ["Use a diverse set of email providers and web domains"],
        },
        "PHONENUMBER": {
            "name": "Phone number",
            "color": "#27ae60",
            "chance": 0.1,
            "hints": [
                "Use a phone number format as well as code and area code applicable in the geographic area"
            ],
        },
        "CITY": {
            "name": "City",
            "color": "#16a085",
            "chance": 0.1,
            "hints": [
                'Use diverse city names, avoid common ones like "New York", "Los Angeles", "London", "Paris", etc.'
            ],
        },
        "URL": {"name": "URL", "color": "#e67e22", "chance": 0.1, "hints": []},
        "COMPANYNAME": {
            "name": "Company name",
            "color": "#795548",
            "chance": 0.1,
            "hints": [],
        },
        "STATE": {"name": "State", "color": "#8e44ad", "chance": 0.1, "hints": []},
        "ZIP": {"name": "Zip code", "color": "#e91e63", "chance": 0.1, "hints": []},
        "STREET": {
            "name": "Street",
            "color": "#c0392b",
            "chance": 0.1,
            "hints": [
                'Use diverse street names and avoid common ones like "Main Street", and "Oak Avenue".'
            ],
        },
        "COUNTRY": {"name": "Country", "color": "#2c3e50", "chance": 0.1, "hints": []},
        "SSN": {
            "name": "Social Security Number",
            "color": "#c2185b",
            "chance": 0.1,
            "hints": [],
        },
        "DRIVERLICENSENUM": {
            "name": "Driver's License Number",
            "color": "#00bcd4",
            "chance": 0.1,
            "hints": [],
        },
        "PASSPORTID": {
            "name": "Passport ID",
            "color": "#ff5722",
            "chance": 0.1,
            "hints": [],
        },
        "NATIONALID": {
            "name": "National ID",
            "color": "#4caf50",
            "chance": 0.1,
            "hints": [],
        },
        "IDCARDNUM": {
            "name": "ID Card Number",
            "color": "#9c27b0",
            "chance": 0.1,
            "hints": [],
        },
        "TAXNUM": {
            "name": "Tax Number",
            "color": "#607d8b",
            "chance": 0.1,
            "hints": [],
        },
        "LICENSEPLATENUM": {
            "name": "License Plate Number",
            "color": "#ffc107",
            "chance": 0.1,
            "hints": [],
        },
        "PASSWORD": {
            "name": "Password",
            "color": "#f44336",
            "chance": 0.1,
            "hints": ["Use different complexities of passwords"],
        },
        "IBAN": {"name": "IBAN", "color": "#673ab7", "chance": 0.1, "hints": []},
        "AGE": {"name": "Age", "color": "#2980b9", "chance": 0.1, "hints": []},
        "SECURITYTOKEN": {
            "name": "API Security Tokens",
            "color": "#9c27b0",
            "chance": 0.1,
            "hints": [
                "Mimic API security tokens similar to AWS, Google Cloud, Microsoft, OpenAI, etc. "
            ],
        },
    }

    # Derived from LABEL_DESCRIPTIONS to ensure consistency
    STANDARD_PII_LABELS: ClassVar[list[str]] = list(LABEL_DESCRIPTIONS.keys())

    # Address-related labels that should be grouped together
    ADDRESS_LABELS: ClassVar[list[str]] = [
        "BUILDINGNUM",
        # "ADDRESS",
        "CITY",
        "STATE",
        "ZIP",
        "COUNTRY",
        "STREET",
    ]

    @classmethod
    def create_standard_label2id(cls) -> tuple[dict[str, int], dict[int, str]]:
        """
        Create a standard label2id and id2label mapping for all PII labels.

        Returns:
            Tuple of (label2id, id2label) dictionaries
        """
        label2id = {"O": 0}
        id2label = {0: "O"}

        # Add B- and I- prefixes for each label (derived from LABEL_DESCRIPTIONS)
        for label in cls.STANDARD_PII_LABELS:
            b_label = f"B-{label}"
            i_label = f"I-{label}"
            label2id[b_label] = len(label2id)
            label2id[i_label] = len(label2id)
            id2label[len(id2label)] = b_label
            id2label[len(id2label)] = i_label

        # Add IGNORE for -100
        id2label[-100] = "IGNORE"

        return label2id, id2label

    @classmethod
    def select_label_subset(
        cls, labels: dict[str, dict], return_count: int, seed: int | None = None
    ) -> dict[str, dict]:
        """
        Select a subset of labels using weighted random selection based on 'chance' field.
        If any address-related label is selected, all address labels are included.

        Args:
            labels: Dictionary of label codes to descriptions
            return_count: Number of labels to randomly select
            seed: Optional random seed for reproducibility

        Returns:
            Filtered dictionary containing only the selected labels
        """
        rng = random.Random(seed) if seed is not None else random

        # Extract labels and their weights (default to 1.0 if no 'chance' specified)
        label_keys = list(labels.keys())
        weights = [labels[key].get("chance", 0.1) for key in label_keys]

        # Normalize weights to probabilities
        total_weight = sum(weights)
        if total_weight == 0:
            # Fallback to uniform distribution if all weights are 0
            weights = [1.0] * len(label_keys)
            total_weight = len(label_keys)

        probabilities = [w / total_weight for w in weights]

        # Use weighted random selection (without replacement)
        selected_keys = []
        remaining_keys = label_keys.copy()
        remaining_probs = probabilities.copy()

        for _ in range(min(return_count, len(remaining_keys))):
            # Renormalize probabilities
            prob_sum = sum(remaining_probs)
            normalized_probs = [p / prob_sum for p in remaining_probs]

            # Select one label based on weights
            selected_idx = rng.choices(
                range(len(remaining_keys)), weights=normalized_probs, k=1
            )[0]

            selected_keys.append(remaining_keys[selected_idx])

            # Remove selected item from remaining pool
            remaining_keys.pop(selected_idx)
            remaining_probs.pop(selected_idx)

        # If any address label is selected, include all address labels
        if any(label in selected_keys for label in cls.ADDRESS_LABELS):
            selected_keys.extend(cls.ADDRESS_LABELS)
        selected_keys = list(set(selected_keys))

        # Filter labels to only include selected keys (which includes all address labels if any was selected)
        return {key: labels[key] for key in selected_keys}

    @classmethod
    def generate_labelstudio_config(cls, output_path: str | Path | None = None) -> str:
        """
        Generate Label Studio configuration XML based on LABEL_DESCRIPTIONS.

        Args:
            output_path: Optional path to write the configuration file.
                        If None, returns the XML string without writing.
                        Defaults to model/dataset/labelstudio/LabelingConfig.txt

        Returns:
            The generated XML configuration as a string
        """
        # Group labels by category for better organization
        categories = {
            "Person Information": ["SURNAME", "FIRSTNAME"],
            "Contact Information": ["EMAIL", "PHONENUMBER", "URL"],
            "Address Information": [
                "BUILDINGNUM",
                "STREET",
                "CITY",
                "STATE",
                "ZIP",
                "COUNTRY",
            ],
            "Date/Time Information": ["DATEOFBIRTH", "AGE"],
            "Identification Numbers": [
                "SSN",
                "DRIVERLICENSENUM",
                "PASSPORTID",
                "NATIONALID",
                "IDCARDNUM",
                "TAXNUM",
                "LICENSEPLATENUM",
            ],
            "Financial Information": ["IBAN"],
            "Organization Information": ["COMPANYNAME"],
            "Security Information": ["PASSWORD", "SECURITY_TOKEN"],
        }

        # Start building the XML
        xml_lines = [
            "<View>",
            '  <Header value="Entity &amp; Coreference Annotation"/>',
            "",
            '  <Labels name="entities" toName="text">',
        ]

        # Add labels by category
        for category, label_codes in categories.items():
            xml_lines.append(f"    <!-- {category} -->")
            for label_code in label_codes:
                if label_code in cls.LABEL_DESCRIPTIONS:
                    label_info = cls.LABEL_DESCRIPTIONS[label_code]
                    color = label_info.get("color", "#D3D3D3")
                    if not color:
                        raise ValueError(f"Missing color for label {label_code}")
                    xml_lines.append(
                        f'    <Label value="{label_code}" background="{color}"/>'
                    )
            xml_lines.append("")

        # Add OTHER label
        xml_lines.extend(
            [
                "    <!-- Other -->",
                '    <Label value="OTHER" background="#D3D3D3" />',
                "",
                "  </Labels>",
                "",
                '  <Text name="text" value="$text" granularity="word"/>',
                "",
                "  <Relations>",
                '    <Relation value="refers-to"/>',
                "  </Relations>",
                "</View>",
                "",
                "<!-- Sample task:",
                '{"text": "John Smith met Mary at the park. He greeted her warmly."}',
                "-->",
            ]
        )

        xml_content = "\n".join(xml_lines)

        # Write to file if path is provided
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(xml_content)

        return xml_content

    @classmethod
    def update_labelstudio_config(cls) -> None:
        """
        Update the Label Studio configuration file with the latest labels.

        This method writes to model/dataset/labelstudio/LabelingConfig.txt
        """
        # Determine the path relative to this file
        current_file = Path(__file__)
        # Go up to model/dataset/, then to labelstudio/
        config_path = current_file.parent / "labelstudio" / "LabelingConfig.txt"

        cls.generate_labelstudio_config(output_path=config_path)
        print(f"Updated Label Studio configuration at: {config_path}")
