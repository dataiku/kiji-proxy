"""Label utilities for PII detection."""
import random
from typing import ClassVar


class LabelUtils:
    """Utilities for managing PII labels and mappings."""

    # Single source of truth: Label code to human-readable description mapping
    # This dictionary defines all available PII labels and their descriptions
    LABEL_DESCRIPTIONS: ClassVar[dict[str, str]] = {
        "SURNAME": "surname",
        "FIRSTNAME": "first name",
        # "MIDDLENAME": "middle name",
        "BUILDINGNUM": "building number",
        "DATEOFBIRTH": "date of birth",
        "EMAIL": "email",
        "PHONENUMBER": "phone number",
        "DOB": "date of birth",
        # "ADDRESS": "address",
        "CITY": "city",
        "URL": "url",
        "COMPANYNAME": "company name",
        "STATE": "state",
        "ZIP": "zip",
        "STREET": "street",
        "COUNTRY": "country",
        "SSN": "social security number",
        "DRIVER_LICENSE": "driver license ID",
        "PASSPORT": "passport ID",
        "NATIONAL_ID": "national id",
        "IDCARDNUM": "id card number",
        "TAXNUM": "tax number",
        "LICENSEPLATENUM": "license plate number",
        "PASSWORD": "password",
        "IBAN": "IBAN",
        "AGE": "age",
        # "TIME": "time",
        # "SECONDARYADDRESS": "secondary address",
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
        cls, labels: dict[str, str], return_count: int, seed: int | None = None
    ) -> dict[str, str]:
        """
        Select a subset of labels with special handling for address labels.

        If any address-related label is selected, all address labels are included.

        Args:
            labels: Dictionary of label codes to descriptions
            return_count: Number of labels to randomly select
            seed: Optional random seed for reproducibility

        Returns:
            Filtered dictionary containing only the selected labels
        """
        # Select subset of labels (use fresh random state for variety)
        rng = random.Random(seed) if seed is not None else random
        selected_keys = rng.sample(list(labels.keys()), return_count)

        # If any address label is selected, include all address labels
        if any(label in selected_keys for label in cls.ADDRESS_LABELS):
            selected_keys.extend(cls.ADDRESS_LABELS)
        selected_keys = list(set(selected_keys))

        # Filter labels to only include selected keys (which includes all address labels if any was selected)
        return {key: labels[key] for key in selected_keys}

