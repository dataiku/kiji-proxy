"""JSON schemas for structured LLM outputs."""


def get_ner_sample_schema() -> dict:
    """Get JSON schema for NER (Named Entity Recognition) sample generation."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "privacy_mask": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "label": {"type": "string"},
                        },
                        "required": ["value", "label"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["text", "privacy_mask"],
            "additionalProperties": False,
        },
    }


def get_coreference_sample_schema() -> dict:
    """Get JSON schema for coreference annotation generation."""
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "privacy_mask": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "label": {"type": "string"},
                    },
                    "required": ["value", "label"],
                    "additionalProperties": False,
                },
            },
            "coreferences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cluster_id": {
                            "type": "integer",
                            "description": "Unique identifier for the coreference cluster",
                        },
                        "mentions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "type": {
                                        "type": "string",
                                        "enum": ["name", "pronoun", "reference"],
                                    },
                                    "privacy_mask_labels": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Optional array of privacy_mask labels this mention maps to",
                                    },
                                },
                                "required": ["text", "type"],
                                "additionalProperties": False,
                            },
                            "description": "List of mention objects that refer to the same entity",
                        },
                        "entity_type": {
                            "type": "string",
                            "enum": ["person", "organization"],
                            "description": "Type of entity",
                        },
                    },
                    "required": ["cluster_id", "mentions", "entity_type"],
                    "additionalProperties": False,
                },
                "description": "Coreference clusters grouping mentions that refer to the same entity",
            },
        },
        "required": ["text", "privacy_mask", "coreferences"],
        "additionalProperties": False,
    }


def get_pii_sample_schema() -> dict:
    """Get JSON schema for PII sample generation."""
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "language": {
                    "type": "string",
                    "description": "Language of the text sample (e.g., 'English', 'Spanish', 'French')",
                },
                "country": {
                    "type": "string",
                    "description": "Geographic area/country for the text sample (e.g., 'United States', 'France', 'Japan')",
                },
                "privacy_mask": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "label": {"type": "string"},
                        },
                        "required": ["value", "label"],
                        "additionalProperties": False,
                    },
                },
                "coreferences": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cluster_id": {
                                "type": "integer",
                                "description": "Unique identifier for the coreference cluster",
                            },
                            "mentions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of text mentions that refer to the same entity (e.g., ['John Doe', 'He', 'His'])",
                            },
                            "entity_type": {
                                "type": "string",
                                "description": "Type of entity (e.g., 'person', 'organization', 'location')",
                            },
                        },
                        "required": ["cluster_id", "mentions", "entity_type"],
                        "additionalProperties": False,
                    },
                    "description": "Coreference clusters grouping mentions that refer to the same entity",
                },
            },
            "required": ["text", "privacy_mask", "coreferences", "language", "country"],
            "additionalProperties": False,
        },
    }


def get_review_sample_schema() -> dict:
    """Get JSON schema for sample review."""
    return {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "language": {
                "type": "string",
                "description": "Language of the text sample (e.g., 'English', 'Spanish', 'French')",
            },
            "country": {
                "type": "string",
                "description": "Geographic area/country for the text sample (e.g., 'United States', 'France', 'Japan')",
            },
            "privacy_mask": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "label": {"type": "string"},
                    },
                    "required": ["value", "label"],
                    "additionalProperties": False,
                },
            },
            "coreferences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cluster_id": {
                            "type": "integer",
                            "description": "Unique identifier for the coreference cluster",
                        },
                        "mentions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of text mentions that refer to the same entity (e.g., ['John Doe', 'He', 'His'])",
                        },
                        "entity_type": {
                            "type": "string",
                            "description": "Type of entity (e.g., 'person', 'organization', 'location')",
                        },
                    },
                    "required": ["cluster_id", "mentions", "entity_type"],
                    "additionalProperties": False,
                },
                "description": "Coreference clusters grouping mentions that refer to the same entity",
            },
        },
        "required": ["text", "privacy_mask", "coreferences", "language", "country"],
        "additionalProperties": False,
    }
