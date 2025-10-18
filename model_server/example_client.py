"""
Example client for PII Detection API

This script demonstrates how to use the PII Detection API
from Python applications.

Usage:
    python example_client.py

    # With custom server URL:
    python example_client.py --url http://localhost:8000
"""

import argparse

import requests


class PIIDetectionClient:
    """Client for PII Detection API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> dict:
        """
        Check server health.

        Returns:
            Health status dictionary
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_model_info(self) -> dict:
        """
        Get model information.

        Returns:
            Model information dictionary
        """
        response = requests.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()

    def detect_pii(self, text: str, include_timing: bool = True) -> dict:
        """
        Detect PII in a single text.

        Args:
            text: Text to analyze
            include_timing: Include inference timing

        Returns:
            Detection results dictionary
        """
        response = requests.post(
            f"{self.base_url}/detect",
            json={"text": text, "include_timing": include_timing},
        )
        response.raise_for_status()
        return response.json()

    def detect_pii_batch(self, texts: list, include_timing: bool = True) -> dict:
        """
        Detect PII in multiple texts.

        Args:
            texts: List of texts to analyze
            include_timing: Include inference timing

        Returns:
            Batch detection results dictionary
        """
        response = requests.post(
            f"{self.base_url}/detect/batch",
            json={"texts": texts, "include_timing": include_timing},
        )
        response.raise_for_status()
        return response.json()


def print_detection_result(result: dict):
    """Pretty print detection result."""
    print(f"\nText: {result['text']}")

    if result.get("inference_time_ms"):
        print(f"Inference Time: {result['inference_time_ms']:.2f} ms")

    print(f"\nDetected {result['entity_count']} PII entities:")

    if result["entities"]:
        for entity in result["entities"]:
            print(
                f"  • [{entity['label']}] '{entity['text']}' "
                f"(position {entity['start_pos']}-{entity['end_pos']})"
            )
    else:
        print("  (No PII detected)")


def example_single_detection(client: PIIDetectionClient):
    """Example: Single text detection."""
    print("\n" + "=" * 80)
    print("Example 1: Single Text Detection")
    print("=" * 80)

    text = "My name is John Smith, email is john.smith@company.com, and phone is 555-123-4567"

    result = client.detect_pii(text)
    print_detection_result(result)


def example_batch_detection(client: PIIDetectionClient):
    """Example: Batch detection."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Detection")
    print("=" * 80)

    texts = [
        "Contact Sarah at sarah.jones@email.com",
        "Patient DOB: 03/15/1985, SSN: 123-45-6789",
        "Credit card: 4532-1234-5678-9010",
        "IP address: 192.168.1.1, Username: admin",
    ]

    result = client.detect_pii_batch(texts)

    print(f"\nProcessed {len(texts)} texts")
    print(f"Total entities detected: {result['total_entities']}")

    if result.get("total_inference_time_ms"):
        print(f"Total time: {result['total_inference_time_ms']:.2f} ms")
        print(f"Average time: {result['average_inference_time_ms']:.2f} ms per text")

    for i, res in enumerate(result["results"], 1):
        print(f"\n--- Text {i} ---")
        print_detection_result(res)


def example_redaction(client: PIIDetectionClient):
    """Example: Redact PII from text."""
    print("\n" + "=" * 80)
    print("Example 3: PII Redaction")
    print("=" * 80)

    text = "Please contact Dr. Emily Chen at emily.chen@hospital.com or 555-987-6543"

    print(f"Original: {text}")

    result = client.detect_pii(text, include_timing=False)

    # Redact PII (replace with [REDACTED])
    redacted = text
    # Sort entities by start position in reverse to maintain positions
    for entity in sorted(result["entities"], key=lambda e: e["start_pos"], reverse=True):
        redacted = (
            redacted[: entity["start_pos"]] + f"[{entity['label']}]" + redacted[entity["end_pos"] :]
        )

    print(f"Redacted: {redacted}")


def example_anonymization(client: PIIDetectionClient):
    """Example: Anonymize PII with fake data."""
    print("\n" + "=" * 80)
    print("Example 4: PII Anonymization")
    print("=" * 80)

    text = "My email is john.doe@company.com and my phone is 555-1234"

    print(f"Original: {text}")

    result = client.detect_pii(text, include_timing=False)

    # Define replacement values for each PII type
    replacements = {
        "EMAIL": "user@example.com",
        "PHONE": "000-0000",
        "SSN": "000-00-0000",
        "CREDIT_CARD": "0000-0000-0000-0000",
        "USERNAME": "anonymous",
        "GIVENNAME": "John",
        "SURNAME": "Doe",
    }

    # Anonymize PII
    anonymized = text
    for entity in sorted(result["entities"], key=lambda e: e["start_pos"], reverse=True):
        replacement = replacements.get(entity["label"], "[ANONYMIZED]")
        anonymized = (
            anonymized[: entity["start_pos"]] + replacement + anonymized[entity["end_pos"] :]
        )

    print(f"Anonymized: {anonymized}")


def example_filtering(client: PIIDetectionClient):
    """Example: Filter by PII type."""
    print("\n" + "=" * 80)
    print("Example 5: Filter by PII Type")
    print("=" * 80)

    text = "Contact: john.doe@email.com, Phone: 555-1234, SSN: 123-45-6789, Username: jdoe"

    result = client.detect_pii(text, include_timing=False)

    print(f"Text: {text}\n")

    # Group entities by type
    by_type = {}
    for entity in result["entities"]:
        if entity["label"] not in by_type:
            by_type[entity["label"]] = []
        by_type[entity["label"]].append(entity["text"])

    print("Entities by type:")
    for pii_type, values in sorted(by_type.items()):
        print(f"  {pii_type}: {', '.join(values)}")

    # Filter only sensitive PII (e.g., SSN, Credit Card)
    sensitive_types = {"SSN", "CREDIT_CARD", "PASSWORD"}
    sensitive_entities = [e for e in result["entities"] if e["label"] in sensitive_types]

    print(f"\nSensitive PII found: {len(sensitive_entities)}")
    for entity in sensitive_entities:
        print(f"  ⚠️  {entity['label']}: {entity['text']}")


def main():
    """Run example usage."""
    parser = argparse.ArgumentParser(description="PII Detection API Client Examples")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API server (default: http://localhost:8000)",
    )

    args = parser.parse_args()

    # Create client
    client = PIIDetectionClient(args.url)

    print("=" * 80)
    print("PII Detection API - Client Examples")
    print("=" * 80)
    print(f"Server URL: {args.url}")

    # Check server health
    try:
        health = client.health_check()
        print(f"Server Status: {health['status']}")
        print(f"Model Loaded: {health['model_loaded']}")
        print(f"Device: {health['device']}")

        if not health["model_loaded"]:
            print("\n❌ Model not loaded on server!")
            return

    except Exception as e:
        print(f"\n❌ Cannot connect to server: {e}")
        print("Make sure the server is running!")
        return

    # Get model info
    try:
        info = client.get_model_info()
        print(f"Model Type: {info['model_type']}")
        print(f"Supported Labels: {', '.join(info['labels'][:10])}...")
    except Exception as e:
        print(f"Could not get model info: {e}")

    # Run examples
    try:
        example_single_detection(client)
        example_batch_detection(client)
        example_redaction(client)
        example_anonymization(client)
        example_filtering(client)

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")


if __name__ == "__main__":
    main()
