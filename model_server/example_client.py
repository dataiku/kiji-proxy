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
import logging

import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    logger.info(f"\nText: {result['text']}")

    if result.get("inference_time_ms"):
        logger.info(f"Inference Time: {result['inference_time_ms']:.2f} ms")

    logger.info(f"\nDetected {result['entity_count']} PII entities:")

    if result["entities"]:
        for entity in result["entities"]:
            logger.info(
                f"  • [{entity['label']}] '{entity['text']}' "
                f"(position {entity['start_pos']}-{entity['end_pos']})"
            )
    else:
        logger.info("  (No PII detected)")


def example_single_detection(client: PIIDetectionClient):
    """Example: Single text detection."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 1: Single Text Detection")
    logger.info("=" * 80)

    text = "My name is John Smith, email is john.smith@company.com, and phone is 555-123-4567"

    result = client.detect_pii(text)
    print_detection_result(result)


def example_batch_detection(client: PIIDetectionClient):
    """Example: Batch detection."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 2: Batch Detection")
    logger.info("=" * 80)

    texts = [
        "Contact Sarah at sarah.jones@email.com",
        "Patient DOB: 03/15/1985, SSN: 123-45-6789",
        "Credit card: 4532-1234-5678-9010",
        "IP address: 192.168.1.1, Username: admin",
    ]

    result = client.detect_pii_batch(texts)

    logger.info(f"\nProcessed {len(texts)} texts")
    logger.info(f"Total entities detected: {result['total_entities']}")

    if result.get("total_inference_time_ms"):
        logger.info(f"Total time: {result['total_inference_time_ms']:.2f} ms")
        logger.info(
            f"Average time: {result['average_inference_time_ms']:.2f} ms per text"
        )

    for i, res in enumerate(result["results"], 1):
        logger.info(f"\n--- Text {i} ---")
        print_detection_result(res)


def example_redaction(client: PIIDetectionClient):
    """Example: Redact PII from text."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 3: PII Redaction")
    logger.info("=" * 80)

    text = "Please contact Dr. Emily Chen at emily.chen@hospital.com or 555-987-6543"

    logger.info(f"Original: {text}")

    result = client.detect_pii(text, include_timing=False)

    # Redact PII (replace with [REDACTED])
    redacted = text
    # Sort entities by start position in reverse to maintain positions
    for entity in sorted(
        result["entities"], key=lambda e: e["start_pos"], reverse=True
    ):
        redacted = (
            redacted[: entity["start_pos"]]
            + f"[{entity['label']}]"
            + redacted[entity["end_pos"] :]
        )

    logger.info(f"Redacted: {redacted}")


def example_anonymization(client: PIIDetectionClient):
    """Example: Anonymize PII with fake data."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 4: PII Anonymization")
    logger.info("=" * 80)

    text = "My email is john.doe@company.com and my phone is 555-1234"

    logger.info(f"Original: {text}")

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
    for entity in sorted(
        result["entities"], key=lambda e: e["start_pos"], reverse=True
    ):
        replacement = replacements.get(entity["label"], "[ANONYMIZED]")
        anonymized = (
            anonymized[: entity["start_pos"]]
            + replacement
            + anonymized[entity["end_pos"] :]
        )

    logger.info(f"Anonymized: {anonymized}")


def example_filtering(client: PIIDetectionClient):
    """Example: Filter by PII type."""
    logger.info("\n" + "=" * 80)
    logger.info("Example 5: Filter by PII Type")
    logger.info("=" * 80)

    text = (
        "Contact: john.doe@email.com, Phone: 555-1234, SSN: 123-45-6789, Username: jdoe"
    )

    result = client.detect_pii(text, include_timing=False)

    logger.info(f"Text: {text}\n")

    # Group entities by type
    by_type = {}
    for entity in result["entities"]:
        if entity["label"] not in by_type:
            by_type[entity["label"]] = []
        by_type[entity["label"]].append(entity["text"])

    logger.info("Entities by type:")
    for pii_type, values in sorted(by_type.items()):
        logger.info(f"  {pii_type}: {', '.join(values)}")

    # Filter only sensitive PII (e.g., SSN, Credit Card)
    sensitive_types = {"SSN", "CREDIT_CARD", "PASSWORD"}
    sensitive_entities = [
        e for e in result["entities"] if e["label"] in sensitive_types
    ]

    logger.info(f"\nSensitive PII found: {len(sensitive_entities)}")
    for entity in sensitive_entities:
        logger.info(f"  ⚠️  {entity['label']}: {entity['text']}")


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

    logger.info("=" * 80)
    logger.info("PII Detection API - Client Examples")
    logger.info("=" * 80)
    logger.info(f"Server URL: {args.url}")

    # Check server health
    try:
        health = client.health_check()
        logger.info(f"Server Status: {health['status']}")
        logger.info(f"Model Loaded: {health['model_loaded']}")
        logger.info(f"Device: {health['device']}")

        if not health["model_loaded"]:
            logger.error("\n❌ Model not loaded on server!")
            return

    except Exception:
        logger.exception("\n❌ Cannot connect to server")
        logger.info("Make sure the server is running!")
        return

    # Get model info
    try:
        info = client.get_model_info()
        logger.info(f"Model Type: {info['model_type']}")
        logger.info(f"Supported Labels: {', '.join(info['labels'][:10])}...")
    except Exception:
        logger.exception("Could not get model info")

    # Run examples
    try:
        example_single_detection(client)
        example_batch_detection(client)
        example_redaction(client)
        example_anonymization(client)
        example_filtering(client)

        logger.info("\n" + "=" * 80)
        logger.info("✅ All examples completed successfully!")
        logger.info("=" * 80 + "\n")

    except Exception:
        logger.exception("\n❌ Error running examples")


if __name__ == "__main__":
    main()
