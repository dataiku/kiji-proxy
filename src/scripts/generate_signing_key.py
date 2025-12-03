#!/usr/bin/env python3
"""
Utility script for generating cryptographic signing keys for model signing.

This script generates EC P-256 key pairs that can be used with the model-signing
library for signing models in CI environments where keyless signing isn't available.
"""

import argparse
import os
from pathlib import Path

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Error: cryptography not installed. Run 'pip install cryptography'")


def generate_key_pair(
    private_key_path: str, public_key_path: str, password: str = None
):
    """
    Generate an EC P-256 key pair for model signing.

    Args:
        private_key_path: Path to save the private key
        public_key_path: Path to save the public key
        password: Optional password to encrypt the private key
    """
    if not CRYPTO_AVAILABLE:
        raise ImportError("cryptography library not available")

    # Generate private key
    private_key = ec.generate_private_key(ec.SECP256R1())

    # Get public key
    public_key = private_key.public_key()

    # Prepare encryption for private key
    encryption_algorithm = serialization.NoEncryption()
    if password:
        encryption_algorithm = serialization.BestAvailableEncryption(password.encode())

    # Serialize private key
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption_algorithm,
    )

    # Serialize public key
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    # Create directories if they don't exist
    Path(private_key_path).parent.mkdir(parents=True, exist_ok=True)
    Path(public_key_path).parent.mkdir(parents=True, exist_ok=True)

    # Write keys to files
    with open(private_key_path, "wb") as f:
        f.write(private_pem)

    with open(public_key_path, "wb") as f:
        f.write(public_pem)

    # Set secure permissions on private key
    os.chmod(private_key_path, 0o600)

    print(f"✓ Private key saved to: {private_key_path}")
    print(f"✓ Public key saved to: {public_key_path}")
    print(f"✓ Private key permissions set to 600 (owner read/write only)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate cryptographic key pair for model signing"
    )
    parser.add_argument(
        "--private-key",
        "-p",
        default="keys/signing_key.pem",
        help="Path for private key file (default: keys/signing_key.pem)",
    )
    parser.add_argument(
        "--public-key",
        "-u",
        default="keys/signing_key.pub",
        help="Path for public key file (default: keys/signing_key.pub)",
    )
    parser.add_argument(
        "--password",
        "-w",
        help="Password to encrypt private key (leave empty for no encryption)",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing keys without confirmation",
    )

    args = parser.parse_args()

    # Check if keys already exist
    if Path(args.private_key).exists() or Path(args.public_key).exists():
        if not args.force:
            response = input(
                f"Keys already exist at {args.private_key} or {args.public_key}. "
                "Overwrite? (y/N): "
            )
            if response.lower() not in ["y", "yes"]:
                print("Aborted.")
                return

    try:
        generate_key_pair(args.private_key, args.public_key, args.password)

        print("\n" + "=" * 50)
        print("KEY GENERATION COMPLETE")
        print("=" * 50)
        print("\nNext steps:")
        print(f"1. Keep your private key ({args.private_key}) secure and secret")
        print(f"2. Store the private key as a CI secret (e.g., SIGNING_PRIVATE_KEY)")
        print(f"3. Use the public key ({args.public_key}) for signature verification")
        print("\nExample CI usage:")
        print(f"  echo '${{secrets.SIGNING_PRIVATE_KEY}}' > {args.private_key}")
        print(f"  python model_signing.py --private-key {args.private_key}")

    except Exception as e:
        print(f"Error generating keys: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
