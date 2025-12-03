"""Model signing utilities for ensuring model integrity and provenance."""

import hashlib
import json
from pathlib import Path
from typing import Optional

try:
    from model_signing import signing

    SIGNING_AVAILABLE = True
except ImportError:
    SIGNING_AVAILABLE = False
    print(
        "Warning: model-signing not installed. Run 'pip install model-signing' to enable."
    )


class ModelSigner:
    """Handle model signing and hash generation."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

    def sign_model(
        self, private_key_path: Optional[str] = None, output_path: Optional[str] = None
    ):
        """
        Sign the model using either Sigstore (keyless) or a private key.

        Args:
            private_key_path: Path to private key file (optional, uses Sigstore if not provided)
            output_path: Path for signature file (default: model_path.sig)

        Returns:
            Path to signature file
        """
        if not SIGNING_AVAILABLE:
            raise ImportError("model-signing library not available")

        output_path = output_path or f"{self.model_path}.sig"

        if private_key_path:
            # Use private key signing
            config = signing.Config().use_elliptic_key_signer(
                private_key=private_key_path
            )
        else:
            # Use Sigstore (keyless signing)
            config = signing.Config().use_sigstore_signer()

        config.sign(str(self.model_path), output_path)
        return output_path

    def verify_signature(self, signature_path: str) -> bool:
        """
        Verify the model signature.

        Args:
            signature_path: Path to signature file

        Returns:
            True if signature is valid
        """
        if not SIGNING_AVAILABLE:
            raise ImportError("model-signing library not available")

        try:
            verifier = signing.Verifier()
            verifier.verify(str(self.model_path), signature_path)
            return True
        except Exception as e:
            print(f"Signature verification failed: {e}")
            return False

    def compute_model_hash(self, algorithm: str = "sha256") -> str:
        """
        Compute cryptographic hash of the model directory.

        Args:
            algorithm: Hash algorithm to use (sha256, sha512, blake2b)

        Returns:
            Hexadecimal hash digest
        """
        hasher = hashlib.new(algorithm)

        # Sort files for deterministic hashing
        model_files = sorted(self.model_path.rglob("*"))

        for file_path in model_files:
            if file_path.is_file():
                # Include file path in hash for structure integrity
                hasher.update(str(file_path.relative_to(self.model_path)).encode())

                # Hash file contents
                with open(file_path, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)

        return hasher.hexdigest()

    def generate_model_manifest(self, output_path: Optional[str] = None) -> dict:
        """
        Generate a manifest with model metadata and hashes.

        Args:
            output_path: Path to save manifest JSON (optional)

        Returns:
            Manifest dictionary
        """
        manifest = {
            "model_path": str(self.model_path),
            "hashes": {
                "sha256": self.compute_model_hash("sha256"),
                "sha512": self.compute_model_hash("sha512"),
            },
            "files": [],
        }

        # Add individual file hashes
        for file_path in sorted(self.model_path.rglob("*")):
            if file_path.is_file():
                with open(file_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()

                manifest["files"].append(
                    {
                        "path": str(file_path.relative_to(self.model_path)),
                        "sha256": file_hash,
                    }
                )

        if output_path:
            with open(output_path, "w") as f:
                json.dump(manifest, f, indent=2)

        return manifest


def sign_trained_model(model_dir: str = "model/quantized"):
    """
    Sign the trained model and generate manifest.

    Args:
        model_dir: Directory containing the trained model
    """
    signer = ModelSigner(model_dir)

    # Generate model hash
    model_hash = signer.compute_model_hash()
    print(f"Model SHA-256 Hash: {model_hash}")

    # Generate manifest
    manifest = signer.generate_model_manifest(f"{model_dir}/model_manifest.json")
    print(f"Generated manifest with {len(manifest['files'])} files")

    # Sign model (using Sigstore by default)
    try:
        sig_path = signer.sign_model()
        print(f"Model signed successfully: {sig_path}")
    except Exception as e:
        print(f"Warning: Model signing failed: {e}")
        print("Model hash is still available for verification")

    return model_hash


if __name__ == "__main__":
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else "model/quantized"
    sign_trained_model(model_dir)
