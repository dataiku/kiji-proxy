"""Model signing utilities for ensuring model integrity and provenance."""

import hashlib
import json
import os
from pathlib import Path

try:
    from model_signing import hashing, signing

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
        self,
        private_key_path: str | None = None,
        output_path: str | None = None,
        use_ci_mode: bool = False,
    ):
        """
        Sign the model using either Sigstore (keyless) or a private key.

        Args:
            private_key_path: Path to private key file (optional, uses Sigstore if not provided)
            output_path: Path for signature file (default: model_path.sig)
            use_ci_mode: Use CI-friendly signing with OIDC token

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
            if use_ci_mode:
                # CI mode: use OIDC token from environment
                config = self._get_ci_signing_config()
            else:
                # Interactive mode: use browser-based authentication
                config = signing.Config().use_sigstore_signer()

        # Allow symlinks (needed for Metaflow model artifacts which use symlinks)
        hashing_config = hashing.Config().set_allow_symlinks(True)
        config.set_hashing_config(hashing_config)

        config.sign(str(self.model_path), output_path)
        return output_path

    def _get_ci_signing_config(self):
        """
        Get signing configuration for CI environments using OIDC tokens.

        Returns:
            Signing configuration for CI
        """
        # Check for GitHub Actions OIDC token
        if os.getenv("GITHUB_ACTIONS"):
            oidc_token = os.getenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
            oidc_url = os.getenv("ACTIONS_ID_TOKEN_REQUEST_URL")

            if oidc_token and oidc_url:
                # Use GitHub Actions OIDC token
                config = signing.Config().use_sigstore_signer()
                # Set environment variables for sigstore-python
                os.environ["SIGSTORE_ID_TOKEN"] = oidc_token
                return config

        # Check for GitLab CI OIDC token
        elif os.getenv("GITLAB_CI"):
            oidc_token = os.getenv("CI_JOB_JWT")

            if oidc_token:
                config = signing.Config().use_sigstore_signer()
                os.environ["SIGSTORE_ID_TOKEN"] = oidc_token
                return config

        # Fallback to ambient credentials or stored token
        print(
            "Warning: No CI OIDC token found, falling back to default Sigstore config"
        )
        return signing.Config().use_sigstore_signer()

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
            # Resolve symlinks to get the actual file
            resolved_path = file_path.resolve()
            if resolved_path.is_file():
                # Include file path in hash for structure integrity
                hasher.update(str(file_path.relative_to(self.model_path)).encode())

                # Hash file contents (follow symlinks)
                with open(resolved_path, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)

        return hasher.hexdigest()

    def generate_model_manifest(self, output_path: str | None = None) -> dict:
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
            # Resolve symlinks to get the actual file
            resolved_path = file_path.resolve()
            if resolved_path.is_file():
                with open(resolved_path, "rb") as f:
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


def sign_trained_model(
    model_dir: str = "model/quantized",
    ci_mode: bool = None,
    private_key_path: str = None,
):
    """
    Sign the trained model and generate manifest.

    Args:
        model_dir: Directory containing the trained model
        ci_mode: Force CI mode (auto-detected if None)
        private_key_path: Path to private key file (if using key-based signing)
                         Can also be set via MODEL_SIGNING_KEY_PATH environment variable
    """
    signer = ModelSigner(model_dir)

    # Auto-detect private key from environment if not provided
    if private_key_path is None:
        private_key_path = os.getenv("MODEL_SIGNING_KEY_PATH")

    # Auto-detect CI environment if not specified
    if ci_mode is None:
        ci_mode = bool(
            os.getenv("CI") or os.getenv("GITHUB_ACTIONS") or os.getenv("GITLAB_CI")
        )

    # Generate model hash
    model_hash = signer.compute_model_hash()
    print(f"Model SHA-256 Hash: {model_hash}")

    # Generate manifest
    manifest = signer.generate_model_manifest(f"{model_dir}/model_manifest.json")
    print(f"Generated manifest with {len(manifest['files'])} files")

    # Sign model (using appropriate mode)
    try:
        if private_key_path:
            print(f"Using private key signing from: {private_key_path}")
            sig_path = signer.sign_model(private_key_path=private_key_path)
            print(f"Model signed successfully: {sig_path}")
        elif ci_mode:
            print("Using CI mode for model signing...")
            sig_path = signer.sign_model(use_ci_mode=ci_mode)
            print(f"Model signed successfully: {sig_path}")
        else:
            print("No private key or CI mode available, skipping signature generation")
            print("Model hash is still available for verification")
    except Exception as e:
        print(f"Warning: Model signing failed: {e}")
        print("Model hash is still available for verification")

    return model_hash


if __name__ == "__main__":
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else "model/quantized"
    sign_trained_model(model_dir)
