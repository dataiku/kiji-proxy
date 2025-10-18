"""
Download Latest PII Detection Model from Google Drive

This script downloads the most recent trained model from Google Drive
to use with the FastAPI server or for local evaluation.

Features:
- Automatic detection of latest model by timestamp
- Support for both Colab and local environments
- Multiple authentication methods (OAuth, service account, shared links)
- Progress tracking during download
- Automatic extraction of model files

Usage:
    # In Google Colab (auto-mount Drive):
    python download_latest_model.py

    # Local with file ID:
    python download_latest_model.py --file-id 1ABC123XYZ --output ./pii_model

    # Local with folder scanning:
    python download_latest_model.py --folder /path/to/gdrive/models --output ./pii_model

    # With credentials file:
    python download_latest_model.py --credentials credentials.json --folder-id ABC123

After downloading, test with curl:
    # Single text detection
    curl -X POST http://localhost:8000/detect \\
      -H "Content-Type: application/json" \\
      -d '{"text": "My email is john@example.com and phone is 555-1234"}'

    # Batch detection
    curl -X POST http://localhost:8000/detect/batch \\
      -H "Content-Type: application/json" \\
      -d '{"texts": ["Email: alice@test.com", "SSN: 123-45-6789"]}'
"""

import argparse
import logging
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    import requests
    from tqdm import tqdm
except ImportError:
    logger.exception("❌ Required packages not found. Install with:")
    logger.info("   pip install requests tqdm")
    sys.exit(1)


# =============================================================================
# GOOGLE DRIVE UTILITIES
# =============================================================================


class GoogleDriveDownloader:
    """Download files from Google Drive with multiple authentication methods."""

    # Google Drive download URL
    DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download"
    DRIVE_API_URL = "https://www.googleapis.com/drive/v3/files"

    def __init__(self, credentials_path: str | None = None):
        """
        Initialize downloader.

        Args:
            credentials_path: Path to Google service account credentials JSON
        """
        self.credentials_path = credentials_path
        self.access_token = None

        if credentials_path and Path(credentials_path).exists():
            self._authenticate()

    def _authenticate(self):
        """Authenticate using service account credentials."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account

            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=["https://www.googleapis.com/auth/drive.readonly"],
            )
            credentials.refresh(Request())
            self.access_token = credentials.token
            logger.info("✅ Authenticated with service account")
        except ImportError:
            logger.warning("⚠️  google-auth not installed. Using public download method.")
            logger.info("   Install with: pip install google-auth")
        except Exception as e:
            logger.warning(f"⚠️  Authentication failed: {e}")
            logger.info("   Falling back to public download method")

    def download_file(self, file_id: str, output_path: str, chunk_size: int = 32768) -> bool:
        """
        Download a file from Google Drive.

        Args:
            file_id: Google Drive file ID
            output_path: Local path to save the file
            chunk_size: Download chunk size in bytes

        Returns:
            True if successful, False otherwise
        """
        logger.info("\n📥 Downloading file from Google Drive...")
        logger.info(f"   File ID: {file_id}")
        logger.info(f"   Output: {output_path}")

        try:
            # Create output directory
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Start download session
            session = requests.Session()

            # Initial request
            response = session.get(self.DRIVE_DOWNLOAD_URL, params={"id": file_id}, stream=True)

            # Check for download warning (large files)
            token = self._get_confirm_token(response)
            if token:
                params = {"id": file_id, "confirm": token}
                response = session.get(self.DRIVE_DOWNLOAD_URL, params=params, stream=True)

            # Get file size if available
            file_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with Path(output_path).open("wb") as f:
                if file_size > 0:
                    with tqdm(total=file_size, unit="B", unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # No size info, download without progress bar
                    for chunk in response.iter_content(chunk_size):
                        if chunk:
                            f.write(chunk)

            logger.info(f"✅ Download complete: {output_path}")
            return True

        except Exception:
            logger.exception("❌ Download failed")
            return False

    @staticmethod
    def _get_confirm_token(response):
        """Extract confirmation token for large file downloads."""
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    def download_folder(self, folder_id: str, output_dir: str, recursive: bool = False) -> bool:
        """
        Download all files from a Google Drive folder.

        Args:
            folder_id: Google Drive folder ID
            output_dir: Local directory to save files
            recursive: Download subfolders recursively

        Returns:
            True if successful, False otherwise
        """
        if not self.access_token:
            logger.error("❌ Folder download requires authentication")
            logger.info("   Provide credentials file with --credentials")
            return False

        logger.info("\n📂 Downloading folder from Google Drive...")
        logger.info(f"   Folder ID: {folder_id}")
        logger.info(f"   Output: {output_dir}")

        try:
            # List files in folder
            headers = {"Authorization": f"Bearer {self.access_token}"}
            params = {
                "q": f"'{folder_id}' in parents",
                "fields": "files(id, name, mimeType, size)",
                "pageSize": 1000,
            }

            response = requests.get(self.DRIVE_API_URL, headers=headers, params=params)
            response.raise_for_status()

            files = response.json().get("files", [])

            if not files:
                logger.warning("⚠️  No files found in folder")
                return False

            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Download each file
            for file_info in files:
                file_name = file_info["name"]
                file_id = file_info["id"]
                mime_type = file_info.get("mimeType", "")

                if "folder" in mime_type and recursive:
                    # Recursive folder download
                    subfolder_path = Path(output_dir) / file_name
                    self.download_folder(file_id, subfolder_path, recursive=True)
                else:
                    # Download file
                    file_path = Path(output_dir) / file_name
                    self.download_file(file_id, file_path)

            logger.info("✅ Folder download complete")
            return True

        except Exception:
            logger.exception("❌ Folder download failed")
            return False


class ColabDriveManager:
    """Manage Google Drive in Colab environment."""

    @staticmethod
    def is_colab() -> bool:
        """Check if running in Google Colab."""
        try:
            import google.colab  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def mount_drive(mount_point: str = "/content/drive") -> bool:
        """
        Mount Google Drive in Colab.

        Args:
            mount_point: Mount point for Google Drive

        Returns:
            True if mounted successfully
        """
        try:
            from google.colab import drive

            drive.mount(mount_point)
            logger.info(f"✅ Google Drive mounted at {mount_point}")
            return True
        except Exception:
            logger.exception("❌ Failed to mount Drive")
            return False

    @staticmethod
    def find_latest_model(drive_folder: str = "MyDrive/pii_models") -> str | None:
        """
        Find the most recent model in Google Drive folder.

        Args:
            drive_folder: Folder path relative to Drive mount

        Returns:
            Path to latest model, or None if not found
        """
        drive_path = f"/content/drive/{drive_folder}"

        if not Path(drive_path).exists():
            logger.error(f"❌ Folder not found: {drive_path}")
            return None

        # List all directories
        model_dirs = [d.name for d in Path(drive_path).iterdir() if d.is_dir()]

        if not model_dirs:
            logger.error(f"❌ No models found in {drive_path}")
            return None

        # Parse timestamps
        model_timestamps = []
        for model_dir in model_dirs:
            match = re.search(r"_(\d{8}_\d{6})$", model_dir)
            if match:
                timestamp_str = match.group(1)
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    model_timestamps.append((timestamp, model_dir))
                except ValueError:
                    continue

        if not model_timestamps:
            logger.warning(f"⚠️  No timestamped models, using first: {model_dirs[0]}")
            return str(Path(drive_path) / model_dirs[0])

        # Get most recent
        model_timestamps.sort(reverse=True)
        latest_model = model_timestamps[0][1]
        latest_path = str(Path(drive_path) / latest_model)

        logger.info(f"✅ Found latest model: {latest_model}")
        return latest_path

    @staticmethod
    def copy_model(source_path: str, dest_path: str) -> bool:
        """
        Copy model from Drive to local directory.

        Args:
            source_path: Source path in Drive
            dest_path: Destination path

        Returns:
            True if successful
        """
        try:
            logger.info("\n📋 Copying model...")
            logger.info(f"   From: {source_path}")
            logger.info(f"   To: {dest_path}")

            if Path(dest_path).exists():
                logger.warning(f"⚠️  Destination exists, removing: {dest_path}")
                shutil.rmtree(dest_path)

            shutil.copytree(source_path, dest_path)
            logger.info("✅ Model copied successfully")
            return True

        except Exception:
            logger.exception("❌ Copy failed")
            return False


# =============================================================================
# EXTRACTION UTILITIES
# =============================================================================


def extract_zip(zip_path: str, extract_to: str) -> bool:
    """
    Extract a ZIP file.

    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to

    Returns:
        True if successful
    """
    try:
        logger.info("\n📦 Extracting ZIP file...")
        logger.info(f"   From: {zip_path}")
        logger.info(f"   To: {extract_to}")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        logger.info("✅ Extraction complete")
        return True

    except Exception:
        logger.exception("❌ Extraction failed")
        return False


def verify_model(model_path: str) -> bool:
    """
    Verify that model directory contains required files.

    Args:
        model_path: Path to model directory

    Returns:
        True if model is valid
    """
    required_files = ["config.json", "tokenizer_config.json", "vocab.txt"]

    # Check for model weights (safetensors or pytorch_model.bin)
    model_path_obj = Path(model_path)
    has_weights = (model_path_obj / "model.safetensors").exists() or (
        model_path_obj / "pytorch_model.bin"
    ).exists()

    if not has_weights:
        logger.error("❌ Model weights not found (model.safetensors or pytorch_model.bin)")
        return False

    # Check required files
    missing = [file for file in required_files if not (model_path_obj / file).exists()]

    if missing:
        logger.error(f"❌ Missing required files: {', '.join(missing)}")
        return False

    logger.info("✅ Model verification passed")
    return True


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download latest PII detection model from Google Drive"
    )

    # Download methods
    parser.add_argument("--file-id", type=str, help="Google Drive file ID for direct download")
    parser.add_argument(
        "--folder-id", type=str, help="Google Drive folder ID to download all files"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Local path to folder containing models (for Colab or mounted Drive)",
    )
    parser.add_argument(
        "--drive-folder",
        type=str,
        default="MyDrive/pii_models",
        help="Google Drive folder path in Colab (default: MyDrive/pii_models)",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="../pii_model",
        help="Output directory for model (default: ../pii_model)",
    )

    # Authentication
    parser.add_argument(
        "--credentials",
        type=str,
        help="Path to Google service account credentials JSON",
    )

    # Options
    parser.add_argument(
        "--extract", action="store_true", help="Extract if downloaded file is a ZIP"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify model after download (default: True)",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PII Detection Model - Download Latest Version")
    logger.info("=" * 80)

    # Check if running in Colab
    is_colab = ColabDriveManager.is_colab()

    if is_colab:
        logger.info("\n🔍 Detected Google Colab environment")

        # Mount Drive
        if not ColabDriveManager.mount_drive():
            logger.error("❌ Failed to mount Google Drive")
            return 1

        # Find latest model
        source_path = ColabDriveManager.find_latest_model(args.drive_folder)
        if not source_path:
            logger.error("❌ No model found in Drive")
            return 1

        # Copy to local
        if not ColabDriveManager.copy_model(source_path, args.output):
            logger.error("❌ Failed to copy model")
            return 1

    elif args.file_id:
        logger.info("\n🔍 Direct file download mode")

        # Initialize downloader
        downloader = GoogleDriveDownloader(args.credentials)

        # Determine output path
        if args.extract:
            temp_zip = f"{args.output}.zip"
            if not downloader.download_file(args.file_id, temp_zip):
                return 1

            if not extract_zip(temp_zip, args.output):
                return 1

            # Clean up ZIP
            Path(temp_zip).unlink()
        # Download directly to output
        elif not downloader.download_file(args.file_id, args.output):
            return 1

    elif args.folder_id:
        logger.info("\n🔍 Folder download mode")

        # Initialize downloader with auth
        downloader = GoogleDriveDownloader(args.credentials)

        if not downloader.download_folder(args.folder_id, args.output):
            return 1

    elif args.folder:
        logger.info("\n🔍 Local folder mode")

        # Find latest in local folder
        if not Path(args.folder).exists():
            logger.error(f"❌ Folder not found: {args.folder}")
            return 1

        # Copy the folder
        if not ColabDriveManager.copy_model(args.folder, args.output):
            return 1

    else:
        logger.error("\n❌ No download method specified!")
        logger.info("\nPlease provide one of:")
        logger.info("  --file-id FILE_ID        Direct file download")
        logger.info("  --folder-id FOLDER_ID    Download entire folder")
        logger.info("  --folder PATH            Copy from local/mounted folder")
        logger.info("\nIn Colab, Drive will be mounted automatically")
        logger.info("\nFor more info: python download_latest_model.py --help")
        return 1

    # Verify model
    if args.verify:
        logger.info("\n🔍 Verifying model...")
        if not verify_model(args.output):
            logger.warning("⚠️  Model verification failed, but files were downloaded")
            logger.info("   The model may still work depending on the error")

    # Success
    logger.info("\n" + "=" * 80)
    logger.info("✅ Download Complete!")
    logger.info("=" * 80)
    logger.info(f"Model location: {Path(args.output).resolve()}")
    logger.info("\nNext steps:")
    logger.info("  1. Start the FastAPI server:")
    logger.info(f"     cd model_server && MODEL_PATH={args.output} ./start_server.sh")
    logger.info("  2. Or run evaluation:")
    logger.info(f"     python model/eval_model.py --local-model {args.output}")
    logger.info("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
