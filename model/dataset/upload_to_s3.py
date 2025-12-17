import argparse
import os

from tqdm import tqdm


def get_datastore_root() -> tuple[str, str]:
    """
    Get the datastore root path and cloud provider.

    Returns:
        Tuple of (root_path, provider) where provider is 's3', 'gs', 'azure', or 'local'
    """
    import metaflow.metaflow_config as config

    # Check cloud providers in order of preference
    if hasattr(config, 'DATASTORE_SYSROOT_S3') and config.DATASTORE_SYSROOT_S3:
        return config.DATASTORE_SYSROOT_S3, 's3'
    if hasattr(config, 'DATASTORE_SYSROOT_GS') and config.DATASTORE_SYSROOT_GS:
        return config.DATASTORE_SYSROOT_GS, 'gs'
    if hasattr(config, 'DATASTORE_SYSROOT_AZURE') and config.DATASTORE_SYSROOT_AZURE:
        return config.DATASTORE_SYSROOT_AZURE, 'azure'

    # Fallback to local
    local_root = getattr(config, 'DATASTORE_SYSROOT_LOCAL', '/tmp/metaflow')
    return local_root, 'local'

def upload_dir_to_s3(local_dir_path: str, s3_root: str, batch_size: int = 1000) -> None:
    """
    Upload a directory to S3 in batches.

    Args:
        local_dir_path: Path to the local directory
        s3_root: S3 path prefix
        batch_size: Number of files per batch (default 1000)
    """
    from metaflow import S3

    print(f"Uploading directory {local_dir_path} to {s3_root}/{local_dir_path}")

    # Collect all file paths, filtering to .json only
    key_paths = []
    for root, _, files in os.walk(local_dir_path):
        for file in files:
            if file.endswith(".json"):
                local_path = os.path.join(root, file)
                key_paths.append((os.path.join(s3_root, local_path), local_path))

    print(f"Found {len(key_paths)} JSON files to upload")

    # Upload in batches to avoid queue size limits
    with S3() as s3:
        for i in tqdm(range(0, len(key_paths), batch_size), desc="Uploading batches"):
            batch = key_paths[i:i + batch_size]
            s3.put_files(batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a directory to S3")
    parser.add_argument("--local-dir-path", type=str, required=True)
    ds_root = get_datastore_root()[0]
    parser.add_argument("--s3-root", type=str, default=ds_root + "/yaak-pii")
    args = parser.parse_args()
    upload_dir_to_s3(args.local_dir_path, args.s3_root)
