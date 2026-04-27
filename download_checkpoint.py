#!/usr/bin/env python3
"""Script to download checkpoint from Google Cloud Storage."""

import os
from google.cloud import storage

# GCS path
GCS_BUCKET = "openpi-assets"
GCS_PREFIX = "checkpoints/pi05_base"
LOCAL_DIR = "checkpoints/pi05_base"

def download_from_gcs(bucket_name, prefix, local_dir):
    """Download files from GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Create local directory
    os.makedirs(local_dir, exist_ok=True)

    # List and download all blobs matching the prefix
    blobs = bucket.list_blobs(prefix=prefix)

    count = 0
    for blob in blobs:
        # Skip if it's a directory marker (ends with /)
        if blob.name.endswith('/'):
            continue

        # Get relative path
        relative_path = blob.name[len(prefix):].lstrip('/')
        local_file = os.path.join(local_dir, relative_path)

        # Create subdirectories if needed
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        # Download file
        print(f"Downloading {blob.name} -> {local_file}")
        blob.download_to_filename(local_file)
        count += 1

    print(f"\nDownload complete! {count} files downloaded to {local_dir}")

if __name__ == "__main__":
    print(f"Starting download from gs://{GCS_BUCKET}/{GCS_PREFIX}")
    print(f"Destination: {LOCAL_DIR}\n")

    try:
        download_from_gcs(GCS_BUCKET, GCS_PREFIX, LOCAL_DIR)
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure you have GCS credentials configured.")
        print("You can use: gcloud auth application-default login")
