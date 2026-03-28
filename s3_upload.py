"""
S3 upload utility for ComfyUI load balancing worker.

Replaces the runpod.serverless.utils.rp_upload module that was used
in the queue-based handler. Reads the same environment variables:
  - BUCKET_ENDPOINT_URL
  - BUCKET_ACCESS_KEY_ID
  - BUCKET_SECRET_ACCESS_KEY
"""

import os
import logging
from urllib.parse import urlparse

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)


def _get_bucket_config():
    """Read S3 bucket configuration from environment variables."""
    endpoint_url = os.environ.get("BUCKET_ENDPOINT_URL", "")
    access_key = os.environ.get("BUCKET_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("BUCKET_SECRET_ACCESS_KEY", "")

    if not endpoint_url:
        raise ValueError("BUCKET_ENDPOINT_URL is not set")

    return {
        "endpoint_url": endpoint_url,
        "access_key": access_key,
        "secret_key": secret_key,
    }


def _get_s3_client(config):
    """Create an S3 client from the given configuration."""
    return boto3.client(
        "s3",
        endpoint_url=config["endpoint_url"],
        aws_access_key_id=config["access_key"],
        aws_secret_access_key=config["secret_key"],
        config=Config(signature_version="s3v4"),
    )


def _parse_bucket_name(endpoint_url):
    """
    Extract the bucket name from the endpoint URL.

    Supports common patterns:
      - https://BUCKET.s3.REGION.amazonaws.com
      - https://BUCKET.REGION.digitaloceanspaces.com
      - https://ACCOUNT_ID.r2.cloudflarestorage.com/BUCKET
    """
    parsed = urlparse(endpoint_url)
    host_parts = parsed.hostname.split(".")

    # Path-style: bucket name in the URL path (e.g. Cloudflare R2)
    if parsed.path and parsed.path.strip("/"):
        return parsed.path.strip("/").split("/")[0]

    # Virtual-hosted style: bucket name is first subdomain
    return host_parts[0]


def upload_to_s3(request_id, file_path):
    """
    Upload a file to S3 and return its public URL.

    Mirrors the behaviour of runpod's rp_upload.upload_image():
      - Uploads to {request_id}/{filename} within the bucket.
      - Returns the full URL to the uploaded object.

    Args:
        request_id (str): Unique ID for the request (used as key prefix).
        file_path (str): Absolute path to the local file to upload.

    Returns:
        str: The URL of the uploaded object.
    """
    config = _get_bucket_config()
    client = _get_s3_client(config)
    bucket_name = _parse_bucket_name(config["endpoint_url"])

    filename = os.path.basename(file_path)
    key = f"{request_id}/{filename}"

    logger.info("Uploading %s to s3://%s/%s", filename, bucket_name, key)
    client.upload_file(file_path, bucket_name, key)

    # Construct the public URL
    url = f"{config['endpoint_url'].rstrip('/')}/{key}"
    logger.info("Upload complete: %s", url)
    return url
