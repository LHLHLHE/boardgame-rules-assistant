import hashlib
import logging
import mimetypes
from collections.abc import Iterable

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError

from boardgame_rules_backend.settings import app_config

logger = logging.getLogger(__name__)

# Full wipe deletes everything under this prefix (sources + processed).
RULES_S3_PREFIX = "rules/"

RULES_SOURCES_PREFIX = "rules/sources/by-hash/"
RULES_PROCESSED_PREFIX = "rules/processed/by-hash/"


def get_s3_client() -> BaseClient:
    return boto3.client(
        "s3",
        endpoint_url=app_config.s3_endpoint_url,
        aws_access_key_id=app_config.aws_access_key_id,
        aws_secret_access_key=app_config.aws_secret_access_key,
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket_exists() -> None:
    client = get_s3_client()
    try:
        client.head_bucket(Bucket=app_config.s3_bucket)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            client.create_bucket(Bucket=app_config.s3_bucket)


def source_storage_key(doc_id: str, extension: str = "bin") -> str:
    """
    S3 key for uploaded originals (PDF/TXT).

    Content-addressed; shared across games when bytes match.
    """
    ext = extension.lstrip(".").lower() if extension else "bin"
    return f"{RULES_SOURCES_PREFIX}{doc_id}.{ext}"


def processed_rules_key(doc_id: str, extension: str = "txt") -> str:
    """S3 key for preprocessed rules text (and manifest texts). Content-addressed."""
    ext = extension.lstrip(".").lower() if extension else "txt"
    return f"{RULES_PROCESSED_PREFIX}{doc_id}.{ext}"


def source_content_type(filename: str) -> str:
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type:
        if mime_type.startswith("text/"):
            return f"{mime_type}; charset=utf-8"
        return mime_type
    return "application/octet-stream"


def upload_rules_file(content: bytes, filename: str) -> tuple[str, str]:
    """Upload rules file to S3 (content-addressed key)."""
    doc_id = hashlib.sha256(content).hexdigest()
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "txt"
    s3_key = processed_rules_key(doc_id, ext)

    client = get_s3_client()
    ensure_bucket_exists()
    client.put_object(
        Bucket=app_config.s3_bucket,
        Key=s3_key,
        Body=content,
        ContentType="text/plain; charset=utf-8",
    )
    return s3_key, doc_id


def upload_source_file(content: bytes, filename: str) -> tuple[str, str]:
    """Upload original source file to S3 using content-addressed key."""
    doc_id = hashlib.sha256(content).hexdigest()
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "bin"
    s3_key = source_storage_key(doc_id, ext)

    client = get_s3_client()
    ensure_bucket_exists()
    client.put_object(
        Bucket=app_config.s3_bucket,
        Key=s3_key,
        Body=content,
        ContentType=source_content_type(filename),
    )
    return s3_key, doc_id


def put_rules_content(s3_key: str, content: bytes) -> None:
    """Upload content to S3 at the given key. Used by background preprocessing task."""
    client = get_s3_client()
    ensure_bucket_exists()
    client.put_object(
        Bucket=app_config.s3_bucket,
        Key=s3_key,
        Body=content,
        ContentType="text/plain; charset=utf-8",
    )


def download_rules_file(s3_key: str) -> bytes:
    client = get_s3_client()
    response = client.get_object(Bucket=app_config.s3_bucket, Key=s3_key)
    return response["Body"].read()


def delete_s3_objects_best_effort(keys: Iterable[str]) -> None:
    """Remove objects from S3; log and ignore errors (compensating cleanup)."""
    keys_list = [k for k in keys if k]
    if not keys_list:
        return
    client = get_s3_client()
    bucket = app_config.s3_bucket
    for key in keys_list:
        try:
            client.delete_object(Bucket=bucket, Key=key)
        except ClientError as e:
            logger.warning("Failed to delete S3 object %s: %s", key, e)


def delete_all_objects_under_prefix_best_effort(prefix: str) -> None:
    """List and delete all objects with the given prefix (full wipe of rules tree)."""
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    client = get_s3_client()
    bucket = app_config.s3_bucket
    paginator = client.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj.get("Key")
                if not key:
                    continue
                try:
                    client.delete_object(Bucket=bucket, Key=key)
                except ClientError as e:
                    logger.warning("Failed to delete S3 object %s: %s", key, e)
    except ClientError as e:
        logger.warning("Failed to list S3 prefix %s: %s", prefix, e)
