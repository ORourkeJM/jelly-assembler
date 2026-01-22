"""
Storage handlers for video output.

Supports:
- Cloudflare R2 (default)
- Local file serving (dev mode)
"""

import os
import uuid
from datetime import datetime
from typing import Optional
import boto3
from botocore.config import Config
import structlog

logger = structlog.get_logger(__name__)


class StorageHandler:
    """Base storage handler interface."""

    async def upload(
        self,
        file_path: str,
        content_type: str = "video/mp4"
    ) -> str:
        """Upload file and return public URL."""
        raise NotImplementedError


class R2StorageHandler(StorageHandler):
    """
    Cloudflare R2 storage handler.

    Uses S3-compatible API with Cloudflare R2.
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        bucket_name: Optional[str] = None,
        public_url_base: Optional[str] = None
    ):
        self.account_id = account_id or os.environ.get("R2_ACCOUNT_ID")
        self.access_key_id = access_key_id or os.environ.get("R2_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.environ.get("R2_SECRET_ACCESS_KEY")
        self.bucket_name = bucket_name or os.environ.get("R2_BUCKET_NAME", "jellybop-videos")
        self.public_url_base = public_url_base or os.environ.get(
            "R2_PUBLIC_URL",
            f"https://{self.bucket_name}.r2.dev"
        )

        if self.account_id and self.access_key_id and self.secret_access_key:
            self.client = boto3.client(
                "s3",
                endpoint_url=f"https://{self.account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                config=Config(signature_version="s3v4"),
            )
            logger.info("R2 storage initialized", bucket=self.bucket_name)
        else:
            self.client = None
            logger.warning("R2 credentials not configured")

    async def upload(
        self,
        file_path: str,
        content_type: str = "video/mp4"
    ) -> str:
        """Upload file to R2 and return public URL."""
        if not self.client:
            raise RuntimeError("R2 storage not configured")

        # Generate unique key
        timestamp = datetime.utcnow().strftime("%Y/%m/%d")
        file_id = uuid.uuid4().hex[:12]
        extension = os.path.splitext(file_path)[1] or ".mp4"
        key = f"assembled/{timestamp}/{file_id}{extension}"

        # Upload file
        with open(file_path, "rb") as f:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=f,
                ContentType=content_type,
            )

        url = f"{self.public_url_base}/{key}"
        logger.info("Uploaded to R2", key=key, url=url)
        return url


class LocalStorageHandler(StorageHandler):
    """
    Local file storage for development.

    Serves files from a local directory.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.output_dir = output_dir or os.environ.get(
            "LOCAL_OUTPUT_DIR",
            "/tmp/jelly-assembler-output"
        )
        self.base_url = base_url or os.environ.get(
            "LOCAL_BASE_URL",
            "http://localhost:8080/files"
        )

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("Local storage initialized", dir=self.output_dir)

    async def upload(
        self,
        file_path: str,
        content_type: str = "video/mp4"
    ) -> str:
        """Copy file to output directory and return URL."""
        import shutil

        file_id = uuid.uuid4().hex[:12]
        extension = os.path.splitext(file_path)[1] or ".mp4"
        filename = f"{file_id}{extension}"
        output_path = os.path.join(self.output_dir, filename)

        shutil.copy2(file_path, output_path)

        url = f"{self.base_url}/{filename}"
        logger.info("Saved locally", path=output_path, url=url)
        return url


def get_storage_handler() -> StorageHandler:
    """Get the appropriate storage handler based on environment."""
    if os.environ.get("R2_ACCOUNT_ID"):
        return R2StorageHandler()
    else:
        return LocalStorageHandler()
