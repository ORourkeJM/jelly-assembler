"""
Artifact Storage for Video Pipeline.

Persists all pipeline artifacts to R2 for use by assembler and refiner:
- Trimmed audio file
- Production brief (brain output)
- Music DNA
- Visual style guide
- Track context

All artifacts are stored under:
  R2/projects/{project_id}/

This enables:
1. Recovery after failures
2. Refiner access to full context
3. Audit trail of all pipeline outputs
"""

import os
import json
import tempfile
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, asdict
import boto3
from botocore.config import Config
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ArtifactManifest:
    """Manifest of all artifacts for a project."""
    project_id: str
    created_at: str
    trimmed_audio_url: Optional[str] = None
    production_brief_url: Optional[str] = None
    music_dna_url: Optional[str] = None
    visual_style_url: Optional[str] = None
    track_context_url: Optional[str] = None
    video_clips: list[str] = None  # URLs of generated video clips
    assembled_video_url: Optional[str] = None
    final_video_url: Optional[str] = None

    def __post_init__(self):
        if self.video_clips is None:
            self.video_clips = []


class ArtifactStore:
    """
    Persistent storage for pipeline artifacts.

    All artifacts are stored in R2 under projects/{project_id}/
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
            logger.info("Artifact store initialized", bucket=self.bucket_name)
        else:
            self.client = None
            logger.warning("Artifact store: R2 credentials not configured")

    def _upload_bytes(self, key: str, data: bytes, content_type: str) -> str:
        """Upload bytes to R2 and return public URL."""
        if not self.client:
            raise RuntimeError("R2 storage not configured")

        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=data,
            ContentType=content_type,
        )

        url = f"{self.public_url_base}/{key}"
        logger.info("Uploaded artifact", key=key)
        return url

    def _upload_file(self, key: str, file_path: str, content_type: str) -> str:
        """Upload file to R2 and return public URL."""
        if not self.client:
            raise RuntimeError("R2 storage not configured")

        with open(file_path, "rb") as f:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=f,
                ContentType=content_type,
            )

        url = f"{self.public_url_base}/{key}"
        logger.info("Uploaded artifact", key=key, file_path=file_path)
        return url

    def _upload_json(self, key: str, data: Any) -> str:
        """Upload JSON data to R2."""
        json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")
        return self._upload_bytes(key, json_bytes, "application/json")

    async def save_trimmed_audio(
        self,
        project_id: str,
        audio_file_path: str,
        audio_format: str = "mp3"
    ) -> str:
        """
        Save trimmed audio file to R2.

        Args:
            project_id: Unique project identifier
            audio_file_path: Path to the trimmed audio file
            audio_format: Audio format (mp3, wav, m4a)

        Returns:
            Public URL of the uploaded audio
        """
        key = f"projects/{project_id}/trimmed_audio.{audio_format}"
        content_type = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "m4a": "audio/mp4",
            "aac": "audio/aac",
        }.get(audio_format, "audio/mpeg")

        return self._upload_file(key, audio_file_path, content_type)

    async def save_production_brief(
        self,
        project_id: str,
        production_brief: dict
    ) -> str:
        """Save production brief (full brain output) to R2."""
        key = f"projects/{project_id}/production_brief.json"
        return self._upload_json(key, production_brief)

    async def save_music_dna(
        self,
        project_id: str,
        music_dna: dict
    ) -> str:
        """Save music DNA analysis to R2."""
        key = f"projects/{project_id}/music_dna.json"
        return self._upload_json(key, music_dna)

    async def save_visual_style(
        self,
        project_id: str,
        visual_style: dict
    ) -> str:
        """Save visual style guide to R2."""
        key = f"projects/{project_id}/visual_style.json"
        return self._upload_json(key, visual_style)

    async def save_track_context(
        self,
        project_id: str,
        track_context: dict
    ) -> str:
        """Save track context (analysis data) to R2."""
        key = f"projects/{project_id}/track_context.json"
        return self._upload_json(key, track_context)

    async def save_video_clip(
        self,
        project_id: str,
        clip_index: int,
        clip_file_path: str
    ) -> str:
        """Save a generated video clip to R2."""
        key = f"projects/{project_id}/clips/clip_{clip_index:03d}.mp4"
        return self._upload_file(key, clip_file_path, "video/mp4")

    async def save_manifest(
        self,
        manifest: ArtifactManifest
    ) -> str:
        """Save the artifact manifest to R2."""
        key = f"projects/{manifest.project_id}/manifest.json"
        return self._upload_json(key, asdict(manifest))

    async def get_manifest(self, project_id: str) -> Optional[ArtifactManifest]:
        """Retrieve the artifact manifest for a project."""
        if not self.client:
            return None

        key = f"projects/{project_id}/manifest.json"

        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
            data = json.loads(response["Body"].read().decode("utf-8"))
            return ArtifactManifest(**data)
        except self.client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.error(f"Failed to get manifest: {e}", project_id=project_id)
            return None

    async def save_all_artifacts(
        self,
        project_id: str,
        trimmed_audio_path: Optional[str] = None,
        production_brief: Optional[dict] = None,
        music_dna: Optional[dict] = None,
        visual_style: Optional[dict] = None,
        track_context: Optional[dict] = None,
    ) -> ArtifactManifest:
        """
        Save all pipeline artifacts and create manifest.

        This is the main entry point for saving all artifacts at once.

        Returns:
            ArtifactManifest with URLs to all saved artifacts
        """
        manifest = ArtifactManifest(
            project_id=project_id,
            created_at=datetime.utcnow().isoformat()
        )

        # Save each artifact if provided
        if trimmed_audio_path:
            ext = os.path.splitext(trimmed_audio_path)[1].lstrip(".") or "mp3"
            manifest.trimmed_audio_url = await self.save_trimmed_audio(
                project_id, trimmed_audio_path, ext
            )

        if production_brief:
            manifest.production_brief_url = await self.save_production_brief(
                project_id, production_brief
            )

        if music_dna:
            manifest.music_dna_url = await self.save_music_dna(
                project_id, music_dna
            )

        if visual_style:
            manifest.visual_style_url = await self.save_visual_style(
                project_id, visual_style
            )

        if track_context:
            manifest.track_context_url = await self.save_track_context(
                project_id, track_context
            )

        # Save the manifest itself
        await self.save_manifest(manifest)

        logger.info(
            "All artifacts saved",
            project_id=project_id,
            has_audio=bool(trimmed_audio_path),
            has_brief=bool(production_brief),
            has_dna=bool(music_dna),
            has_style=bool(visual_style),
            has_context=bool(track_context)
        )

        return manifest


# Global instance
_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    """Get or create the global artifact store instance."""
    global _artifact_store
    if _artifact_store is None:
        _artifact_store = ArtifactStore()
    return _artifact_store
