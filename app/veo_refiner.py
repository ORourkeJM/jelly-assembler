"""
Google Veo Video Refiner - Uses Vertex AI Veo 3.1 for video enhancement.

Features:
1. Frames-to-Video: Generate smooth transitions using first/last frames
2. Video Extension: Extend clips by 7 seconds
3. Reference Images: Maintain character/style consistency across clips

Pricing: $0.10-$0.50/second depending on model
"""

import os
import json
import base64
import asyncio
from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum
import httpx
import structlog

logger = structlog.get_logger(__name__)


class VeoModel(str, Enum):
    """Available Veo models."""
    VEO_31_FAST = "veo-3.1-fast-generate-001"  # Faster, cheaper ($0.10/s)
    VEO_31 = "veo-3.1-generate-001"            # Better quality ($0.40/s)
    VEO_30 = "veo-3.0-generate-001"            # Standard ($0.50/s)


@dataclass
class VeoRequest:
    """Request for Veo video generation/refinement."""
    prompt: str
    first_frame_url: Optional[str] = None      # Starting frame image
    last_frame_url: Optional[str] = None       # Ending frame image
    video_to_extend_url: Optional[str] = None  # Video to extend
    reference_image_urls: Optional[list[str]] = None  # Character/style references
    duration_seconds: int = 8                   # 4, 6, or 8 seconds
    aspect_ratio: str = "9:16"                 # 9:16 or 16:9
    model: VeoModel = VeoModel.VEO_31_FAST
    seed: Optional[int] = None


@dataclass
class VeoResult:
    """Result from Veo video generation."""
    success: bool
    video_url: Optional[str] = None
    video_base64: Optional[str] = None  # If not using GCS storage
    duration: Optional[float] = None
    processing_time_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None


class VeoRefiner:
    """
    Google Veo video refiner via Vertex AI.

    Uses Veo 3.1 for:
    - Generating videos from first/last frames (smooth transitions)
    - Extending existing videos
    - Maintaining consistency with reference images
    """

    # Cost per second by model
    COSTS = {
        VeoModel.VEO_31_FAST: 0.10,
        VeoModel.VEO_31: 0.40,
        VeoModel.VEO_30: 0.50,
    }

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        storage_bucket: Optional[str] = None
    ):
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.storage_bucket = storage_bucket or os.environ.get("GCS_BUCKET")

        if not self.project_id:
            logger.warning("GOOGLE_CLOUD_PROJECT not configured - Veo disabled")

    @property
    def _base_url(self) -> str:
        return f"https://{self.location}-aiplatform.googleapis.com/v1"

    async def _get_access_token(self) -> Optional[str]:
        """Get Google Cloud access token."""
        # Try JSON credentials from environment first
        json_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        logger.info(f"GOOGLE_APPLICATION_CREDENTIALS_JSON present: {bool(json_creds)}")

        if json_creds:
            try:
                import json
                import base64
                from google.oauth2 import service_account
                from google.auth.transport.requests import Request

                logger.info(f"JSON creds length: {len(json_creds)}")

                # Try base64 decode first (preferred method)
                try:
                    decoded = base64.b64decode(json_creds).decode('utf-8')
                    creds_dict = json.loads(decoded)
                    logger.info("Decoded from base64")
                except Exception:
                    # Fall back to raw JSON
                    creds_dict = json.loads(json_creds)
                    logger.info("Parsed raw JSON")
                logger.info(f"Parsed JSON, project_id: {creds_dict.get('project_id')}")

                credentials = service_account.Credentials.from_service_account_info(
                    creds_dict,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                logger.info("Created credentials object, refreshing...")
                credentials.refresh(Request())
                logger.info("Token obtained successfully")
                return credentials.token
            except Exception as e:
                logger.error(f"Failed to use JSON credentials: {e}", exc_info=True)

        # Try using gcloud CLI
        try:
            proc = await asyncio.create_subprocess_exec(
                "gcloud", "auth", "print-access-token",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                return stdout.decode().strip()
        except Exception:
            pass

        # Try using Application Default Credentials
        try:
            from google.auth import default
            from google.auth.transport.requests import Request

            credentials, _ = default()
            credentials.refresh(Request())
            return credentials.token
        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            return None

    async def _download_image_as_base64(self, url: str) -> Optional[str]:
        """Download image and convert to base64."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                return base64.b64encode(response.content).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to download image: {e}")
            return None

    async def generate(self, request: VeoRequest) -> VeoResult:
        """
        Generate or refine video using Veo.

        Supports:
        - Text-to-video: Just provide prompt
        - First frame to video: Provide first_frame_url
        - Frames to video: Provide first_frame_url and last_frame_url
        - Extend video: Provide video_to_extend_url
        """
        if not self.project_id:
            return VeoResult(
                success=False,
                error="GOOGLE_CLOUD_PROJECT not configured"
            )

        import time
        start_time = time.time()

        try:
            access_token = await self._get_access_token()
            if not access_token:
                return VeoResult(
                    success=False,
                    error="Failed to get Google Cloud access token"
                )

            # Build request payload
            instance: dict[str, Any] = {"prompt": request.prompt}

            # Add first frame if provided
            if request.first_frame_url:
                image_b64 = await self._download_image_as_base64(request.first_frame_url)
                if image_b64:
                    instance["image"] = {
                        "bytesBase64Encoded": image_b64,
                        "mimeType": "image/png"
                    }

            # Add last frame if provided
            if request.last_frame_url:
                last_b64 = await self._download_image_as_base64(request.last_frame_url)
                if last_b64:
                    instance["lastFrame"] = {
                        "bytesBase64Encoded": last_b64,
                        "mimeType": "image/png"
                    }

            # Add video to extend if provided
            if request.video_to_extend_url:
                # Video must be downloaded and converted
                # For now, require GCS URL
                if request.video_to_extend_url.startswith("gs://"):
                    instance["video"] = {"gcsUri": request.video_to_extend_url}

            # Add reference images for consistency
            if request.reference_image_urls:
                ref_images = []
                for url in request.reference_image_urls[:4]:  # Max 4 references
                    ref_b64 = await self._download_image_as_base64(url)
                    if ref_b64:
                        ref_images.append({
                            "referenceImage": {
                                "bytesBase64Encoded": ref_b64,
                                "mimeType": "image/png"
                            },
                            "referenceType": "STYLE"
                        })
                if ref_images:
                    instance["referenceImages"] = ref_images

            # Build parameters
            parameters: dict[str, Any] = {
                "sampleCount": 1,
                "durationSeconds": request.duration_seconds,
                "aspectRatio": request.aspect_ratio,
            }

            if request.seed is not None:
                parameters["seed"] = request.seed

            # Add storage URI if bucket configured
            if self.storage_bucket:
                parameters["storageUri"] = f"gs://{self.storage_bucket}/veo-outputs/"

            payload = {
                "instances": [instance],
                "parameters": parameters
            }

            # Submit request
            model_id = request.model.value
            url = f"{self._base_url}/projects/{self.project_id}/locations/{self.location}/publishers/google/models/{model_id}:predictLongRunning"

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()

            operation_name = result.get("name")
            if not operation_name:
                return VeoResult(
                    success=False,
                    error="No operation name in response"
                )

            # Poll for completion
            video_result = await self._poll_operation(operation_name, access_token)

            processing_time = (time.time() - start_time) * 1000
            cost = request.duration_seconds * self.COSTS.get(request.model, 0.40)

            if video_result:
                return VeoResult(
                    success=True,
                    video_url=video_result.get("url"),
                    video_base64=video_result.get("base64"),
                    duration=request.duration_seconds,
                    processing_time_ms=processing_time,
                    cost_usd=cost
                )
            else:
                return VeoResult(
                    success=False,
                    error="Operation completed but no video returned",
                    processing_time_ms=processing_time
                )

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if e.response else str(e)
            return VeoResult(
                success=False,
                error=f"HTTP {e.response.status_code}: {error_detail}",
                processing_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            logger.error(f"Veo generation failed: {e}")
            return VeoResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )

    async def _poll_operation(
        self,
        operation_name: str,
        access_token: str,
        timeout_seconds: float = 300.0
    ) -> Optional[dict]:
        """Poll operation until complete."""
        poll_url = f"{self._base_url}/{operation_name}"
        headers = {"Authorization": f"Bearer {access_token}"}

        start_time = asyncio.get_event_loop().time()

        async with httpx.AsyncClient(timeout=30.0) as client:
            while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
                response = await client.get(poll_url, headers=headers)
                result = response.json()

                if result.get("done"):
                    # Check for error
                    if "error" in result:
                        logger.error(f"Veo operation failed: {result['error']}")
                        return None

                    # Extract video from response
                    response_data = result.get("response", {})
                    videos = response_data.get("generatedSamples", [])

                    if videos:
                        video = videos[0].get("video", {})
                        return {
                            "url": video.get("gcsUri"),
                            "base64": video.get("bytesBase64Encoded")
                        }

                    return None

                await asyncio.sleep(5)

        logger.warning("Veo operation timed out")
        return None

    async def create_transition(
        self,
        first_frame_url: str,
        last_frame_url: str,
        prompt: str,
        duration: int = 4
    ) -> VeoResult:
        """
        Create a smooth transition between two frames.

        Useful for creating transitions between video clips.
        """
        return await self.generate(VeoRequest(
            prompt=f"Smooth cinematic transition. {prompt}",
            first_frame_url=first_frame_url,
            last_frame_url=last_frame_url,
            duration_seconds=duration,
            model=VeoModel.VEO_31_FAST
        ))

    async def extend_clip(
        self,
        video_gcs_url: str,
        prompt: str,
        reference_images: Optional[list[str]] = None
    ) -> VeoResult:
        """
        Extend an existing video by 7 seconds.

        Video must be on GCS.
        """
        return await self.generate(VeoRequest(
            prompt=prompt,
            video_to_extend_url=video_gcs_url,
            reference_image_urls=reference_images,
            duration_seconds=8,  # Extension is always ~7s
            model=VeoModel.VEO_31
        ))

    async def refine_with_style(
        self,
        prompt: str,
        first_frame_url: str,
        style_reference_urls: list[str],
        duration: int = 8
    ) -> VeoResult:
        """
        Generate video with style references for consistency.

        Uses reference images to maintain visual style.
        """
        return await self.generate(VeoRequest(
            prompt=prompt,
            first_frame_url=first_frame_url,
            reference_image_urls=style_reference_urls,
            duration_seconds=duration,
            model=VeoModel.VEO_31
        ))


# Global instance
_veo_refiner: Optional[VeoRefiner] = None


def get_veo_refiner() -> VeoRefiner:
    """Get or create the global Veo refiner instance."""
    global _veo_refiner
    if _veo_refiner is None:
        _veo_refiner = VeoRefiner()
    return _veo_refiner
