"""
Video Refiner - AI-powered video polishing and consistency enhancement.

Uses fal.ai Kling O1 models for:
1. Edit mode - Polish/fix specific elements in generated clips
2. Reference mode - Maintain consistency across multiple clips

Pricing: $0.168/second
"""

import os
import asyncio
from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum
import httpx
import structlog

logger = structlog.get_logger(__name__)


class RefinementMode(str, Enum):
    """Refinement modes available."""
    EDIT = "edit"           # Polish/fix specific elements
    CONSISTENCY = "consistency"  # Maintain consistency using reference
    UPSCALE = "upscale"     # Enhance resolution/quality


@dataclass
class RefinementRequest:
    """Request for video refinement."""
    video_url: str
    prompt: str
    mode: RefinementMode = RefinementMode.EDIT
    negative_prompt: Optional[str] = None
    reference_video_url: Optional[str] = None  # For consistency mode
    style_image_urls: Optional[list[str]] = None  # Up to 4
    element_images: Optional[list[dict]] = None  # Character/object references
    duration: int = 5  # 5 or 10 seconds
    aspect_ratio: str = "auto"  # auto, 16:9, 9:16, 1:1
    keep_audio: bool = False


@dataclass
class RefinementResult:
    """Result from video refinement."""
    success: bool
    video_url: Optional[str] = None
    duration: Optional[float] = None
    file_size_bytes: Optional[int] = None
    processing_time_ms: Optional[float] = None
    cost_usd: Optional[float] = None
    error: Optional[str] = None


class VideoRefiner:
    """
    AI video refiner using fal.ai Kling O1 models.

    Supports:
    - Edit mode: Fix specific issues while preserving overall video
    - Consistency mode: Ensure visual coherence across clips
    """

    # fal.ai endpoints
    EDIT_ENDPOINT = "fal-ai/kling-video/o1/video-to-video/edit"
    REFERENCE_ENDPOINT = "fal-ai/kling-video/o1/video-to-video/reference"

    # Pricing per second
    COST_PER_SECOND = 0.168

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("FAL_KEY")
        if not self.api_key:
            logger.warning("FAL_KEY not configured - video refinement disabled")

        self.base_url = "https://queue.fal.run"

    async def refine(self, request: RefinementRequest) -> RefinementResult:
        """
        Refine a video using AI enhancement.

        Args:
            request: Refinement parameters

        Returns:
            RefinementResult with refined video URL
        """
        if not self.api_key:
            return RefinementResult(
                success=False,
                error="FAL_KEY not configured"
            )

        import time
        start_time = time.time()

        try:
            if request.mode == RefinementMode.EDIT:
                result = await self._refine_edit(request)
            elif request.mode == RefinementMode.CONSISTENCY:
                result = await self._refine_consistency(request)
            else:
                return RefinementResult(
                    success=False,
                    error=f"Unsupported mode: {request.mode}"
                )

            processing_time = (time.time() - start_time) * 1000

            if result.success:
                result.processing_time_ms = processing_time
                # Estimate cost based on duration
                if result.duration:
                    result.cost_usd = result.duration * self.COST_PER_SECOND

            return result

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return RefinementResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )

    async def _refine_edit(self, request: RefinementRequest) -> RefinementResult:
        """Edit mode - polish/fix specific elements."""

        payload = {
            "prompt": request.prompt,
            "video_url": request.video_url,
            "keep_audio": request.keep_audio,
        }

        if request.style_image_urls:
            payload["image_urls"] = request.style_image_urls[:4]

        if request.element_images:
            payload["elements"] = request.element_images[:4]

        return await self._submit_and_wait(self.EDIT_ENDPOINT, payload)

    async def _refine_consistency(self, request: RefinementRequest) -> RefinementResult:
        """Consistency mode - maintain visual coherence."""

        payload = {
            "prompt": request.prompt,
            "video_url": request.video_url,
            "aspect_ratio": request.aspect_ratio,
            "duration": str(request.duration),
            "keep_audio": request.keep_audio,
        }

        if request.style_image_urls:
            payload["image_urls"] = request.style_image_urls[:4]

        if request.element_images:
            payload["elements"] = request.element_images[:4]

        return await self._submit_and_wait(self.REFERENCE_ENDPOINT, payload)

    async def _submit_and_wait(
        self,
        endpoint: str,
        payload: dict,
        timeout: float = 300.0
    ) -> RefinementResult:
        """Submit job to fal.ai and wait for completion."""

        headers = {
            "Authorization": f"Key {self.api_key}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            # Submit job
            submit_url = f"{self.base_url}/{endpoint}"
            logger.info("Submitting refinement job", endpoint=endpoint)

            response = await client.post(submit_url, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            request_id = result.get("request_id")

            if not request_id:
                return RefinementResult(
                    success=False,
                    error="No request_id in response"
                )

            # Poll for completion
            status_url = f"{self.base_url}/{endpoint}/requests/{request_id}/status"
            result_url = f"https://fal.run/{endpoint}/requests/{request_id}"

            for attempt in range(60):  # Max 5 minutes of polling
                await asyncio.sleep(5)

                status_response = await client.get(status_url, headers=headers)
                status_data = status_response.json()

                status = status_data.get("status")
                logger.debug(f"Refinement status: {status}", attempt=attempt)

                if status == "COMPLETED":
                    # Get result
                    result_response = await client.get(result_url, headers=headers)
                    result_data = result_response.json()

                    video_info = result_data.get("video", {})
                    return RefinementResult(
                        success=True,
                        video_url=video_info.get("url"),
                        file_size_bytes=video_info.get("file_size"),
                    )

                elif status == "FAILED":
                    error = status_data.get("error", "Unknown error")
                    return RefinementResult(
                        success=False,
                        error=f"fal.ai job failed: {error}"
                    )

            return RefinementResult(
                success=False,
                error="Refinement timed out"
            )

    async def polish_clip(
        self,
        video_url: str,
        style_description: str,
        fix_issues: Optional[list[str]] = None
    ) -> RefinementResult:
        """
        Convenience method to polish a single clip.

        Args:
            video_url: URL of the video to polish
            style_description: Desired style (from VisualStyleGuide)
            fix_issues: Specific issues to fix (e.g., ["remove artifacts", "smooth motion"])

        Returns:
            RefinementResult
        """
        prompt_parts = [f"Polish this video to match {style_description} style."]

        if fix_issues:
            prompt_parts.append("Fix the following issues:")
            prompt_parts.extend([f"- {issue}" for issue in fix_issues])

        prompt_parts.append("Maintain the original composition and motion.")

        return await self.refine(RefinementRequest(
            video_url=video_url,
            prompt=" ".join(prompt_parts),
            mode=RefinementMode.EDIT
        ))

    async def ensure_consistency(
        self,
        clips: list[str],
        reference_clip_index: int = 0,
        style_description: str = ""
    ) -> list[RefinementResult]:
        """
        Ensure visual consistency across multiple clips.

        Args:
            clips: List of video URLs
            reference_clip_index: Which clip to use as the style reference
            style_description: Additional style guidance

        Returns:
            List of RefinementResults for each processed clip
        """
        if not clips:
            return []

        reference_url = clips[reference_clip_index]
        results = []

        for i, clip_url in enumerate(clips):
            if i == reference_clip_index:
                # Skip reference clip
                results.append(RefinementResult(
                    success=True,
                    video_url=clip_url
                ))
                continue

            prompt = f"Match the visual style of @Video1. {style_description}. Maintain motion and composition."

            result = await self.refine(RefinementRequest(
                video_url=clip_url,
                prompt=prompt,
                mode=RefinementMode.CONSISTENCY,
                reference_video_url=reference_url
            ))
            results.append(result)

        return results


# Global instance
_refiner: Optional[VideoRefiner] = None


def get_video_refiner() -> VideoRefiner:
    """Get or create the global video refiner instance."""
    global _refiner
    if _refiner is None:
        _refiner = VideoRefiner()
    return _refiner
