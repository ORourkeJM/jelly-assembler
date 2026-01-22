"""
Video Assembler - Core assembly logic.

Pipeline:
1. Assemble: Concatenate clips into single video (no audio)
2. Refine: Send to AI service for enhancement (optional)
3. Finalize: Overlay audio track on refined video

Uses ffmpeg for all processing operations.
"""

import asyncio
import os
import tempfile
import shutil
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import httpx
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ClipInfo:
    """Information about a single video clip."""
    url: str
    order: int
    duration: Optional[float] = None
    start_time: Optional[float] = None  # Where it fits in the final video


@dataclass
class AssemblyJob:
    """A video assembly job."""
    job_id: str
    clips: list[ClipInfo]
    output_format: str = "mp4"
    resolution: str = "1080x1920"  # 9:16 vertical
    fps: int = 30


@dataclass
class RefinementJob:
    """AI refinement job."""
    job_id: str
    video_path: str
    refinement_type: str = "consistency"  # consistency, upscale, smooth
    model: str = "default"


@dataclass
class FinalizeJob:
    """Finalization job - add audio."""
    job_id: str
    video_path: str
    audio_url: str
    audio_start: float = 0.0
    audio_end: Optional[float] = None
    quality: str = "standard"


@dataclass
class StageResult:
    """Result of a pipeline stage."""
    success: bool
    stage: str
    output_path: Optional[str] = None
    duration: Optional[float] = None
    file_size_bytes: Optional[int] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None


class VideoAssembler:
    """
    Video assembly pipeline with 3 stages:
    1. Assemble - combine clips
    2. Refine - AI enhancement (optional)
    3. Finalize - add audio

    Features:
    - Downloads clips from URLs
    - Concatenates in order
    - Integrates with AI refinement services
    - Overlays audio track
    """

    # Quality presets for ffmpeg encoding
    QUALITY_PRESETS = {
        "draft": {
            "crf": "28",
            "preset": "veryfast",
            "audio_bitrate": "128k",
        },
        "standard": {
            "crf": "23",
            "preset": "medium",
            "audio_bitrate": "192k",
        },
        "premium": {
            "crf": "18",
            "preset": "slow",
            "audio_bitrate": "320k",
        },
    }

    def __init__(
        self,
        work_dir: Optional[str] = None,
        refinement_url: Optional[str] = None
    ):
        """Initialize assembler with working directory."""
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="jelly_assembler_")
        self.refinement_url = refinement_url or os.environ.get("REFINEMENT_SERVICE_URL")
        self.logger = logger.bind(component="video_assembler")

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 1: Assemble - Concatenate clips (no audio)
    # ═══════════════════════════════════════════════════════════════════════════

    async def assemble(self, job: AssemblyJob) -> StageResult:
        """
        Stage 1: Assemble clips into a single video (no audio).

        Args:
            job: Assembly job specification

        Returns:
            StageResult with output path to concatenated video
        """
        start_time = time.time()
        job_dir = os.path.join(self.work_dir, job.job_id)
        os.makedirs(job_dir, exist_ok=True)

        self.logger.info(
            "Stage 1: Assemble",
            job_id=job.job_id,
            num_clips=len(job.clips)
        )

        try:
            # Download all clips
            self.logger.info("Downloading clips...")
            clip_paths = await self._download_clips(job.clips, job_dir)

            # Normalize clips (same resolution, fps, no audio)
            self.logger.info("Normalizing clips...")
            normalized_paths = await self._normalize_clips(
                clip_paths, job_dir, job.resolution, job.fps
            )

            # Concatenate clips
            self.logger.info("Concatenating clips...")
            output_path = os.path.join(job_dir, f"assembled.{job.output_format}")
            await self._concatenate_clips(normalized_paths, output_path)

            # Get output info
            duration = await self._get_duration(output_path)
            file_size = os.path.getsize(output_path)
            processing_time = (time.time() - start_time) * 1000

            self.logger.info(
                "Stage 1 complete",
                job_id=job.job_id,
                duration=duration,
                file_size_mb=round(file_size / 1024 / 1024, 2),
                processing_time_ms=round(processing_time)
            )

            return StageResult(
                success=True,
                stage="assemble",
                output_path=output_path,
                duration=duration,
                file_size_bytes=file_size,
                processing_time_ms=processing_time
            )

        except Exception as e:
            self.logger.error(f"Assembly failed: {e}", job_id=job.job_id)
            return StageResult(
                success=False,
                stage="assemble",
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 2: Refine - AI Enhancement (optional)
    # ═══════════════════════════════════════════════════════════════════════════

    async def refine(self, job: RefinementJob) -> StageResult:
        """
        Stage 2: Send video to AI refinement service.

        Refinement types:
        - consistency: Improve temporal consistency between frames
        - upscale: AI upscaling to higher resolution
        - smooth: Motion smoothing / interpolation

        Args:
            job: Refinement job specification

        Returns:
            StageResult with path to refined video
        """
        start_time = time.time()

        self.logger.info(
            "Stage 2: Refine",
            job_id=job.job_id,
            refinement_type=job.refinement_type
        )

        if not self.refinement_url:
            # No refinement service configured - pass through
            self.logger.info("No refinement service configured, passing through")
            return StageResult(
                success=True,
                stage="refine",
                output_path=job.video_path,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        try:
            # Upload video to refinement service
            async with httpx.AsyncClient(timeout=600.0) as client:
                with open(job.video_path, "rb") as f:
                    files = {"video": f}
                    data = {
                        "job_id": job.job_id,
                        "refinement_type": job.refinement_type,
                        "model": job.model
                    }

                    response = await client.post(
                        f"{self.refinement_url}/api/v1/refine",
                        files=files,
                        data=data
                    )
                    response.raise_for_status()
                    result = response.json()

            # Download refined video
            if result.get("output_url"):
                refined_path = job.video_path.replace(".mp4", "_refined.mp4")
                await self._download_file(result["output_url"], os.path.dirname(refined_path), os.path.basename(refined_path))

                processing_time = (time.time() - start_time) * 1000

                return StageResult(
                    success=True,
                    stage="refine",
                    output_path=refined_path,
                    processing_time_ms=processing_time
                )

            return StageResult(
                success=False,
                stage="refine",
                error="No output URL from refinement service",
                processing_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            self.logger.error(f"Refinement failed: {e}", job_id=job.job_id)
            # On refinement failure, return original video (graceful degradation)
            return StageResult(
                success=True,  # Still "success" - we can continue with unrefined
                stage="refine",
                output_path=job.video_path,
                error=f"Refinement skipped: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # Stage 3: Finalize - Add Audio
    # ═══════════════════════════════════════════════════════════════════════════

    async def finalize(self, job: FinalizeJob) -> StageResult:
        """
        Stage 3: Overlay audio track on video.

        Args:
            job: Finalize job specification

        Returns:
            StageResult with path to final video with audio
        """
        start_time = time.time()
        job_dir = os.path.dirname(job.video_path)

        self.logger.info(
            "Stage 3: Finalize (add audio)",
            job_id=job.job_id,
            audio_start=job.audio_start
        )

        try:
            # Download audio
            audio_path = os.path.join(job_dir, "audio.mp3")
            await self._download_file(job.audio_url, job_dir, "audio.mp3")

            # Get video duration
            video_duration = await self._get_duration(job.video_path)

            # Trim audio to match video
            audio_end = job.audio_end or (job.audio_start + video_duration)
            trimmed_audio = os.path.join(job_dir, "audio_trimmed.mp3")
            await self._trim_audio(audio_path, trimmed_audio, job.audio_start, audio_end)

            # Overlay audio on video
            output_path = job.video_path.replace(".mp4", "_final.mp4")
            await self._overlay_audio(job.video_path, trimmed_audio, output_path, job.quality)

            # Get output info
            duration = await self._get_duration(output_path)
            file_size = os.path.getsize(output_path)
            processing_time = (time.time() - start_time) * 1000

            self.logger.info(
                "Stage 3 complete",
                job_id=job.job_id,
                duration=duration,
                file_size_mb=round(file_size / 1024 / 1024, 2)
            )

            return StageResult(
                success=True,
                stage="finalize",
                output_path=output_path,
                duration=duration,
                file_size_bytes=file_size,
                processing_time_ms=processing_time
            )

        except Exception as e:
            self.logger.error(f"Finalization failed: {e}", job_id=job.job_id)
            return StageResult(
                success=False,
                stage="finalize",
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # Full Pipeline
    # ═══════════════════════════════════════════════════════════════════════════

    async def run_pipeline(
        self,
        job_id: str,
        clips: list[ClipInfo],
        audio_url: str,
        audio_start: float = 0.0,
        audio_end: Optional[float] = None,
        resolution: str = "1080x1920",
        fps: int = 30,
        quality: str = "standard",
        refinement_type: Optional[str] = None,
        on_progress: Optional[callable] = None
    ) -> StageResult:
        """
        Run full pipeline: Assemble → Refine → Finalize.

        Args:
            job_id: Unique job identifier
            clips: List of clips to combine
            audio_url: URL of audio track
            audio_start: Start offset in audio
            audio_end: End offset in audio
            resolution: Output resolution
            fps: Output framerate
            quality: Encoding quality preset
            refinement_type: Optional AI refinement type
            on_progress: Progress callback(stage, progress, message)

        Returns:
            StageResult with final video
        """
        def progress(stage: str, pct: int, msg: str):
            if on_progress:
                on_progress(stage, pct, msg)
            self.logger.info(f"[{stage}] {pct}% - {msg}")

        progress("pipeline", 0, "Starting pipeline")

        # Stage 1: Assemble
        progress("assemble", 10, "Assembling clips...")
        assemble_job = AssemblyJob(
            job_id=job_id,
            clips=clips,
            resolution=resolution,
            fps=fps
        )
        assemble_result = await self.assemble(assemble_job)

        if not assemble_result.success:
            return assemble_result

        progress("assemble", 40, "Assembly complete")
        current_video = assemble_result.output_path

        # Stage 2: Refine (optional)
        if refinement_type:
            progress("refine", 45, f"Refining video ({refinement_type})...")
            refine_job = RefinementJob(
                job_id=job_id,
                video_path=current_video,
                refinement_type=refinement_type
            )
            refine_result = await self.refine(refine_job)

            if refine_result.output_path:
                current_video = refine_result.output_path

            progress("refine", 70, "Refinement complete")
        else:
            progress("refine", 70, "Skipping refinement")

        # Stage 3: Finalize (add audio)
        progress("finalize", 75, "Adding audio track...")
        finalize_job = FinalizeJob(
            job_id=job_id,
            video_path=current_video,
            audio_url=audio_url,
            audio_start=audio_start,
            audio_end=audio_end,
            quality=quality
        )
        finalize_result = await self.finalize(finalize_job)

        if finalize_result.success:
            progress("pipeline", 100, "Pipeline complete!")
        else:
            progress("pipeline", 100, f"Pipeline failed: {finalize_result.error}")

        return finalize_result

    # ═══════════════════════════════════════════════════════════════════════════
    # Helper Methods
    # ═══════════════════════════════════════════════════════════════════════════

    async def _download_clips(
        self, clips: list[ClipInfo], job_dir: str
    ) -> list[str]:
        """Download all clips in parallel."""
        tasks = []
        for clip in sorted(clips, key=lambda c: c.order):
            filename = f"clip_{clip.order:03d}.mp4"
            tasks.append(self._download_file(clip.url, job_dir, filename))

        return await asyncio.gather(*tasks)

    async def _download_file(
        self, url: str, job_dir: str, filename: str
    ) -> str:
        """Download a file from URL."""
        output_path = os.path.join(job_dir, filename)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

        self.logger.debug(f"Downloaded: {filename}", size_mb=round(os.path.getsize(output_path) / 1024 / 1024, 2))
        return output_path

    async def _normalize_clips(
        self,
        clip_paths: list[str],
        job_dir: str,
        resolution: str,
        fps: int
    ) -> list[str]:
        """Normalize all clips to same resolution and fps (no audio)."""
        width, height = resolution.split("x")
        normalized = []

        for i, path in enumerate(clip_paths):
            output = os.path.join(job_dir, f"norm_{i:03d}.mp4")

            # Scale and pad to exact resolution, set fps, remove audio
            cmd = [
                "ffmpeg", "-y", "-i", path,
                "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,fps={fps}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-an",  # Remove audio from clips
                output
            ]

            await self._run_ffmpeg(cmd)
            normalized.append(output)

        return normalized

    async def _concatenate_clips(
        self, clip_paths: list[str], output_path: str
    ) -> None:
        """Concatenate clips using ffmpeg concat demuxer."""
        # Create concat file list
        concat_file = output_path.replace(".mp4", "_list.txt")
        with open(concat_file, "w") as f:
            for path in clip_paths:
                f.write(f"file '{path}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_path
        ]

        await self._run_ffmpeg(cmd)

    async def _trim_audio(
        self,
        input_path: str,
        output_path: str,
        start: float,
        end: float
    ) -> None:
        """Trim audio to specified range."""
        duration = end - start

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-t", str(duration),
            "-i", input_path,
            "-c:a", "libmp3lame", "-q:a", "2",
            output_path
        ]

        await self._run_ffmpeg(cmd)

    async def _overlay_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        quality: str
    ) -> None:
        """Overlay audio on video with quality preset."""
        preset = self.QUALITY_PRESETS.get(quality, self.QUALITY_PRESETS["standard"])

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "libx264",
            "-preset", preset["preset"],
            "-crf", preset["crf"],
            "-c:a", "aac",
            "-b:a", preset["audio_bitrate"],
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            output_path
        ]

        await self._run_ffmpeg(cmd)

    async def _get_duration(self, path: str) -> float:
        """Get duration of a video/audio file."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()

        return float(stdout.decode().strip())

    async def _run_ffmpeg(self, cmd: list[str]) -> None:
        """Run ffmpeg command."""
        self.logger.debug(f"Running: {' '.join(cmd[:5])}...")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            error = stderr.decode()[-500:]  # Last 500 chars of error
            raise RuntimeError(f"ffmpeg failed: {error}")

    def cleanup(self, job_id: Optional[str] = None) -> None:
        """Clean up temporary files."""
        if job_id:
            job_dir = os.path.join(self.work_dir, job_id)
            if os.path.exists(job_dir):
                shutil.rmtree(job_dir)
        else:
            # Clean entire work dir
            if os.path.exists(self.work_dir):
                shutil.rmtree(self.work_dir)
