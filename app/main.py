"""
Jelly Assembler - Video Assembly Service

FastAPI service that assembles video clips, runs AI refinement, and adds audio overlay.

Pipeline:
1. Assemble - Concatenate clips into single video (no audio)
2. Refine - Send to AI refinement service (optional)
3. Finalize - Overlay audio track
"""

import os
import uuid
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import structlog

from app.assembler import VideoAssembler, ClipInfo
from app.storage import get_storage_handler

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer()
    ]
)

logger = structlog.get_logger(__name__)

# Global instances
assembler: Optional[VideoAssembler] = None
storage = None
jobs: dict[str, dict] = {}  # In-memory job tracking


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global assembler, storage

    # Startup
    assembler = VideoAssembler()
    storage = get_storage_handler()
    logger.info("Jelly Assembler started")

    yield

    # Shutdown
    if assembler:
        assembler.cleanup()
    logger.info("Jelly Assembler stopped")


app = FastAPI(
    title="Jelly Assembler",
    description="Video assembly pipeline - combines clips, refines with AI, adds audio",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════════════════

class ClipInput(BaseModel):
    """Input for a single video clip."""
    url: str = Field(..., description="URL to download the clip from")
    order: int = Field(..., description="Order in final video (0-indexed)")
    duration: Optional[float] = Field(None, description="Clip duration in seconds")


class PipelineRequest(BaseModel):
    """Request for full assembly pipeline."""
    clips: list[ClipInput] = Field(..., description="List of clips to combine")
    audio_url: str = Field(..., description="URL of audio track")
    audio_start: float = Field(0.0, description="Start offset in audio (seconds)")
    audio_end: Optional[float] = Field(None, description="End offset in audio (seconds)")
    quality: str = Field("standard", description="Quality preset: draft, standard, premium")
    resolution: str = Field("1080x1920", description="Output resolution (WxH)")
    fps: int = Field(30, description="Output framerate")
    refinement_type: Optional[str] = Field(None, description="AI refinement: consistency, upscale, smooth")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion")


class AssembleOnlyRequest(BaseModel):
    """Request for assembly only (no audio, no refinement)."""
    clips: list[ClipInput] = Field(..., description="List of clips to combine")
    resolution: str = Field("1080x1920", description="Output resolution (WxH)")
    fps: int = Field(30, description="Output framerate")


class FinalizeRequest(BaseModel):
    """Request to add audio to a video."""
    video_url: str = Field(..., description="URL of video to add audio to")
    audio_url: str = Field(..., description="URL of audio track")
    audio_start: float = Field(0.0, description="Start offset in audio (seconds)")
    audio_end: Optional[float] = Field(None, description="End offset in audio (seconds)")
    quality: str = Field("standard", description="Quality preset")


class JobResponse(BaseModel):
    """Response for job creation."""
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Status of a job."""
    job_id: str
    status: str  # pending, processing, complete, failed
    stage: Optional[str] = None  # assemble, refine, finalize
    progress: Optional[int] = None
    created_at: str
    completed_at: Optional[str] = None
    output_url: Optional[str] = None
    duration: Optional[float] = None
    file_size_bytes: Optional[int] = None
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "jelly-assembler",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/pipeline", response_model=JobResponse)
async def create_pipeline_job(
    request: PipelineRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a full pipeline job: Assemble → Refine → Finalize.

    The job runs in the background. Poll /api/v1/jobs/{job_id} for status.
    """
    job_id = uuid.uuid4().hex[:16]

    # Validate clips
    if not request.clips:
        raise HTTPException(status_code=400, detail="At least one clip required")

    if len(request.clips) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 clips allowed")

    # Create job record
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "stage": "queued",
        "progress": 0,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Start background processing
    background_tasks.add_task(
        process_pipeline_job,
        job_id,
        request,
        request.callback_url
    )

    logger.info(
        "Pipeline job created",
        job_id=job_id,
        num_clips=len(request.clips),
        refinement=request.refinement_type
    )

    return JobResponse(
        job_id=job_id,
        status="pending",
        message=f"Pipeline started with {len(request.clips)} clips"
    )


@app.post("/api/v1/assemble", response_model=JobResponse)
async def create_assemble_job(
    request: AssembleOnlyRequest,
    background_tasks: BackgroundTasks
):
    """
    Assemble clips only (no audio, no refinement).

    Use this for the first stage, then call /api/v1/finalize to add audio.
    """
    job_id = uuid.uuid4().hex[:16]

    if not request.clips:
        raise HTTPException(status_code=400, detail="At least one clip required")

    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "stage": "assemble",
        "progress": 0,
        "created_at": datetime.utcnow().isoformat(),
    }

    background_tasks.add_task(process_assemble_job, job_id, request)

    return JobResponse(
        job_id=job_id,
        status="pending",
        message=f"Assembly started with {len(request.clips)} clips"
    )


@app.post("/api/v1/finalize", response_model=JobResponse)
async def create_finalize_job(
    request: FinalizeRequest,
    background_tasks: BackgroundTasks
):
    """
    Add audio to a video (finalization stage).

    Use after assembly and/or refinement.
    """
    job_id = uuid.uuid4().hex[:16]

    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "stage": "finalize",
        "progress": 0,
        "created_at": datetime.utcnow().isoformat(),
    }

    background_tasks.add_task(process_finalize_job, job_id, request)

    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Finalization started"
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(**jobs[job_id])


@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and clean up its files."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if assembler:
        assembler.cleanup(job_id)

    del jobs[job_id]
    return {"status": "deleted", "job_id": job_id}


@app.get("/api/v1/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """List recent jobs."""
    result = list(jobs.values())

    if status:
        result = [j for j in result if j.get("status") == status]

    result.sort(key=lambda j: j.get("created_at", ""), reverse=True)

    return {"jobs": result[:limit], "total": len(result)}


@app.post("/api/v1/upload")
async def upload_file(
    file: UploadFile = File(...),
    folder: str = Form("uploads"),
):
    """
    Upload a file to R2 storage.

    Returns the public URL of the uploaded file.
    """
    global storage

    if not storage:
        raise HTTPException(status_code=500, detail="Storage not initialized")

    # Validate file type
    allowed_types = ["video/mp4", "audio/mpeg", "audio/mp3", "audio/wav", "video/quicktime"]
    content_type = file.content_type or "application/octet-stream"

    if content_type not in allowed_types and not content_type.startswith("video/") and not content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Allowed: video/*, audio/*"
        )

    # Save to temp file
    import tempfile
    import shutil

    ext = os.path.splitext(file.filename or "file")[1] or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Upload to R2 with custom folder
        from datetime import datetime

        file_id = uuid.uuid4().hex[:12]
        key = f"{folder}/{file_id}{ext}"

        # Direct R2 upload with custom key
        if hasattr(storage, 'client') and storage.client:
            with open(tmp_path, "rb") as f:
                storage.client.put_object(
                    Bucket=storage.bucket_name,
                    Key=key,
                    Body=f,
                    ContentType=content_type,
                )
            url = f"{storage.public_url_base}/{key}"
        else:
            # Fallback to standard upload
            url = await storage.upload(tmp_path, content_type)

        logger.info("File uploaded", filename=file.filename, key=key, url=url)

        return {
            "success": True,
            "url": url,
            "key": key,
            "filename": file.filename,
            "content_type": content_type,
            "size_bytes": os.path.getsize(tmp_path)
        }

    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# Background Processing
# ═══════════════════════════════════════════════════════════════════════════

def update_job(job_id: str, **kwargs):
    """Update job status."""
    if job_id in jobs:
        jobs[job_id].update(kwargs)


async def process_pipeline_job(
    job_id: str,
    request: PipelineRequest,
    callback_url: Optional[str] = None
):
    """Run full pipeline: Assemble → Refine → Finalize."""
    global assembler, storage

    def on_progress(stage: str, pct: int, msg: str):
        update_job(job_id, stage=stage, progress=pct, status="processing")

    update_job(job_id, status="processing", stage="assemble", progress=0)

    try:
        clips = [
            ClipInfo(url=c.url, order=c.order, duration=c.duration)
            for c in request.clips
        ]

        result = await assembler.run_pipeline(
            job_id=job_id,
            clips=clips,
            audio_url=request.audio_url,
            audio_start=request.audio_start,
            audio_end=request.audio_end,
            resolution=request.resolution,
            fps=request.fps,
            quality=request.quality,
            refinement_type=request.refinement_type,
            on_progress=on_progress
        )

        if result.success:
            output_url = await storage.upload(result.output_path)

            update_job(
                job_id,
                status="complete",
                stage="done",
                progress=100,
                completed_at=datetime.utcnow().isoformat(),
                output_url=output_url,
                duration=result.duration,
                file_size_bytes=result.file_size_bytes,
                processing_time_ms=result.processing_time_ms,
            )

            logger.info("Pipeline complete", job_id=job_id, output_url=output_url)
        else:
            update_job(
                job_id,
                status="failed",
                completed_at=datetime.utcnow().isoformat(),
                error=result.error,
                processing_time_ms=result.processing_time_ms,
            )

        if callback_url:
            await send_callback(callback_url, jobs[job_id])

    except Exception as e:
        logger.error(f"Pipeline error: {e}", job_id=job_id)
        update_job(
            job_id,
            status="failed",
            completed_at=datetime.utcnow().isoformat(),
            error=str(e)
        )


async def process_assemble_job(job_id: str, request: AssembleOnlyRequest):
    """Run assembly stage only."""
    global assembler, storage

    update_job(job_id, status="processing", stage="assemble", progress=10)

    try:
        from app.assembler import AssemblyJob

        clips = [
            ClipInfo(url=c.url, order=c.order, duration=c.duration)
            for c in request.clips
        ]

        job = AssemblyJob(
            job_id=job_id,
            clips=clips,
            resolution=request.resolution,
            fps=request.fps
        )

        result = await assembler.assemble(job)

        if result.success:
            output_url = await storage.upload(result.output_path)

            update_job(
                job_id,
                status="complete",
                stage="assemble",
                progress=100,
                completed_at=datetime.utcnow().isoformat(),
                output_url=output_url,
                duration=result.duration,
                file_size_bytes=result.file_size_bytes,
                processing_time_ms=result.processing_time_ms,
            )
        else:
            update_job(
                job_id,
                status="failed",
                completed_at=datetime.utcnow().isoformat(),
                error=result.error,
            )

    except Exception as e:
        logger.error(f"Assemble error: {e}", job_id=job_id)
        update_job(job_id, status="failed", error=str(e))


async def process_finalize_job(job_id: str, request: FinalizeRequest):
    """Run finalization (add audio) stage only."""
    global assembler, storage

    update_job(job_id, status="processing", stage="finalize", progress=10)

    try:
        from app.assembler import FinalizeJob
        import tempfile
        import httpx

        # Download video to temp file
        job_dir = os.path.join(assembler.work_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)

        video_path = os.path.join(job_dir, "input.mp4")
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(request.video_url, follow_redirects=True)
            response.raise_for_status()
            with open(video_path, "wb") as f:
                f.write(response.content)

        job = FinalizeJob(
            job_id=job_id,
            video_path=video_path,
            audio_url=request.audio_url,
            audio_start=request.audio_start,
            audio_end=request.audio_end,
            quality=request.quality
        )

        result = await assembler.finalize(job)

        if result.success:
            output_url = await storage.upload(result.output_path)

            update_job(
                job_id,
                status="complete",
                stage="finalize",
                progress=100,
                completed_at=datetime.utcnow().isoformat(),
                output_url=output_url,
                duration=result.duration,
                file_size_bytes=result.file_size_bytes,
                processing_time_ms=result.processing_time_ms,
            )
        else:
            update_job(
                job_id,
                status="failed",
                completed_at=datetime.utcnow().isoformat(),
                error=result.error,
            )

    except Exception as e:
        logger.error(f"Finalize error: {e}", job_id=job_id)
        update_job(job_id, status="failed", error=str(e))


async def send_callback(url: str, data: dict):
    """Send webhook callback."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(url, json=data)
            logger.info("Callback sent", url=url)
    except Exception as e:
        logger.warning(f"Callback failed: {e}", url=url)


# ═══════════════════════════════════════════════════════════════════════════
# Local file serving (dev mode)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/files/{filename}")
async def serve_file(filename: str):
    """Serve assembled files in dev mode."""
    output_dir = os.environ.get("LOCAL_OUTPUT_DIR", "/tmp/jelly-assembler-output")
    file_path = os.path.join(output_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, media_type="video/mp4", filename=filename)


# ═══════════════════════════════════════════════════════════════════════════
# Video Refinement (Veo 3.1)
# ═══════════════════════════════════════════════════════════════════════════

class RefineRequest(BaseModel):
    """Request for video refinement using Veo."""
    prompt: str = Field(..., description="Description of desired video")
    first_frame_url: Optional[str] = Field(None, description="URL of starting frame image")
    last_frame_url: Optional[str] = Field(None, description="URL of ending frame image")
    reference_image_urls: Optional[list[str]] = Field(None, description="Style/character reference images")
    duration_seconds: int = Field(8, description="Video duration (4, 6, or 8 seconds)")
    aspect_ratio: str = Field("9:16", description="Aspect ratio: 9:16 or 16:9")
    model: str = Field("fast", description="Model: fast ($0.10/s) or standard ($0.40/s)")


class TransitionRequest(BaseModel):
    """Request for creating a transition between frames."""
    first_frame_url: str = Field(..., description="URL of first frame (end of clip 1)")
    last_frame_url: str = Field(..., description="URL of last frame (start of clip 2)")
    prompt: str = Field(..., description="Description of transition style")
    duration_seconds: int = Field(4, description="Transition duration (4, 6, or 8 seconds)")


@app.post("/api/v1/refine")
async def refine_video(request: RefineRequest, background_tasks: BackgroundTasks):
    """
    Generate or refine video using Google Veo 3.1.

    Supports:
    - Text-to-video: Just provide prompt
    - Image-to-video: Provide first_frame_url
    - Frames-to-video: Provide first_frame_url and last_frame_url
    - Style consistency: Provide reference_image_urls
    """
    from app.veo_refiner import get_veo_refiner, VeoRequest, VeoModel

    refiner = get_veo_refiner()

    model = VeoModel.VEO_31_FAST if request.model == "fast" else VeoModel.VEO_31

    job_id = uuid.uuid4().hex[:16]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "stage": "refine",
        "progress": 0,
        "created_at": datetime.utcnow().isoformat(),
    }

    async def run_refinement():
        update_job(job_id, status="processing", progress=10)

        result = await refiner.generate(VeoRequest(
            prompt=request.prompt,
            first_frame_url=request.first_frame_url,
            last_frame_url=request.last_frame_url,
            reference_image_urls=request.reference_image_urls,
            duration_seconds=request.duration_seconds,
            aspect_ratio=request.aspect_ratio,
            model=model
        ))

        if result.success:
            update_job(
                job_id,
                status="complete",
                progress=100,
                completed_at=datetime.utcnow().isoformat(),
                output_url=result.video_url,
                duration=result.duration,
                processing_time_ms=result.processing_time_ms
            )
        else:
            update_job(
                job_id,
                status="failed",
                completed_at=datetime.utcnow().isoformat(),
                error=result.error
            )

    background_tasks.add_task(run_refinement)

    return JobResponse(
        job_id=job_id,
        status="pending",
        message=f"Veo refinement started ({request.model} model)"
    )


@app.post("/api/v1/transition")
async def create_transition(request: TransitionRequest, background_tasks: BackgroundTasks):
    """
    Create a smooth transition between two video clips.

    Provide the last frame of clip 1 and first frame of clip 2.
    Veo will generate a smooth transition between them.
    """
    from app.veo_refiner import get_veo_refiner

    refiner = get_veo_refiner()

    job_id = uuid.uuid4().hex[:16]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "stage": "transition",
        "progress": 0,
        "created_at": datetime.utcnow().isoformat(),
    }

    async def run_transition():
        update_job(job_id, status="processing", progress=10)

        result = await refiner.create_transition(
            first_frame_url=request.first_frame_url,
            last_frame_url=request.last_frame_url,
            prompt=request.prompt,
            duration=request.duration_seconds
        )

        if result.success:
            update_job(
                job_id,
                status="complete",
                progress=100,
                completed_at=datetime.utcnow().isoformat(),
                output_url=result.video_url,
                duration=result.duration,
                processing_time_ms=result.processing_time_ms
            )
        else:
            update_job(
                job_id,
                status="failed",
                completed_at=datetime.utcnow().isoformat(),
                error=result.error
            )

    background_tasks.add_task(run_transition)

    return JobResponse(
        job_id=job_id,
        status="pending",
        message=f"Creating {request.duration_seconds}s transition"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Video Refinement (fal.ai Kling O1) - Alternative to Veo
# ═══════════════════════════════════════════════════════════════════════════

class FalRefineRequest(BaseModel):
    """Request for video refinement using fal.ai Kling O1."""
    video_url: str = Field(..., description="URL of video to refine")
    prompt: str = Field(..., description="Description of desired refinement")
    mode: str = Field("edit", description="Mode: edit (fix issues) or consistency (match style)")
    style_image_urls: Optional[list[str]] = Field(None, description="Style reference images (up to 4)")
    element_images: Optional[list[dict]] = Field(None, description="Character/object references")
    keep_audio: bool = Field(False, description="Preserve original audio")


@app.post("/api/v1/refine/fal")
async def refine_video_fal(request: FalRefineRequest, background_tasks: BackgroundTasks):
    """
    Refine video using fal.ai Kling O1 models.

    Alternative to Veo - costs $0.168/second.

    Modes:
    - edit: Polish/fix specific elements while preserving composition
    - consistency: Maintain visual coherence using style references

    Example:
        POST /api/v1/refine/fal
        {
            "video_url": "https://...",
            "prompt": "Enhance colors, smooth motion, add cinematic look",
            "mode": "edit"
        }
    """
    from app.video_refiner import get_video_refiner, RefinementRequest, RefinementMode

    refiner = get_video_refiner()

    if not refiner.api_key:
        raise HTTPException(status_code=500, detail="FAL_KEY not configured")

    mode = RefinementMode.EDIT if request.mode == "edit" else RefinementMode.CONSISTENCY

    job_id = uuid.uuid4().hex[:16]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "stage": "refine",
        "progress": 0,
        "created_at": datetime.utcnow().isoformat(),
    }

    async def run_fal_refinement():
        update_job(job_id, status="processing", progress=10)

        result = await refiner.refine(RefinementRequest(
            video_url=request.video_url,
            prompt=request.prompt,
            mode=mode,
            style_image_urls=request.style_image_urls,
            element_images=request.element_images,
            keep_audio=request.keep_audio
        ))

        if result.success:
            update_job(
                job_id,
                status="complete",
                progress=100,
                completed_at=datetime.utcnow().isoformat(),
                output_url=result.video_url,
                duration=result.duration,
                processing_time_ms=result.processing_time_ms
            )
        else:
            update_job(
                job_id,
                status="failed",
                completed_at=datetime.utcnow().isoformat(),
                error=result.error
            )

    background_tasks.add_task(run_fal_refinement)

    return JobResponse(
        job_id=job_id,
        status="pending",
        message=f"fal.ai refinement started ({request.mode} mode)"
    )


@app.post("/api/v1/polish")
async def polish_clip(
    video_url: str,
    style: str = "cinematic",
    fix_issues: Optional[list[str]] = None,
    background_tasks: BackgroundTasks = None
):
    """
    Convenience endpoint to polish a single video clip.

    Example:
        POST /api/v1/polish?video_url=https://...&style=neon%20aesthetic
    """
    from app.video_refiner import get_video_refiner

    refiner = get_video_refiner()

    if not refiner.api_key:
        raise HTTPException(status_code=500, detail="FAL_KEY not configured")

    job_id = uuid.uuid4().hex[:16]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "stage": "polish",
        "progress": 0,
        "created_at": datetime.utcnow().isoformat(),
    }

    async def run_polish():
        update_job(job_id, status="processing", progress=10)

        result = await refiner.polish_clip(
            video_url=video_url,
            style_description=style,
            fix_issues=fix_issues or []
        )

        if result.success:
            update_job(
                job_id,
                status="complete",
                progress=100,
                completed_at=datetime.utcnow().isoformat(),
                output_url=result.video_url,
                duration=result.duration,
                processing_time_ms=result.processing_time_ms
            )
        else:
            update_job(
                job_id,
                status="failed",
                completed_at=datetime.utcnow().isoformat(),
                error=result.error
            )

    background_tasks.add_task(run_polish)

    return JobResponse(
        job_id=job_id,
        status="pending",
        message=f"Polishing clip with {style} style"
    )
