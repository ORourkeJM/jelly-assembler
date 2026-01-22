# Jelly Assembler

Video assembly microservice for the Jellybop platform. Combines video clips, runs AI refinement, and adds audio overlay.

## Pipeline Stages

```
1. ASSEMBLE  →  2. REFINE (optional)  →  3. FINALIZE
   (concat)        (AI enhancement)        (add audio)
```

1. **Assemble**: Concatenate clips into single video (no audio)
2. **Refine**: Send to AI refinement service for enhancement (optional)
3. **Finalize**: Overlay audio track on the video

## API Endpoints

### Full Pipeline
```bash
POST /api/v1/pipeline
{
  "clips": [
    {"url": "https://...", "order": 0},
    {"url": "https://...", "order": 1}
  ],
  "audio_url": "https://...",
  "audio_start": 0.0,
  "audio_end": 30.0,
  "quality": "standard",
  "resolution": "1080x1920",
  "refinement_type": "consistency",  // optional
  "callback_url": "https://your-webhook.com"
}
```

### Assembly Only (no audio)
```bash
POST /api/v1/assemble
{
  "clips": [{"url": "...", "order": 0}, ...],
  "resolution": "1080x1920",
  "fps": 30
}
```

### Finalize (add audio to existing video)
```bash
POST /api/v1/finalize
{
  "video_url": "https://...",
  "audio_url": "https://...",
  "audio_start": 0.0,
  "audio_end": 30.0,
  "quality": "standard"
}
```

### Job Status
```bash
GET /api/v1/jobs/{job_id}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `R2_ACCOUNT_ID` | Cloudflare R2 account | - |
| `R2_ACCESS_KEY_ID` | R2 access key | - |
| `R2_SECRET_ACCESS_KEY` | R2 secret key | - |
| `R2_BUCKET_NAME` | R2 bucket name | `jellybop-videos` |
| `R2_PUBLIC_URL` | Public URL base | `https://{bucket}.r2.dev` |
| `REFINEMENT_SERVICE_URL` | AI refinement service URL | - |
| `LOCAL_OUTPUT_DIR` | Local output (dev mode) | `/tmp/jelly-assembler-output` |

## Quality Presets

| Preset | Video CRF | Encoding | Audio |
|--------|-----------|----------|-------|
| draft | 28 | veryfast | 128k |
| standard | 23 | medium | 192k |
| premium | 18 | slow | 320k |

## Local Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

## Docker

```bash
docker build -t jelly-assembler .
docker run -p 8080:8080 jelly-assembler
```

## Railway Deployment

This service is deployed on Railway. Set the required environment variables in the Railway dashboard.
