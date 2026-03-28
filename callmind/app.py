"""CallMind — FastAPI application.

Routes:
    GET  /                         → Landing page (upload form + client list)
    POST /register                 → Register new user (get API key via Unkey)
    POST /upload                   → Accept video file or YouTube URL
    GET  /client/{client_id}       → Client dashboard (all insights, Q-values)
    GET  /client/{client_id}/prep  → Call prep briefing
    POST /client/{client_id}/outcome → Record deal outcome (Q-learning reward)
    GET  /api/status/{job_id}      → Processing status
    POST /api/upload               → API upload (requires Unkey API key)
"""

import asyncio
import logging
import shutil
import uuid
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import APP_HOST, APP_PORT, UNKEY_API_ID, UNKEY_ROOT_KEY, UPLOAD_DIR
from . import memory
from . import video_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- App setup ---

app = FastAPI(
    title="CallMind",
    description="AI Sales Memory That Learns",
    version="0.1.0",
)

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.on_event("startup")
async def startup():
    """Ensure Qdrant collection exists on startup."""
    logger.info("CallMind starting up...")
    memory.ensure_collection()
    if UNKEY_ROOT_KEY:
        logger.info("Unkey auth enabled (API ID: %s)", UNKEY_API_ID)
    else:
        logger.info("Unkey auth disabled (no UNKEY_ROOT_KEY)")
    logger.info("CallMind ready at http://%s:%d", APP_HOST, APP_PORT)


# --- Unkey helpers ---


def _unkey_create_key(owner_name: str) -> dict:
    """Create a new API key for a user via Unkey REST API."""
    import httpx

    resp = httpx.post(
        "https://api.unkey.com/v2/keys.createKey",
        headers={"Authorization": f"Bearer {UNKEY_ROOT_KEY}", "Content-Type": "application/json"},
        json={
            "apiId": UNKEY_API_ID,
            "name": f"callmind_{owner_name}",
            "meta": {"registered_via": "callmind_web", "owner": owner_name},
        },
    )
    resp.raise_for_status()
    data = resp.json().get("data", resp.json())
    return {"key": data["key"], "key_id": data["keyId"]}


def _unkey_verify(api_key: str) -> dict | None:
    """Verify an API key via Unkey REST API."""
    import httpx

    resp = httpx.post(
        "https://api.unkey.com/v2/keys.verifyKey",
        headers={"Authorization": f"Bearer {UNKEY_ROOT_KEY}", "Content-Type": "application/json"},
        json={"key": api_key},
    )
    resp.raise_for_status()
    data = resp.json().get("data", resp.json())
    if data.get("valid"):
        meta = data.get("meta", {})
        return {"key_id": data.get("keyId"), "owner_id": meta.get("owner"), "name": data.get("name")}
    return None


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Landing page: upload form + list of clients."""
    clients = memory.get_all_clients()
    return templates.TemplateResponse(request=request, name="index.html", context={
        "clients": clients,
    })


@app.post("/register")
async def register(request: Request, username: str = Form(...)):
    """Register a new user and get an API key via Unkey."""
    if not UNKEY_ROOT_KEY:
        return JSONResponse(status_code=503, content={"error": "Unkey not configured"})

    try:
        key_data = _unkey_create_key(username.strip().lower().replace(" ", "_"))
        return templates.TemplateResponse(request=request, name="registered.html", context={
            "username": username,
            "api_key": key_data["key"],
            "key_id": key_data["key_id"],
        })
    except Exception as e:
        logger.exception("Unkey registration failed: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    request: Request,
    client_name: str = Form(""),
    youtube_url: str = Form(""),
    call_date: str = Form(""),
    video_file: UploadFile | None = File(None),
):
    """Accept a video upload or YouTube URL, start processing."""
    job_id = str(uuid.uuid4())[:8]

    # Auto-generate client name if not provided
    if not client_name.strip():
        if youtube_url and youtube_url.strip():
            client_name = f"Call {job_id}"
        elif video_file and video_file.filename:
            client_name = Path(video_file.filename).stem.replace("_", " ").replace("-", " ").title()
        else:
            client_name = f"Call {job_id}"

    client_id = client_name.strip().lower().replace(" ", "_")

    if youtube_url and youtube_url.strip():
        source = youtube_url.strip()
    elif video_file and video_file.filename:
        file_ext = Path(video_file.filename).suffix or ".mp4"
        dest = UPLOAD_DIR / f"{job_id}{file_ext}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(video_file.file, f)
        source = str(dest)
        logger.info("Uploaded file saved: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Provide either a YouTube URL or upload a video file"},
        )

    background_tasks.add_task(_run_pipeline, source, client_id, call_date, job_id)

    return templates.TemplateResponse(request=request, name="processing.html", context={
        "job_id": job_id,
        "client_id": client_id,
        "client_name": client_name,
        "source": source,
    })


# --- API endpoints (Unkey-authenticated) ---


@app.post("/api/upload")
async def api_upload(
    background_tasks: BackgroundTasks,
    client_name: str = Form(...),
    youtube_url: str = Form(""),
    call_date: str = Form(""),
    video_file: UploadFile | None = File(None),
    authorization: str = Header(""),
):
    """API endpoint for programmatic upload. Requires Unkey API key."""
    # Verify API key
    api_key = authorization.replace("Bearer ", "").strip()
    if UNKEY_ROOT_KEY and api_key:
        key_info = _unkey_verify(api_key)
        if not key_info:
            return JSONResponse(status_code=401, content={"error": "Invalid API key"})
        logger.info("API upload authorized for: %s", key_info.get("owner_id"))
    elif UNKEY_ROOT_KEY:
        return JSONResponse(status_code=401, content={"error": "API key required. Register at /register"})

    client_id = client_name.strip().lower().replace(" ", "_")
    job_id = str(uuid.uuid4())[:8]

    if youtube_url and youtube_url.strip():
        source = youtube_url.strip()
    elif video_file and video_file.filename:
        file_ext = Path(video_file.filename).suffix or ".mp4"
        dest = UPLOAD_DIR / f"{job_id}{file_ext}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(video_file.file, f)
        source = str(dest)
    else:
        return JSONResponse(status_code=400, content={"error": "Provide youtube_url or video_file"})

    background_tasks.add_task(_run_pipeline, source, client_id, call_date, job_id)

    return JSONResponse(content={
        "job_id": job_id,
        "client_id": client_id,
        "status": "processing",
        "status_url": f"/api/status/{job_id}",
    })


@app.get("/api/client/{client_id}/insights")
async def api_insights(
    client_id: str,
    q: str = "",
    authorization: str = Header(""),
):
    """API endpoint: get client insights. Requires Unkey API key."""
    api_key = authorization.replace("Bearer ", "").strip()
    if UNKEY_ROOT_KEY and api_key:
        key_info = _unkey_verify(api_key)
        if not key_info:
            return JSONResponse(status_code=401, content={"error": "Invalid API key"})
    elif UNKEY_ROOT_KEY:
        return JSONResponse(status_code=401, content={"error": "API key required"})

    insights = memory.get_client_insights(client_id, query=q, limit=50)
    return JSONResponse(content={"client_id": client_id, "insights": insights})


def _run_pipeline(source: str, client_id: str, call_date: str, job_id: str):
    """Run the video pipeline (called as background task)."""
    try:
        video_pipeline.process_video(
            file_path_or_url=source,
            client_id=client_id,
            call_date=call_date,
            job_id=job_id,
        )
    except Exception as e:
        logger.exception("Background pipeline failed: %s", e)


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    """Get processing status for a job."""
    job = video_pipeline.get_job(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    result = job.get("result")
    if result:
        result = {
            "insights_count": result.get("insights_count", 0),
            "text_insights": result.get("text_insights", 0),
            "visual_insights": result.get("visual_insights", 0),
            "fusion_insights": result.get("fusion_insights", 0),
            "transcript_length": result.get("transcript_length", 0),
        }

    return JSONResponse(content={
        "job_id": job["job_id"],
        "client_id": job["client_id"],
        "status": job["status"],
        "progress": job["progress"],
        "error": job.get("error"),
        "result": result,
    })


@app.get("/client/{client_id}", response_class=HTMLResponse)
async def client_dashboard(request: Request, client_id: str, q: str = ""):
    """Client dashboard: all insights with Q-values."""
    insights = memory.get_client_insights(client_id, query=q, limit=50)

    by_type: dict[str, list] = {}
    for insight in insights:
        t = insight["type"]
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(insight)

    q_values = [i["q_value"] for i in insights]
    stats = {
        "total": len(insights),
        "avg_q": round(sum(q_values) / len(q_values), 3) if q_values else 0,
        "max_q": round(max(q_values), 3) if q_values else 0,
        "min_q": round(min(q_values), 3) if q_values else 0,
        "types": len(by_type),
    }

    return templates.TemplateResponse(request=request, name="client.html", context={
        "client_id": client_id,
        "client_name": client_id.replace("_", " ").title(),
        "insights": insights,
        "by_type": by_type,
        "stats": stats,
        "query": q,
    })


@app.get("/client/{client_id}/prep", response_class=HTMLResponse)
async def call_prep(request: Request, client_id: str):
    """Pre-call briefing: top insights ranked by Q-value."""
    prep = memory.get_call_prep(client_id)

    return templates.TemplateResponse(request=request, name="prep.html", context={
        "client_id": client_id,
        "client_name": client_id.replace("_", " ").title(),
        "prep": prep,
    })


@app.post("/client/{client_id}/outcome")
async def record_outcome(
    request: Request,
    client_id: str,
    outcome: str = Form(...),
    reward: float = Form(...),
):
    """Record a deal outcome and update Q-values for all client insights."""
    result = memory.update_q_values(client_id, outcome, reward)
    return RedirectResponse(url=f"/client/{client_id}", status_code=303)


# --- Entry point ---


def main():
    uvicorn.run(
        "callmind.app:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()
