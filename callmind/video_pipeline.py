"""CallMind video processing pipeline.

Flow: video/URL → download → Gemini multimodal (transcript + visual) → insight extraction → Qdrant storage.
"""

import json
import logging
import re
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .config import GEMINI_API_KEY, GEMINI_MODEL, UPLOAD_DIR

logger = logging.getLogger(__name__)

# --- Job tracking (in-memory for hackathon simplicity) ---

_jobs: dict[str, dict[str, Any]] = {}


def get_job(job_id: str) -> dict[str, Any] | None:
    return _jobs.get(job_id)


def _update_job(job_id: str, **kwargs: Any) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)


# --- YouTube download ---


def download_youtube(url: str) -> Path:
    """Download a YouTube video to local storage using yt-dlp.

    Returns path to the downloaded file.
    """
    import yt_dlp

    output_dir = UPLOAD_DIR / "youtube"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(output_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "worst[ext=mp4]/worst",  # Smallest file for faster upload to Gemini
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    file_path = Path(filename)
    if not file_path.exists():
        # yt-dlp sometimes changes extension
        possible = list(output_dir.glob(f"{info['id']}.*"))
        if possible:
            file_path = possible[0]

    logger.info("Downloaded YouTube video: %s → %s", url, file_path)
    return file_path


# --- Gemini upload + wait ---


def _upload_and_wait(file_path: Path):
    """Upload video to Gemini Files API and wait until ACTIVE."""
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    logger.info("Uploading video to Gemini: %s (%.1f MB)", file_path.name, file_path.stat().st_size / 1e6)
    video_file = client.files.upload(file=file_path)

    max_wait = 300
    waited = 0
    while video_file.state.name == "PROCESSING" and waited < max_wait:
        time.sleep(5)
        waited += 5
        video_file = client.files.get(name=video_file.name)
        logger.debug("Video processing... (%ds)", waited)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Video processing failed: state={video_file.state.name} after {waited}s")

    logger.info("Video ready for analysis (waited %ds)", waited)
    return video_file


# --- Gemini transcription ---


def transcribe_video(video_file) -> str:
    """Generate transcript from an already-uploaded Gemini video file."""
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Generating transcript...")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            video_file,
            """You are transcribing a sales call recording. Provide a complete, accurate transcript.

Rules:
1. Identify different speakers as "Speaker 1", "Speaker 2", etc.
2. If you can identify who is the salesperson vs. the prospect, label them as "Sales Rep" and "Prospect"
3. Include timestamps roughly every 30 seconds in [MM:SS] format
4. Capture everything said — don't summarize or skip parts
5. Note any significant non-verbal cues like [long pause], [laughter], [screen sharing starts]

Format:
[00:00] Sales Rep: Hello, thanks for taking the time...
[00:15] Prospect: Of course, happy to chat about...

Begin the transcript now.""",
        ],
    )

    transcript = response.text
    logger.info("Transcript generated: %d characters", len(transcript))
    return transcript


# --- Visual analysis (parallel multimodal channel) ---

VISUAL_ANALYSIS_PROMPT = """You are an expert in non-verbal communication and sales psychology.
Analyze the VISUAL content of this sales call video. Focus ONLY on what you can SEE, not what is said.

Analyze these visual signals:
1. BODY LANGUAGE: Posture changes, leaning forward/back, arm crossing, hand gestures
2. FACIAL EXPRESSIONS: Smiles, frowns, raised eyebrows, micro-expressions indicating doubt/interest
3. ENGAGEMENT LEVEL: Eye contact, looking away, checking phone, note-taking
4. ENERGY SHIFTS: Moments where visual engagement noticeably increases or decreases
5. SCREEN SHARING: If any screen content is shown, describe what was presented
6. ENVIRONMENT: Professional setup, background, lighting (indicates preparation level)

For each observation, provide:
- type: one of [body_language, facial_expression, engagement_shift, visual_cue, environment]
- content: clear description of what you observed (1-2 sentences)
- timestamp: approximate MM:SS when this was observed
- confidence: 0.0-1.0
- signal: "positive", "negative", or "neutral" — what this signals about the prospect's state

Return ONLY valid JSON array. No markdown. Example:
[
  {
    "type": "body_language",
    "content": "Prospect leaned forward and started taking notes when pricing was discussed",
    "timestamp": "05:30",
    "confidence": 0.85,
    "signal": "positive"
  }
]

Extract 5-10 visual observations. Focus on moments that reveal the prospect's true feelings beyond their words."""


def analyze_video_visuals(video_file) -> list[dict[str, Any]]:
    """Extract visual/non-verbal insights from video using Gemini multimodal."""
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Analyzing video visuals...")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[video_file, VISUAL_ANALYSIS_PROMPT],
    )

    raw_text = response.text.strip()

    # Clean markdown fences
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    try:
        observations = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            try:
                observations = json.loads(match.group())
            except json.JSONDecodeError:
                observations = []
        else:
            observations = []

    if not isinstance(observations, list):
        observations = []

    valid_types = {"body_language", "facial_expression", "engagement_shift", "visual_cue", "environment"}
    cleaned = []
    for obs in observations:
        if not isinstance(obs, dict) or not obs.get("content"):
            continue

        otype = obs.get("type", "visual_cue")
        if otype not in valid_types:
            otype = "visual_cue"

        signal = obs.get("signal", "neutral")
        timestamp = obs.get("timestamp", "")
        content = str(obs["content"]).strip()
        if timestamp:
            content = f"[{timestamp}] {content}"

        cleaned.append({
            "type": otype,
            "content": content,
            "confidence": min(1.0, max(0.0, float(obs.get("confidence", 0.5)))),
            "quote": f"Visual signal: {signal}",
        })

    logger.info("Visual analysis: %d observations extracted", len(cleaned))
    return cleaned


# --- Multimodal fusion: cross-reference text + visual ---

FUSION_PROMPT = """You are an expert in multimodal sales analysis. You have two sets of insights from the SAME sales call:

1. TEXT INSIGHTS (from transcript — what was SAID):
{text_insights}

2. VISUAL INSIGHTS (from video — what was SEEN):
{visual_insights}

Your job: find CORRELATIONS between what was said and what was seen. Look for:
- Moments where body language CONFIRMS verbal statements (congruence = high trust signal)
- Moments where body language CONTRADICTS verbal statements (incongruence = prospect may be lying or uncertain)
- Emotional shifts in video that align with specific topics in transcript
- Visual engagement peaks/drops that correlate with discussion topics

For each correlation, provide:
- type: "multimodal_confirm" (visual confirms text) or "multimodal_conflict" (visual contradicts text)
- content: clear description of the correlation (2-3 sentences)
- confidence: 0.0-1.0
- quote: summary of the key evidence from both channels
- signal_strength: "strong", "moderate", or "weak"
- action_point: specific action the sales rep should take on the next call (1 sentence, imperative). For CONFIRM: how to leverage. For CONFLICT: how to probe deeper.

Return ONLY valid JSON array. Example:
[
  {
    "type": "multimodal_confirm",
    "content": "When prospect said 'I'm definitely interested' at 05:30, video shows them leaning forward with engaged eye contact.",
    "confidence": 0.9,
    "quote": "Verbal: 'definitely interested' + Visual: forward lean, direct eye contact",
    "signal_strength": "strong",
    "action_point": "Reference their genuine enthusiasm early in the next call to rebuild positive momentum"
  },
  {
    "type": "multimodal_conflict",
    "content": "Prospect verbally agreed to the price but video shows crossed arms and gaze aversion.",
    "confidence": 0.85,
    "quote": "Verbal: 'that works for me' + Visual: crossed arms, looking away",
    "signal_strength": "strong",
    "action_point": "Revisit pricing with a softer approach — ask 'what would make this feel like a no-brainer?' to surface hidden objections"
  }
]

Find 3-8 correlations. Focus on the most actionable ones for the next call."""


def fuse_insights(
    text_insights: list[dict[str, Any]],
    visual_insights: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cross-reference text and visual insights to find multimodal correlations."""
    from google import genai

    if not text_insights or not visual_insights:
        logger.info("Skipping fusion: need both text and visual insights")
        return []

    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Running multimodal fusion: %d text + %d visual...", len(text_insights), len(visual_insights))

    text_summary = json.dumps([
        {"type": i["type"], "content": i["content"], "quote": i.get("quote", "")}
        for i in text_insights
    ], indent=2)

    visual_summary = json.dumps([
        {"type": i["type"], "content": i["content"], "quote": i.get("quote", "")}
        for i in visual_insights
    ], indent=2)

    prompt = FUSION_PROMPT.replace("{text_insights}", text_summary).replace("{visual_insights}", visual_summary)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    raw_text = response.text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    try:
        correlations = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            try:
                correlations = json.loads(match.group())
            except json.JSONDecodeError:
                correlations = []
        else:
            correlations = []

    if not isinstance(correlations, list):
        correlations = []

    valid_types = {"multimodal_confirm", "multimodal_conflict"}
    cleaned = []
    for cor in correlations:
        if not isinstance(cor, dict) or not cor.get("content"):
            continue

        ctype = cor.get("type", "multimodal_confirm")
        if ctype not in valid_types:
            ctype = "multimodal_confirm"

        strength = cor.get("signal_strength", "moderate")
        content = str(cor["content"]).strip()

        cleaned.append({
            "type": ctype,
            "content": f"[{strength.upper()}] {content}",
            "confidence": min(1.0, max(0.0, float(cor.get("confidence", 0.7)))),
            "quote": str(cor.get("quote", "")).strip(),
            "action_point": str(cor.get("action_point", "")).strip(),
        })

    logger.info("Fusion complete: %d multimodal correlations found", len(cleaned))
    return cleaned


# --- Insight extraction ---

INSIGHT_EXTRACTION_PROMPT = """You are an expert sales analyst. Analyze this sales call transcript and extract structured insights.

For each insight, provide:
- type: one of [pain_point, objection, need, decision_maker, budget, timeline, competitor, next_step, sentiment, relationship]
- content: clear, actionable description (1-2 sentences)
- confidence: 0.0-1.0 how confident you are in this insight
- quote: verbatim quote from the transcript that supports this insight (if available)
- action_point: specific action the sales rep should take on the next call based on this insight (1 sentence, imperative)

INSIGHT TYPES:
- pain_point: Problems the prospect is experiencing
- objection: Concerns or pushback raised
- need: Explicit or implicit requirements
- decision_maker: Who makes decisions, budget authority, stakeholders
- budget: Any mention of budget, pricing, cost concerns
- timeline: Urgency, deadlines, implementation timeframe
- competitor: Mentions of competitors or alternative solutions
- next_step: Agreed action items or follow-ups
- sentiment: Overall tone, enthusiasm level, engagement
- relationship: Rapport indicators, personal connections, trust signals

Return ONLY valid JSON array. No markdown, no explanation. Example:
[
  {
    "type": "pain_point",
    "content": "Prospect struggles with manual data entry taking 4 hours daily",
    "confidence": 0.9,
    "quote": "We spend about four hours every day just entering data manually",
    "action_point": "Open next call by asking how many hours they lost to manual entry this week — quantify the cost"
  }
]

Extract at least 5 insights, up to 15. Focus on actionable intelligence for the next call.

TRANSCRIPT:
{transcript}"""


def extract_insights(transcript: str, client_id: str) -> list[dict[str, Any]]:
    """Extract structured sales insights from a transcript using Gemini.

    Returns a list of insight dicts ready for storage.
    """
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = INSIGHT_EXTRACTION_PROMPT.replace("{transcript}", transcript)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    raw_text = response.text.strip()

    # Clean up common Gemini output issues
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    try:
        insights = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse Gemini response as JSON: %s\nRaw: %s", e, raw_text[:500])
        # Attempt to find JSON array in the response
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            try:
                insights = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error("Secondary parse also failed")
                insights = []
        else:
            insights = []

    if not isinstance(insights, list):
        logger.error("Expected list of insights, got: %s", type(insights))
        insights = []

    # Validate and clean each insight
    valid_types = {
        "pain_point", "objection", "need", "decision_maker", "budget",
        "timeline", "competitor", "next_step", "sentiment", "relationship",
        "body_language", "facial_expression", "engagement_shift", "visual_cue", "environment",
        "multimodal_confirm", "multimodal_conflict",
    }
    cleaned = []
    for insight in insights:
        if not isinstance(insight, dict):
            continue
        if not insight.get("content"):
            continue

        itype = insight.get("type", "insight")
        if itype not in valid_types:
            itype = "insight"

        cleaned.append({
            "type": itype,
            "content": str(insight["content"]).strip(),
            "confidence": min(1.0, max(0.0, float(insight.get("confidence", 0.5)))),
            "quote": str(insight.get("quote", "")).strip(),
            "action_point": str(insight.get("action_point", "")).strip(),
        })

    logger.info("Extracted %d insights for client '%s'", len(cleaned), client_id)
    return cleaned


# --- Full pipeline orchestration ---


def process_video(
    file_path_or_url: str,
    client_id: str,
    call_date: str = "",
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run the full pipeline: download (if URL) → transcribe → extract → store.

    Returns a summary of what was processed and stored.
    """
    from . import memory

    if job_id is None:
        job_id = str(uuid.uuid4())[:8]

    _jobs[job_id] = {
        "job_id": job_id,
        "client_id": client_id,
        "status": "starting",
        "progress": 0,
        "source": file_path_or_url,
        "error": None,
        "result": None,
    }

    try:
        # Step 1: Get the video file
        _update_job(job_id, status="downloading", progress=10)

        if file_path_or_url.startswith(("http://", "https://")):
            video_path = download_youtube(file_path_or_url)
            source_label = file_path_or_url
        else:
            video_path = Path(file_path_or_url)
            source_label = video_path.name

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Step 2: Upload to Gemini once
        _update_job(job_id, status="uploading", progress=20)
        video_file = _upload_and_wait(video_path)

        # Step 3: Run transcript + visual analysis IN PARALLEL
        _update_job(job_id, status="analyzing", progress=35)
        logger.info("Running parallel analysis: transcript + visual...")

        transcript = ""
        visual_insights = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_transcript = executor.submit(transcribe_video, video_file)
            future_visual = executor.submit(analyze_video_visuals, video_file)

            transcript = future_transcript.result()
            try:
                visual_insights = future_visual.result()
            except Exception as e:
                logger.warning("Visual analysis failed (continuing with transcript only): %s", e)

        # Save transcript
        transcript_path = UPLOAD_DIR / f"{job_id}_transcript.txt"
        transcript_path.write_text(transcript)

        _update_job(job_id, status="extracting", progress=65)

        # Step 4: Extract text-based insights from transcript
        text_insights = extract_insights(transcript, client_id)

        # Step 5: Multimodal fusion — cross-reference text + visual
        _update_job(job_id, status="fusing", progress=75)
        fusion_insights = []
        try:
            fusion_insights = fuse_insights(text_insights, visual_insights)
        except Exception as e:
            logger.warning("Fusion failed (continuing without): %s", e)

        # Step 6: Merge all three channels
        all_insights = text_insights + visual_insights + fusion_insights
        logger.info(
            "Merged insights: %d text + %d visual + %d fusion = %d total",
            len(text_insights), len(visual_insights), len(fusion_insights), len(all_insights),
        )

        _update_job(job_id, status="storing", progress=85)

        # Step 6: Store in Qdrant
        stored_ids = memory.store_insights(
            insights=all_insights,
            client_id=client_id,
            source_video=source_label,
            call_date=call_date,
        )

        result = {
            "job_id": job_id,
            "client_id": client_id,
            "source": source_label,
            "transcript_length": len(transcript),
            "insights_count": len(all_insights),
            "text_insights": len(text_insights),
            "visual_insights": len(visual_insights),
            "fusion_insights": len(fusion_insights),
            "stored_ids": stored_ids,
            "insights": all_insights,
        }

        _update_job(job_id, status="completed", progress=100, result=result)
        logger.info(
            "Pipeline complete for '%s': %d text + %d visual + %d fusion = %d total",
            client_id, len(text_insights), len(visual_insights),
            len(fusion_insights), len(all_insights),
        )
        return result

    except Exception as e:
        logger.exception("Pipeline failed for job %s: %s", job_id, e)
        _update_job(job_id, status="failed", error=str(e))
        raise
