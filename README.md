# CallMind: what the call said, and what the faces showed

Upload a recording of a video call. CallMind returns:
- **Transcript insights:** pain points, objections, budget, timeline, next steps, each with a quote and an action point.
- **Non-verbal and emotion signals:** facial expressions, body language and engagement shifts, each with a timestamp, positive/negative/neutral signal and confidence.
- **Said vs shown:** moments where words and body language agree (`multimodal_confirm`) or contradict each other (`multimodal_conflict`, e.g. "agreed to the price, but crossed arms and looked away").
- **A prep briefing for the next call**, plus semantic search across all analyzed calls.

Built in one night at the Multimodal Frontier Hackathon (San Francisco, March 2026). The hackathon demo video for v0.1 is [on Loom](https://www.loom.com/share/09f4252ff673466da4a1a08388de6aa5).

## How it works

```
video file / YouTube URL
   └─► upload once to the Gemini Files API
         ├─► Channel 1 · Transcript   speakers + timestamps → structured insights   ┐ run in
         └─► Channel 2 · Visual       expressions, posture, engagement, timestamps  ┘ parallel
                    └─► Channel 3 · Fusion: cross-reference said vs shown
   └─► video deleted from Gemini
   └─► insights embedded locally (FastEmbed, bge-small) → Qdrant
   └─► web UI: client dashboard · call prep · search
```

All three channels use **Gemini 2.5 Flash** with plain prompts (`callmind/video_pipeline.py`). There is no separate face-detection or emotion-classification model: Gemini watches the video natively and describes what it sees.

## Quick start (≈5 minutes)

You need Docker and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/anthroos/callmind.git
cd callmind
cp .env.example .env        # put your GEMINI_API_KEY in it
docker compose up --build
```
Open http://localhost:8000 and upload a video.

Without Docker (Python 3.11+, with Qdrant running on `localhost:6333`):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m callmind.app
```

## API

```bash
# Upload a video (file or YouTube URL)
curl -X POST http://localhost:8000/api/upload \
  -F "client_name=Acme Corp" -F "video_file=@recording.mp4"

# Poll the job
curl http://localhost:8000/api/status/<job_id>

# Read the insights (add ?q=... for semantic search)
curl http://localhost:8000/api/client/acme_corp/insights
```
If `UNKEY_ROOT_KEY` is set, `/api/upload` and `/api/client/*/insights` require `Authorization: Bearer <key>`. Users get keys from `POST /register`.

## Limits: read before using it on real people

- **The emotion signals are an LLM's reading of the video, not a measurement.** Nobody has checked their accuracy against labelled data. Gemini can over-read "micro-expressions". Each signal carries a `confidence` and a timestamp, so check those moments yourself.
- **Consent.** Only analyze recordings where the people on screen know about it and agree. Emotion recognition is legally restricted in some places: the EU AI Act prohibits it in workplace and education settings. Personal and research use with consent is the intended use.
- **The video goes to Google's Gemini API.** CallMind deletes it from Gemini Files after analysis. A local copy stays in `uploads/`, which you can delete.
- **Not hardened for the public internet.** It listens on `127.0.0.1` by default, and Qdrant is not exposed. The web UI and the `/api/memory/*` routes have no login. Before deploying it publicly, put it behind an authenticating proxy.
- Hackathon code: in-memory job tracking (lost on restart) and no test suite yet.

## Tech stack

| | |
|---|---|
| Video understanding | Gemini 2.5 Flash (native multimodal video) |
| Embeddings | FastEmbed, `BAAI/bge-small-en-v1.5` (384-dim, runs locally) |
| Vector store | Qdrant |
| API keys (optional) | Unkey |
| Web | FastAPI + Jinja2 |
| Video download | yt-dlp |

## License

MIT, see [LICENSE](LICENSE).

Built by [Ivan Pasichnyk](https://github.com/anthroos).
