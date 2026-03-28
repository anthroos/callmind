# CallMind — Video Intelligence with Q-Learning

**3-channel multimodal video analysis. Insights flow into vector memory. Q-learning ranks what closes deals.**

Built at the [Multimodal Frontier Hackathon](https://lu.ma/multimodal-frontier) (YC x Google DeepMind, March 2026).

## The Problem

Sales teams lose critical context between calls. Hours of video recordings pile up unwatched. The intelligence trapped in body language, tone shifts, and verbal cues never reaches the next call prep.

## The Solution

CallMind analyzes sales call recordings through **3 parallel AI channels**, extracts structured insights, and stores them in a vector knowledge base with **Q-learning feedback**.

The key: **insights get smarter over time**. Record deal outcomes, and Q-learning updates the value of every insight. Over multiple deals, the system learns which signals actually predict success.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Video/Audio │────▶│  CallMind Engine  │────▶│   API / MCP     │
│  (Upload/URL)│     │                  │     │                 │
└─────────────┘     │  Channel 1: Text  │     │  REST + Unkey   │
                    │  Channel 2: Visual│     │  Connect to any │
                    │  Channel 3: Fusion│     │  AI agent / CRM │
                    │                  │     └─────────────────┘
                    │  Qdrant + Q-Learn │
                    └──────────────────┘
                             ▲
                             │
                    Deal Outcomes (reward)
```

CallMind is **one service/block** in a larger AI stack. Qdrant with Q-learning sits at the core. CRM, email, documents — all connect through API or MCP. Claude Code (or any AI agent) orchestrates.

## 3-Channel Parallel Analysis

| Channel | What it does | Model |
|---------|-------------|-------|
| **Transcript** | Full speech-to-text, speaker ID, structured insight extraction | Gemini 2.5 Flash |
| **Visual** | Body language, facial expressions, engagement, micro-expressions | Gemini 2.5 Flash |
| **Fusion** | Cross-modal correlations (e.g., "said yes but leaned back") | Gemini 2.5 Flash |

All 3 channels run **in parallel** on the same uploaded video via Gemini's native multimodal capabilities.

## Q-Learning

Every insight gets a Q-value. When you record deal outcomes:

```
Q_new = clamp(Q_old + 0.25 × reward, -0.5, 1.0)
```

- **Deal won** → +1.0 reward → insights move up
- **Deal lost** → -1.0 reward → insights move down
- **Progressed** → +0.5 → moderate boost

Over time, high-Q insights float to the top of your pre-call briefing. The system learns what matters empirically.

## Sponsor Integration

### Gemini (Google DeepMind)
- **Gemini 2.5 Flash** multimodal — video uploaded via Files API, then queried 3× in parallel
- Native video understanding (not frame extraction) — sees motion, gestures, expressions
- Structured JSON output for reliable insight extraction

### Unkey
- **API key management** — users register via web UI, get API keys instantly
- **Key verification** on every API request (v2 REST API)
- Enables programmatic access: upload videos, fetch insights, integrate with any tool

### DigitalOcean
- Production deployment target (Qdrant + FastAPI app)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Video Understanding | Gemini 2.5 Flash (3-channel multimodal) |
| Embeddings | FastEmbed (BAAI/bge-small-en-v1.5, 384-dim, local) |
| Vector Store | Qdrant (hybrid: 60% vector + 40% Q-value) |
| Q-Learning | Custom (alpha=0.25, range [-0.5, 1.0]) |
| API Auth | Unkey (v2 REST API) |
| Web Framework | FastAPI + Jinja2 |
| Video Download | yt-dlp |
| Language | Python 3.11+ |

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/anthroos/callmind.git
cd callmind
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# 3. Configure
cp .env.example .env
# Edit .env with your GEMINI_API_KEY, UNKEY_API_ID, UNKEY_ROOT_KEY

# 4. Run
python -m callmind.app
# → http://localhost:8000
```

## API Usage

```bash
# Register for an API key (or use the web UI)
curl -X POST http://localhost:8000/register \
  -d "username=your_name"

# Upload a video
curl -X POST http://localhost:8000/api/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "client_name=Acme Corp" \
  -F "video_file=@recording.mp4"

# Get insights (ranked by Q-value)
curl http://localhost:8000/api/client/acme_corp/insights \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Built On OpenExp

[OpenExp](https://github.com/anthroos/openexp) — open-source Q-learning memory for AI agents. Same core: Qdrant + FastEmbed + Q-values. CallMind extends it with multimodal video intelligence.

## License

MIT

---

Built by [Ivan Pasichnyk](https://github.com/anthroos) at the Multimodal Frontier Hackathon (YC x Google DeepMind, March 2026).
