# CallMind — Devpost Submission

## Title
CallMind: Video Intelligence with Q-Learning

## Tagline
3-channel multimodal video analysis. Insights flow into vector memory. Q-learning ranks what closes deals.

## Inspiration
Sales teams lose critical context between calls. Hours of video recordings pile up unwatched. We built CallMind to watch sales calls with AI eyes — not just transcribing words, but reading body language, detecting tone shifts, and cross-referencing verbal and non-verbal signals.

## What it does
CallMind analyzes sales call recordings through **3 parallel AI channels**:
1. **Transcript Channel** — Full speech-to-text with structured insight extraction (pain points, objections, budget signals, decision makers)
2. **Visual Channel** — Body language analysis, facial expressions, engagement patterns, micro-expressions
3. **Fusion Channel** — Cross-modal correlations (e.g., "said yes but crossed arms and leaned back")

All insights are embedded and stored in **Qdrant vector memory** with **Q-learning feedback**. Record deal outcomes (won/lost/progressed) and Q-values update — over time, the system learns which signals actually predict deal success. Your pre-call briefing becomes a **ranked intelligence report**.

## How we built it
- **Gemini 2.5 Flash** — Native multimodal video understanding via Files API. Upload once, query 3× in parallel. No frame extraction — Gemini sees motion, gestures, expressions natively.
- **Qdrant** — Vector database with hybrid scoring: 60% semantic similarity + 40% Q-value. FastEmbed (BAAI/bge-small-en-v1.5) for local embeddings.
- **Q-Learning** — `Q_new = clamp(Q_old + 0.25 × reward, -0.5, 1.0)`. Deal outcomes propagate through all associated insights. Over multiple deals, empirically valuable signals rise to the top.
- **Unkey** — API key management for programmatic access. Register via web UI, get a key instantly, authenticate all API calls.
- **FastAPI + Jinja2** — Lightweight web framework with server-side rendering.
- **DigitalOcean** — Production deployment (Docker Compose: Qdrant + CallMind).

## Challenges we ran into
- **Unkey SDK vs. pip package conflict** — The `unkey` pip package is actually a Python linting tool that removes extra `.keys()` calls, NOT the Unkey API SDK. Had to rewrite to use plain HTTP (httpx) against the v2 REST API.
- **Gemini structured output** — Getting reliable JSON from multimodal video analysis required careful prompt engineering. Each channel has a distinct system prompt with strict output schema.
- **Cross-modal fusion** — The most valuable insights come from *contradictions* between channels (verbal positive + body language negative). Designing the fusion prompt to detect these was the most challenging part.

## Accomplishments
- **3-channel parallel analysis** running on a single uploaded video — no frame extraction, pure multimodal
- **Q-learning feedback loop** that actually differentiates insights over time (demo shows positive vs. negative Q clients)
- **Full working pipeline**: upload → analyze → store → learn → prep briefing
- **REST API with Unkey auth** for connecting to any external tool (Claude Code, CRM, etc.)

## What we learned
- Gemini's native video understanding is remarkably good at body language detection — it notices things humans miss
- Q-learning with just alpha=0.25 creates meaningful signal differentiation after just 2-3 deal outcomes
- The cross-modal fusion channel produces the highest-value insights (verbal-visual contradictions)

## What's next
- **MCP server** — so Claude Code and other AI agents can query CallMind natively
- **Live call analysis** — real-time processing during calls
- **Multi-source fusion** — CRM data, emails, LinkedIn activity combined with video insights in the same Q-learning loop
- **Team-level learning** — Q-values shared across reps to build institutional sales intelligence

## Built with
Gemini 2.5 Flash, Qdrant, FastEmbed, Unkey, DigitalOcean, FastAPI, Python, Docker

## Links
- **GitHub**: https://github.com/anthroos/callmind
- **Live Demo**: [TODO: DigitalOcean URL]
- **Video Demo**: [TODO]
