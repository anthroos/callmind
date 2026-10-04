---
name: callmind
description: Video call intelligence — analyze call recordings with 3-channel Gemini multimodal analysis (transcript, body language and emotion signals, said-vs-shown fusion) and search the insights in Qdrant
license: MIT
compatibility: Requires a running CallMind server (localhost:8000 or remote). Needs curl or httpx for API calls.
metadata:
  author: anthroos
  version: "0.2.0"
---

# CallMind — Video Intelligence Skill

Use this skill to analyze call recordings and retrieve structured insights via the CallMind API.

## Setup

CallMind must be running (locally or remote). Set the base URL:

```
CALLMIND_URL=http://localhost:8000
```

Get an API key by registering:
```bash
curl -X POST $CALLMIND_URL/register -d "username=your_name"
```

## Upload a Video for Analysis

Upload a video file or YouTube URL for 3-channel multimodal analysis:

```bash
# File upload
curl -X POST $CALLMIND_URL/api/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "client_name=Acme Corp" \
  -F "video_file=@recording.mp4"

# YouTube URL
curl -X POST $CALLMIND_URL/api/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "client_name=Acme Corp" \
  -F "youtube_url=https://youtube.com/watch?v=..."
```

Response includes a `job_id`. Poll status:
```bash
curl $CALLMIND_URL/api/status/{job_id}
```

## Get Client Insights

Retrieve insights (newest first, or ranked by semantic similarity with `?q=`):

```bash
curl "$CALLMIND_URL/api/client/{client_id}/insights" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Each insight includes:
- `type`: pain_point, objection, need, decision_maker, budget, timeline, competitor, next_step, sentiment, relationship
- `channel`: text (transcript), visual (body language), fusion (cross-modal)
- `content`: the insight text
- `action_point`: concrete recommended action

## Workflow Pattern

1. Upload sales call recording after each client meeting
2. Review insights in client dashboard or via API
3. Before the next call, open the pre-call briefing (`/client/{client_id}/prep`)
