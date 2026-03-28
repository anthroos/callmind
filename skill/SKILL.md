---
name: callmind
description: Video intelligence with Q-learning — analyze sales calls with 3-channel Gemini multimodal analysis, store insights in Qdrant with Q-learning feedback
license: MIT
compatibility: Requires a running CallMind server (localhost:8000 or remote). Needs curl or httpx for API calls.
metadata:
  author: anthroos
  version: "0.1.0"
---

# CallMind — Video Intelligence Skill

Use this skill to analyze sales call recordings and retrieve Q-learning-ranked insights via the CallMind API.

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

Retrieve insights ranked by Q-value (hybrid: 60% semantic + 40% Q-value):

```bash
curl "$CALLMIND_URL/api/client/{client_id}/insights" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

Each insight includes:
- `type`: pain_point, objection, need, decision_maker, budget, timeline, competitor, next_step, sentiment, relationship
- `channel`: text (transcript), visual (body language), fusion (cross-modal)
- `q_value`: learned importance score (-0.5 to 1.0)
- `content`: the insight text
- `action_point`: concrete recommended action

## Q-Learning

Insights learn from deal outcomes. After a deal closes:
- **Won** → reward +1.0, all insights Q-values increase
- **Lost** → reward -1.0, Q-values decrease
- **Progressed** → reward +0.5, moderate boost

High Q-value insights float to the top of pre-call briefings.

## Workflow Pattern

1. Upload sales call recording after each client meeting
2. Review insights in client dashboard or via API
3. Before next call, get pre-call briefing (top insights by Q-value)
4. After deal outcome, record result to update Q-values
5. Over time, the system learns which signals predict deal success
