"""CallMind memory layer — Qdrant + FastEmbed + Q-learning.

Same architecture as OpenExp but with a dedicated collection for sales call insights.
Each insight is stored as a vector with client_id filtering and Q-value tracking.
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from .config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    Q_ALPHA,
    Q_CEILING,
    Q_FLOOR,
    Q_INIT,
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
    UPLOAD_DIR,
)

logger = logging.getLogger(__name__)

# --- Singletons (thread-safe lazy init, same pattern as OpenExp) ---

_init_lock = threading.Lock()
_embedder: Optional[TextEmbedding] = None
_qdrant: Optional[QdrantClient] = None

# Local Q-cache file (lightweight, same concept as OpenExp's QCache)
_Q_CACHE_PATH = UPLOAD_DIR.parent / "q_cache.json"
_q_cache: dict[str, dict] = {}


def _get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        with _init_lock:
            if _embedder is None:
                cache_dir = str(Path.home() / ".cache" / "fastembed")
                _embedder = TextEmbedding(model_name=EMBEDDING_MODEL, cache_dir=cache_dir)
                logger.info("FastEmbed model loaded: %s", EMBEDDING_MODEL)
    return _embedder


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        with _init_lock:
            if _qdrant is None:
                _qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
    return _qdrant


def _embed(text: str) -> list[float]:
    """Embed a single text string."""
    embedder = _get_embedder()
    vectors = list(embedder.embed([text]))
    return vectors[0].tolist()


def _load_q_cache() -> dict[str, dict]:
    """Load Q-cache from disk."""
    global _q_cache
    if _Q_CACHE_PATH.exists():
        try:
            _q_cache = json.loads(_Q_CACHE_PATH.read_text())
        except Exception:
            _q_cache = {}
    return _q_cache


def _save_q_cache() -> None:
    """Persist Q-cache to disk."""
    _Q_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _Q_CACHE_PATH.write_text(json.dumps(_q_cache, indent=2))


def _clamp_q(value: float) -> float:
    return max(Q_FLOOR, min(Q_CEILING, value))


# --- Collection setup ---


def ensure_collection() -> None:
    """Create the Qdrant collection if it doesn't exist."""
    qc = _get_qdrant()
    try:
        qc.get_collection(COLLECTION_NAME)
        logger.info("Collection '%s' already exists", COLLECTION_NAME)
    except (UnexpectedResponse, Exception):
        qc.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        # Create payload indices for filtering
        qc.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="client_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        qc.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="insight_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        qc.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="source_video",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Created collection '%s' with indices", COLLECTION_NAME)


# --- Core operations ---


def store_insights(
    insights: list[dict[str, Any]],
    client_id: str,
    source_video: str = "",
    call_date: str = "",
) -> list[str]:
    """Store extracted insights into Qdrant with embeddings.

    Each insight dict should have:
        - type: str (objection, need, decision_maker, budget, timeline, pain_point,
                     competitor, next_step, sentiment, relationship)
        - content: str (the actual insight text)
        - confidence: float (0-1, how confident the extraction is)
        - quote: str (optional, verbatim quote from transcript)

    Returns list of stored point IDs.
    """
    qc = _get_qdrant()
    _load_q_cache()

    points = []
    stored_ids = []
    now = datetime.now(timezone.utc).isoformat()

    for insight in insights:
        content = insight.get("content", "")
        if not content.strip():
            continue

        # Build embedding text: type prefix + content for better retrieval
        embed_text = f"[{insight.get('type', 'insight')}] {content}"
        vector = _embed(embed_text)
        point_id = str(uuid.uuid4())

        payload = {
            "memory": content,
            "client_id": client_id,
            "insight_type": insight.get("type", "insight"),
            "confidence": insight.get("confidence", 0.5),
            "quote": insight.get("quote", ""),
            "action_point": insight.get("action_point", ""),
            "source_video": source_video,
            "call_date": call_date or now[:10],
            "created_at": now,
            "status": "active",
        }

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        # Initialize Q-value
        _q_cache[point_id] = {
            "q_value": Q_INIT,
            "q_visits": 0,
            "client_id": client_id,
            "insight_type": insight.get("type", "insight"),
        }
        stored_ids.append(point_id)

    if points:
        qc.upsert(collection_name=COLLECTION_NAME, points=points)
        _save_q_cache()
        logger.info("Stored %d insights for client '%s'", len(points), client_id)

    return stored_ids


def get_client_insights(
    client_id: str,
    query: str = "",
    limit: int = 20,
    insight_type: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve insights for a client, ranked by relevance + Q-value.

    If query is provided, does semantic search. Otherwise returns recent insights.
    """
    qc = _get_qdrant()
    _load_q_cache()

    # Build filter
    must_conditions = [
        FieldCondition(key="client_id", match=MatchValue(value=client_id)),
    ]
    if insight_type:
        must_conditions.append(
            FieldCondition(key="insight_type", match=MatchValue(value=insight_type)),
        )

    qdrant_filter = Filter(must=must_conditions)

    if query:
        # Semantic search
        query_vector = _embed(query)
        search_result = qc.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit * 2,
            with_payload=True,
        )
        results = []
        for point in search_result.points:
            payload = point.payload or {}
            q_data = _q_cache.get(str(point.id), {"q_value": Q_INIT, "q_visits": 0})
            results.append({
                "id": str(point.id),
                "content": payload.get("memory", ""),
                "type": payload.get("insight_type", "insight"),
                "confidence": payload.get("confidence", 0.5),
                "quote": payload.get("quote", ""),
                "action_point": payload.get("action_point", ""),
                "source_video": payload.get("source_video", ""),
                "call_date": payload.get("call_date", ""),
                "created_at": payload.get("created_at", ""),
                "vector_score": point.score,
                "q_value": q_data.get("q_value", Q_INIT),
                "q_visits": q_data.get("q_visits", 0),
            })
    else:
        # Scroll all insights for this client (no query vector needed)
        scroll_result = qc.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qdrant_filter,
            limit=limit * 2,
            with_payload=True,
            with_vectors=False,
        )
        results = []
        for point in scroll_result[0]:
            payload = point.payload or {}
            q_data = _q_cache.get(str(point.id), {"q_value": Q_INIT, "q_visits": 0})
            results.append({
                "id": str(point.id),
                "content": payload.get("memory", ""),
                "type": payload.get("insight_type", "insight"),
                "confidence": payload.get("confidence", 0.5),
                "quote": payload.get("quote", ""),
                "action_point": payload.get("action_point", ""),
                "source_video": payload.get("source_video", ""),
                "call_date": payload.get("call_date", ""),
                "created_at": payload.get("created_at", ""),
                "vector_score": 0.0,
                "q_value": q_data.get("q_value", Q_INIT),
                "q_visits": q_data.get("q_visits", 0),
            })

    # Hybrid ranking: combine vector score (if available) with Q-value
    # Q-value weight increases as the system learns
    for r in results:
        q_weight = 0.4
        vec_weight = 0.6
        r["hybrid_score"] = (vec_weight * r["vector_score"]) + (q_weight * max(0, r["q_value"]))

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return results[:limit]


def get_all_clients() -> list[dict[str, Any]]:
    """Get a list of all unique clients with insight counts."""
    qc = _get_qdrant()
    _load_q_cache()

    # Scroll through all points to collect client_ids
    clients: dict[str, dict] = {}
    offset = None
    while True:
        scroll_result = qc.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        points, next_offset = scroll_result

        for point in points:
            payload = point.payload or {}
            cid = payload.get("client_id", "unknown")
            if cid not in clients:
                clients[cid] = {
                    "client_id": cid,
                    "insight_count": 0,
                    "latest_call": "",
                    "avg_q_value": 0.0,
                    "q_values": [],
                }

            clients[cid]["insight_count"] += 1
            call_date = payload.get("call_date", "")
            if call_date > clients[cid]["latest_call"]:
                clients[cid]["latest_call"] = call_date

            q_data = _q_cache.get(str(point.id), {"q_value": Q_INIT})
            clients[cid]["q_values"].append(q_data.get("q_value", Q_INIT))

        if next_offset is None:
            break
        offset = next_offset

    # Calculate averages
    result = []
    for cid, data in clients.items():
        q_vals = data.pop("q_values", [])
        data["avg_q_value"] = round(sum(q_vals) / len(q_vals), 3) if q_vals else 0.0
        result.append(data)

    result.sort(key=lambda x: x["latest_call"], reverse=True)
    return result


def get_call_prep(client_id: str) -> dict[str, Any]:
    """Generate a call prep briefing for a client.

    Returns top insights organized by category, ranked by Q-value.
    High Q-value = historically important for deals. Low Q-value = noise.
    """
    _load_q_cache()

    # Get all insights, ranked by Q-value
    all_insights = get_client_insights(client_id, limit=50)

    # Organize by type
    categorized: dict[str, list] = {}
    for insight in all_insights:
        itype = insight["type"]
        if itype not in categorized:
            categorized[itype] = []
        categorized[itype].append(insight)

    # Priority order for sales prep
    type_priority = [
        "pain_point",
        "objection",
        "decision_maker",
        "budget",
        "timeline",
        "need",
        "competitor",
        "next_step",
        "sentiment",
        "relationship",
    ]

    ordered_sections = []
    for t in type_priority:
        if t in categorized:
            ordered_sections.append({
                "type": t,
                "label": t.replace("_", " ").title(),
                "insights": categorized[t][:5],  # Top 5 per category
            })

    # Add any types not in priority list
    for t, items in categorized.items():
        if t not in type_priority:
            ordered_sections.append({
                "type": t,
                "label": t.replace("_", " ").title(),
                "insights": items[:5],
            })

    # Top 3 most important insights across all categories (by Q-value)
    top_insights = sorted(all_insights, key=lambda x: x["q_value"], reverse=True)[:3]

    return {
        "client_id": client_id,
        "total_insights": len(all_insights),
        "sections": ordered_sections,
        "top_insights": top_insights,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def update_q_values(client_id: str, outcome: str, reward: float) -> dict[str, Any]:
    """Apply reward signal to all insights for a client.

    Simulates a deal outcome:
        reward > 0: deal progressed/won — insights that predicted this are valuable
        reward < 0: deal lost/stalled — insights may have been misleading
        reward = 0: neutral, no update

    Uses same Q-learning formula as OpenExp:
        Q_new = clamp(Q_old + alpha * reward, floor, ceiling)
    """
    _load_q_cache()

    updated = 0
    for point_id, q_data in _q_cache.items():
        if q_data.get("client_id") == client_id:
            old_q = q_data.get("q_value", Q_INIT)
            new_q = _clamp_q(old_q + Q_ALPHA * reward)
            q_data["q_value"] = round(new_q, 4)
            q_data["q_visits"] = q_data.get("q_visits", 0) + 1
            q_data["last_outcome"] = outcome
            q_data["last_reward"] = reward
            updated += 1

    _save_q_cache()

    return {
        "client_id": client_id,
        "outcome": outcome,
        "reward": reward,
        "insights_updated": updated,
    }


def get_insight_by_id(insight_id: str) -> dict[str, Any] | None:
    """Get a single insight by its Qdrant point ID."""
    qc = _get_qdrant()
    _load_q_cache()

    try:
        points = qc.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[insight_id],
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None

        point = points[0]
        payload = point.payload or {}
        q_data = _q_cache.get(str(point.id), {"q_value": Q_INIT, "q_visits": 0})

        return {
            "id": str(point.id),
            "content": payload.get("memory", ""),
            "type": payload.get("insight_type", "insight"),
            "confidence": payload.get("confidence", 0.5),
            "quote": payload.get("quote", ""),
            "source_video": payload.get("source_video", ""),
            "call_date": payload.get("call_date", ""),
            "created_at": payload.get("created_at", ""),
            "client_id": payload.get("client_id", ""),
            "q_value": q_data.get("q_value", Q_INIT),
            "q_visits": q_data.get("q_visits", 0),
        }
    except Exception as e:
        logger.error("Failed to retrieve insight %s: %s", insight_id, e)
        return None
