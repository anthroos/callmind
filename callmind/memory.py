"""CallMind memory layer — Qdrant + FastEmbed.

Each insight is stored as a vector with client_id filtering. Search is semantic.
"""

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
    QDRANT_API_KEY,
    QDRANT_HOST,
    QDRANT_PORT,
)

logger = logging.getLogger(__name__)

# --- Singletons (thread-safe lazy init, same pattern as OpenExp) ---

_init_lock = threading.Lock()
_embedder: Optional[TextEmbedding] = None
_qdrant: Optional[QdrantClient] = None


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

        stored_ids.append(point_id)

    if points:
        qc.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info("Stored %d insights for client '%s'", len(points), client_id)

    return stored_ids


def get_client_insights(
    client_id: str,
    query: str = "",
    limit: int = 20,
    insight_type: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve insights for a client.

    With a query: semantic search, ranked by similarity. Without: newest first.
    """
    qc = _get_qdrant()

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
            })

    if query:
        results.sort(key=lambda x: x["vector_score"], reverse=True)
    else:
        results.sort(key=lambda x: x["created_at"], reverse=True)
    return results[:limit]


def get_all_clients() -> list[dict[str, Any]]:
    """Get a list of all unique clients with insight counts."""
    qc = _get_qdrant()

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
                }

            clients[cid]["insight_count"] += 1
            call_date = payload.get("call_date", "")
            if call_date > clients[cid]["latest_call"]:
                clients[cid]["latest_call"] = call_date


        if next_offset is None:
            break
        offset = next_offset

    result = list(clients.values())

    result.sort(key=lambda x: x["latest_call"], reverse=True)
    return result


def get_call_prep(client_id: str) -> dict[str, Any]:
    """Generate a call prep briefing for a client.

    Returns insights organized by category, plus the top insights by extraction confidence.
    """

    # Get all insights
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

    # Top 3 insights across all categories, by extraction confidence
    top_insights = sorted(all_insights, key=lambda x: x["confidence"], reverse=True)[:3]

    return {
        "client_id": client_id,
        "total_insights": len(all_insights),
        "sections": ordered_sections,
        "top_insights": top_insights,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def search_all(query: str, limit: int = 20, source: str = "") -> list[dict[str, Any]]:
    """Search across all clients and sources, ranked by semantic similarity."""
    qc = _get_qdrant()

    query_vector = _embed(query)

    # Optional source filter
    qdrant_filter = None
    if source:
        qdrant_filter = Filter(must=[
            FieldCondition(key="insight_type", match=MatchValue(value=source)),
        ])

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
        results.append({
            "id": str(point.id),
            "content": payload.get("memory", ""),
            "type": payload.get("insight_type", "insight"),
            "client_id": payload.get("client_id", ""),
            "source": payload.get("source_video", "callmind"),
            "call_date": payload.get("call_date", ""),
            "vector_score": round(point.score, 3),
            "action_point": payload.get("action_point", ""),
        })

    return results[:limit]


def add_memory(content: str, memory_type: str = "note", client_id: str = "global", source: str = "manual") -> str:
    """Add a single note to the knowledge base."""
    qc = _get_qdrant()

    vector = _embed(f"[{memory_type}] {content}")
    point_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    payload = {
        "memory": content,
        "client_id": client_id,
        "insight_type": memory_type,
        "confidence": 0.8,
        "source_video": source,
        "call_date": now[:10],
        "created_at": now,
        "status": "active",
    }

    qc.upsert(collection_name=COLLECTION_NAME, points=[
        PointStruct(id=point_id, vector=vector, payload=payload),
    ])


    return point_id


def get_stats() -> dict[str, Any]:
    """Get knowledge base statistics from Qdrant."""
    qc = _get_qdrant()
    total = 0
    types: dict[str, int] = {}
    clients: set[str] = set()
    offset = None
    while True:
        points, offset = qc.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=["client_id", "insight_type"],
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            total += 1
            t = payload.get("insight_type", "unknown")
            types[t] = types.get(t, 0) + 1
            clients.add(payload.get("client_id", "unknown"))
        if offset is None:
            break

    return {
        "total_memories": total,
        "total_clients": len(clients),
        "types": types,
    }


def get_insight_by_id(insight_id: str) -> dict[str, Any] | None:
    """Get a single insight by its Qdrant point ID."""
    qc = _get_qdrant()

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
        }
    except Exception as e:
        logger.error("Failed to retrieve insight %s: %s", insight_id, e)
        return None
