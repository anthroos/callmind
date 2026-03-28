"""CallMind configuration — all settings from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Gemini ---
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# --- Qdrant ---
QDRANT_HOST: str = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_API_KEY: str | None = os.environ.get("QDRANT_API_KEY", "").strip() or None
COLLECTION_NAME: str = os.environ.get("CALLMIND_COLLECTION", "callmind_memories")

# --- Embedding ---
EMBEDDING_MODEL: str = os.environ.get("CALLMIND_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM: int = int(os.environ.get("CALLMIND_EMBEDDING_DIM", "384"))

# --- File storage ---
UPLOAD_DIR: Path = Path(os.environ.get("CALLMIND_UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# --- Q-learning defaults (same as OpenExp) ---
Q_INIT: float = 0.0
Q_ALPHA: float = 0.25
Q_FLOOR: float = -0.5
Q_CEILING: float = 1.0

# --- Unkey ---
UNKEY_ROOT_KEY: str = os.environ.get("UNKEY_ROOT_KEY", "")
UNKEY_API_ID: str = os.environ.get("UNKEY_API_ID", "")

# --- App ---
APP_HOST: str = os.environ.get("CALLMIND_HOST", "0.0.0.0")
APP_PORT: int = int(os.environ.get("CALLMIND_PORT", "8000"))
