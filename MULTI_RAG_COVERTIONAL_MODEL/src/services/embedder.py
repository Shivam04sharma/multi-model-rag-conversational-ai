"""
Embedder — converts text to vector embeddings via Vertex AI text-embedding-004 (768-dim).

Query embedding performance is critical (every search calls it), so all embeddings
are cached in a process-local LRU cache keyed on the input text.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import structlog
from config import settings

logger = structlog.get_logger()

_GEMINI_MODEL = "text-embedding-004"
_GEMINI_DIMS = 768

_genai_client = None  # lazy-loaded google.genai Client


_embed_cache: dict[str, list[float]] = {}
_embed_access_order: list[str] = []
_CACHE_MAX_SIZE = 500
_CACHE_LOCK = asyncio.Lock()


def _cache_key(text: str) -> str:
    """Stable cache key: normalize whitespace/case."""
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()[:32]


def _cache_get(key: str) -> list[float] | None:
    """Thread-safe LRU cache read."""
    if key not in _embed_cache:
        return None
    # Move to end (most recently used)
    _embed_access_order.remove(key)
    _embed_access_order.append(key)
    return _embed_cache[key]


def _cache_put(key: str, vector: list[float]) -> None:
    """Thread-safe LRU cache write with 500-item cap."""
    if key in _embed_cache:
        _embed_access_order.remove(key)
    elif len(_embed_cache) >= _CACHE_MAX_SIZE:
        oldest = _embed_access_order.pop(0)
        del _embed_cache[oldest]
    _embed_cache[key] = vector
    _embed_access_order.append(key)


def _has_vertex() -> bool:
    return bool(getattr(settings, "vertex_ai_project_id", None))


def _build_credentials():
    """
    Return google.oauth2 credentials.
    Priority:
      1. GOOGLE_APPLICATION_CREDENTIALS file path (mounted secret / local )
      2. Individual GCP env vars injected by k8s (private_key, client_email, …)
    """
    from google.oauth2 import service_account

    creds_path = getattr(settings, "google_application_credentials", "")
    if creds_path and Path(creds_path).exists():
        return service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    # Fall back to individual fields (k8s injects the SA JSON as separate env vars)
    private_key = getattr(settings, "gcp_private_key", "") or ""
    client_email = getattr(settings, "gcp_client_email", "") or ""
    if private_key and client_email:
        info = {
            "type": "service_account",
            "project_id": settings.vertex_ai_project_id,
            "private_key_id": getattr(settings, "gcp_private_key_id", "") or "",
            "private_key": private_key,
            "client_email": client_email,
            "client_id": getattr(settings, "gcp_client_id", "") or "",
            "auth_uri": getattr(
                settings, "gcp_auth_uri", "https://accounts.google.com/o/oauth2/auth"
            ),
            "token_uri": getattr(settings, "gcp_token_uri", "https://oauth2.googleapis.com/token"),
        }
        return service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    return None  # let ADC handle it (local dev with gcloud auth)


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        from google import genai

        credentials = _build_credentials()
        _genai_client = genai.Client(
            vertexai=True,
            project=settings.vertex_ai_project_id,
            location=settings.vertex_ai_location,
            credentials=credentials,
        )
        logger.info("vertex_genai_client_loaded", model=_GEMINI_MODEL)
    return _genai_client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns list of float vectors."""
    if not texts:
        return []

    def _call():
        client = _get_genai_client()
        response = client.models.embed_content(
            model=_GEMINI_MODEL,
            contents=texts,
        )
        return [e.values for e in response.embeddings]

    return await asyncio.to_thread(_call)


async def embed_query(text: str) -> list[float]:
    """
    Embed a single query string with LRU cache.
    Cache hit returns immediately; cache miss calls Gemini (~1-2s → ~0ms).
    """
    key = _cache_key(text)
    cached = _cache_get(key)
    if cached is not None:
        logger.debug("embed_query_cache_hit", key=key[:8])
        return cached

    results = await embed_texts([text])
    vector = results[0] if results else []
    _cache_put(key, vector)
    logger.debug("embed_query_cache_miss", key=key[:8])
    return vector


def embedding_dims() -> int:
    """Return the dimension of embeddings (always 768 for Vertex AI)."""
    return _GEMINI_DIMS
