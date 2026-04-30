"""Async SQLAlchemy engine + session factory, and Redis pool."""

from collections.abc import AsyncGenerator

import structlog
from config import settings
from redis.asyncio import ConnectionPool, Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

logger = structlog.get_logger()

_engine = None
_session_factory: async_sessionmaker | None = None
_redis_pool: ConnectionPool | None = None


async def init_db() -> None:
    global _engine, _session_factory

    _engine = create_async_engine(
        settings.db_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        echo=settings.debug,
        pool_pre_ping=True,
        connect_args={"server_settings": {"search_path": f"public, {settings.db_schema}"}},
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    logger.info("db_connected", url=settings.db_url.split("@")[-1])

    async with _engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    logger.info("pgvector_enabled")

    if settings.db_schema:
        _schema = settings.db_schema
        async with _engine.begin() as conn:
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {_schema}"))
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {_schema}.rag_documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    namespace VARCHAR(128) NOT NULL,
                    filename VARCHAR(512),
                    source_url TEXT,
                    source VARCHAR(32) NOT NULL DEFAULT 'upload',
                    metadata JSONB NOT NULL DEFAULT '{{}}',
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    status VARCHAR(16) NOT NULL DEFAULT 'processing',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    created_by VARCHAR(128),
                    updated_by VARCHAR(128),
                    deleted_by VARCHAR(128),
                    row_version INTEGER NOT NULL DEFAULT 1,
                    is_deleted BOOLEAN NOT NULL DEFAULT false
                )
            """))
            await conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {_schema}.rag_chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id UUID NOT NULL REFERENCES {_schema}.rag_documents(id) ON DELETE CASCADE,
                    namespace VARCHAR(128) NOT NULL DEFAULT '',
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    filename VARCHAR(512),
                    embedding vector(768),
                    tokens INTEGER,
                    data_classification VARCHAR(16) NOT NULL DEFAULT 'internal',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    deleted_at TIMESTAMPTZ,
                    created_by VARCHAR(128),
                    updated_by VARCHAR(128),
                    deleted_by VARCHAR(128),
                    row_version INTEGER NOT NULL DEFAULT 1,
                    is_deleted BOOLEAN NOT NULL DEFAULT false
                )
            """))
            await conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS rag_chunks_fts_idx "
                f"ON {_schema}.rag_chunks USING GIN "
                f"(to_tsvector('english', coalesce(content, '') || ' ' || coalesce(filename, '')))"
            ))
            await conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS rag_chunks_vec_idx "
                f"ON {_schema}.rag_chunks USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = 16, ef_construction = 64)"
            ))
        logger.info("schema_ready", schema=_schema)


async def close_db() -> None:
    if _engine:
        await _engine.dispose()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _session_factory() as session:  # type: ignore[misc]
        yield session


def get_standalone_session() -> AsyncSession:
    return _session_factory()  # type: ignore[misc]


# ── Redis ──────────────────────────────────────────────────────────────────────


async def init_redis() -> None:
    global _redis_pool
    _redis_pool = ConnectionPool.from_url(
        settings.redis_url,
        db=getattr(settings, "redis_db", 0),
        decode_responses=True,
        max_connections=50,
    )
    logger.info("redis_connected", url=settings.redis_url)


def get_redis_client() -> Redis:
    return Redis(connection_pool=_redis_pool)
