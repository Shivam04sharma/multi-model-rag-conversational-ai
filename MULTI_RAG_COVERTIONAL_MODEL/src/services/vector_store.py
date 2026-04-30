"""
pgvector wrapper — semantic similarity search on rag_chunks.
"""

from __future__ import annotations

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


async def upsert_vectors(
    session: AsyncSession,
    namespace: str,
    chunk_ids: list[str],
    vectors: list[list[float]],
    payloads: list[dict],
) -> None:
    if not vectors:
        return

    try:
        for cid, vec in zip(chunk_ids, vectors):
            vec_literal = "[" + ",".join(str(v) for v in vec) + "]"
            await session.execute(
                text("UPDATE rag_chunks SET embedding = :vec::vector WHERE id = :id::uuid"),
                {"vec": vec_literal, "id": cid},
            )
        await session.commit()
        logger.info("pgvector_upserted", namespace=namespace, count=len(vectors))
    except Exception as exc:
        await session.rollback()
        logger.error("pgvector_upsert_failed", namespace=namespace, error=str(exc))
        raise


async def search_vectors(
    session: AsyncSession,
    namespace: str,
    query_vector: list[float],
    top_k: int = 5,
    score_threshold: float = 0.30,
) -> list[dict]:
    if not query_vector:
        return []

    try:
        vec_literal = "[" + ",".join(str(v) for v in query_vector) + "]"
        result = await session.execute(
            text("""
                SELECT
                    id,
                    document_id,
                    content,
                    filename,
                    1 - (embedding <=> cast(:vec as vector)) AS similarity
                FROM rag_chunks
                WHERE embedding IS NOT NULL
                  AND namespace = :namespace
                ORDER BY embedding <=> cast(:vec as vector)
                LIMIT :top_k
            """),
            {"vec": vec_literal, "namespace": namespace, "top_k": top_k},
        )
        rows = result.fetchall()

        hits = []
        for row in rows:
            similarity = float(row.similarity) if row.similarity else 0.0
            if similarity >= score_threshold:
                hits.append(
                    {
                        "chunk_id": str(row.id),
                        "score": round(similarity, 4),
                        "payload": {
                            "document_id": str(row.document_id) if row.document_id else "",
                            "content": row.content or "",
                            "filename": row.filename or "",
                        },
                    }
                )
        return hits
    except Exception as exc:
        await session.rollback()
        logger.warning("pgvector_search_failed", namespace=namespace, error=str(exc))
        return []


async def delete_by_document(
    session: AsyncSession,
    namespace: str,
    document_id: str,
) -> int:
    try:
        result = await session.execute(
            text("""
                UPDATE rag_chunks SET embedding = NULL
                WHERE document_id = :document_id::uuid AND namespace = :namespace
            """),
            {"document_id": document_id, "namespace": namespace},
        )
        await session.commit()
        logger.info("pgvector_deleted", namespace=namespace, document_id=document_id)
        return result.rowcount or 0
    except Exception as exc:
        logger.warning("pgvector_delete_failed", error=str(exc))
        return 0
