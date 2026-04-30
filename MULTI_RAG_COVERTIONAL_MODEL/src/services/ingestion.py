"""
Ingestion pipeline — text in, indexed chunks out.
"""

from __future__ import annotations

import uuid

import structlog
from db.models import RagChunk, RagDocument
from services.chunker import chunk_text
from services.embedder import embed_texts
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


async def ingest_text(
    db: AsyncSession,
    text: str,
    namespace: str,
    filename: str | None = None,
    source: str = "upload",
    metadata: dict | None = None,
    data_classification: str = "internal",
    ingested_by: str | None = None,
) -> RagDocument:
    """
    Full ingestion pipeline. Returns the saved RagDocument.
    Embeddings stored in rag_chunks.embedding (pgvector).
    tsvector for BM25 is generated automatically by the DB.
    """
    metadata = metadata or {}

    # 1. Create document record
    doc = RagDocument(
        namespace=namespace,
        filename=filename,
        source=source,
        metadata_=metadata,
        status="processing",
        created_by=ingested_by,
    )
    db.add(doc)
    await db.flush()  # get doc.id without committing

    doc_id = str(doc.id)
    logger.info("ingest_started", document_id=doc_id, namespace=namespace, source=source)

    try:
        # 2. Chunk
        chunks = chunk_text(text)
        if not chunks:
            doc.status = "failed"
            await db.commit()
            raise ValueError("No chunks produced from text")

        # 3. Embed
        vectors = await embed_texts(chunks)

        # 4. Build chunk records with embeddings
        db_chunks: list[RagChunk] = []
        for i, (content, vector) in enumerate(zip(chunks, vectors)):
            cid = uuid.uuid4()
            db_chunks.append(
                RagChunk(
                    id=cid,
                    document_id=doc.id,
                    namespace=namespace,
                    chunk_index=i,
                    content=content,
                    filename=filename,
                    embedding=vector,
                    tokens=len(content) // 4,
                    data_classification=data_classification,
                    created_by=ingested_by,
                )
            )

        # 5. Save chunks to DB (embedding stored as pgvector, tsvector auto-generated)
        db.add_all(db_chunks)

        # 6. Update document
        doc.chunk_count = len(chunks)
        doc.status = "ready"
        await db.commit()
        await db.refresh(doc)

        logger.info("ingest_complete", document_id=doc_id, chunks=len(chunks))
        return doc

    except Exception as exc:
        doc.status = "failed"
        await db.commit()
        logger.error("ingest_failed", document_id=doc_id, error=str(exc))
        raise


async def delete_document(
    db: AsyncSession,
    document_id: str,
    namespace: str,
) -> bool:
    """Delete document + all its chunks from PostgreSQL."""
    result = await db.execute(
        select(RagDocument).where(
            RagDocument.id == uuid.UUID(document_id),
            RagDocument.namespace == namespace,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        return False

    await db.delete(doc)
    await db.commit()

    logger.info("document_deleted", document_id=document_id, namespace=namespace)
    return True
