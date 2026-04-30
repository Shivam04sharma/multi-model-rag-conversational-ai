"""
RAG routes.

  POST   /api/v1/rag/ingest
  POST   /api/v1/rag/query
  GET    /api/v1/rag/documents
  GET    /api/v1/rag/documents/{id}
  DELETE /api/v1/rag/documents/{id}
"""

from __future__ import annotations

import uuid

import structlog
from db.models import RagDocument
from deps import get_db
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from services.embedder import embed_query
from services.ingestion import delete_document, ingest_text
from services.vector_store import search_vectors
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()
router = APIRouter()


# ── Schemas ────────────────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    namespace: str | None = None
    filename: str | None = None
    source: str = "upload"
    metadata: dict = Field(default_factory=dict)
    data_classification: str = Field(
        default="internal",
        pattern="^(public|internal|confidential|restricted)$",
    )


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    namespace: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = False


# ── Routes ─────────────────────────────────────────────────────────────────────


@router.post("/ingest")
async def ingest(
    body: IngestRequest,
    db: AsyncSession = Depends(get_db),
):
    namespace = body.namespace or "default"
    doc = await ingest_text(
        db=db,
        text=body.text,
        namespace=namespace,
        filename=body.filename,
        source=body.source,
        metadata=body.metadata,
        data_classification=body.data_classification,
    ingested_by=None,
    )
    return {
        "status": "success",
        "document_id": str(doc.id),
        "chunks_created": doc.chunk_count,
        "namespace": namespace,
    }


@router.post("/query")
async def query_rag(
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    namespace = body.namespace or "default"
    query_vector = await embed_query(body.query)
    hits = await search_vectors(db, namespace, query_vector, top_k=body.top_k)

    chunks = []
    for h in hits:
        payload = h.get("payload", {})
        chunks.append(
            {
                "chunk_id": h["chunk_id"],
                "content": payload.get("content", ""),
                "score": round(h["score"], 4),
                "document_id": payload.get("document_id", ""),
                "filename": payload.get("filename", ""),
            }
        )

    return {"data": {"chunks": chunks, "total": len(chunks), "namespace": namespace}}


@router.get("/documents")
async def list_documents(
    namespace: str | None = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    ns = namespace or "default"
    result = await db.execute(
        select(RagDocument)
        .where(RagDocument.namespace == ns)
        .order_by(RagDocument.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    docs = result.scalars().all()
    return {
        "documents": [
            {
                "document_id": str(d.id),
                "filename": d.filename,
                "source": d.source,
                "chunk_count": d.chunk_count,
                "status": d.status,
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ],
        "total": len(docs),
        "namespace": ns,
    }


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(RagDocument).where(
            RagDocument.id == uuid.UUID(document_id),
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "document_id": str(doc.id),
        "filename": doc.filename,
        "source": doc.source,
        "metadata": doc.metadata_,
        "chunk_count": doc.chunk_count,
        "status": doc.status,
        "created_at": doc.created_at.isoformat(),
    }


@router.delete("/documents/{document_id}")
async def delete_doc(
    document_id: str,
    namespace: str = "default",
    db: AsyncSession = Depends(get_db),
):
    deleted = await delete_document(db, document_id, namespace)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted", "document_id": document_id}
