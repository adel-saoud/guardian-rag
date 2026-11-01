"""Retrieval-side domain models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from guardian_rag.ingestion.models import ParentChunk


class RetrievalHit(BaseModel):
    """One child chunk that survived hybrid ranking.

    `score` is the final ranking score: the cross-encoder score when the
    reranker ran, otherwise the fused hybrid score. The raw dense, BM25,
    fused and rerank scores are kept too so the dashboard can show users
    why a chunk surfaced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    document_id: str
    parent_id: str
    text: str
    order: int = Field(ge=0)
    score: float
    dense_score: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    """Top-K children plus the parent chunks they belong to.

    Generation takes `parents` as its grounded context. `children` is kept
    so the audit can show what was retrieved, including the parent
    handoff.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    children: list[RetrievalHit]
    parents: list[ParentChunk]
