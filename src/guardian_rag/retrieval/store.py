"""Qdrant adapter.

Single-collection design. Each child chunk is a Qdrant point whose vector
is the child embedding and whose payload carries the enclosing parent's
text and order. Retrieving top-K children therefore yields the parent
context for generation without a second round-trip.

The payload also carries the parent's prompt-injection verdict. Children
of a quarantined parent are excluded from dense search and from the BM25
corpus, so they can never be retrieved. Points written before screening
existed have no verdict and are re-screened at query time.

The store implements the `VectorSink` Protocol from the ingestion layer so
the ingestion pipeline can use it transparently.
"""

from __future__ import annotations

from typing import Any, cast

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from guardian_rag.ingestion.models import ChildChunk, ParentChunk
from guardian_rag.retrieval.embedder import Embedder
from guardian_rag.retrieval.models import RetrievalHit

logger = structlog.get_logger(__name__)


_BM25_SCROLL_BATCH = 256
"""Points per `scroll` page when rebuilding BM25. Bounds memory per round-trip."""


class QdrantStore:
    """Vector sink + dense-retrieval source backed by Qdrant.

    Parameters
    ----------
    client:
        A configured `QdrantClient`. Production callers pass a remote URL;
        tests can pass `QdrantClient(location=":memory:")` for a hermetic
        in-process instance.
    embedder:
        Anything matching the `Embedder` API.
    collection_name:
        Qdrant collection name. Defaults to `guardian_chunks`.
    recreate:
        If True, drop and recreate the collection on startup. Defaults to
        False so repeated ingest runs accumulate rather than wipe state.
    """

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        collection_name: str = "guardian_chunks",
        recreate: bool = False,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection_name
        self._ensure_collection(recreate=recreate)

    def _ensure_collection(self, recreate: bool) -> None:
        """Create the cosine collection sized to the embedder, dropping it first if asked."""
        exists = self._client.collection_exists(self._collection)
        if exists and not recreate:
            return
        if exists:
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=self._embedder.dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "qdrant.collection.created",
            collection=self._collection,
            dim=self._embedder.dim,
        )

    def upsert(
        self,
        parent_chunks: list[ParentChunk],
        child_chunks: list[ChildChunk],
    ) -> None:
        """Embed the children and store one point per child, carrying its parent in the payload."""
        if not child_chunks:
            return
        parents_by_id = {p.chunk_id: p for p in parent_chunks}
        texts = [c.text for c in child_chunks]
        vectors = self._embedder.embed(texts)

        points: list[PointStruct] = []
        for child, vector in zip(child_chunks, vectors, strict=True):
            parent = parents_by_id.get(child.parent_id)
            if parent is None:
                # An orphan child cannot land in the store; the pipeline
                # always produces matched parent + child sets, so this is
                # a contract violation worth surfacing.
                raise ValueError(
                    f"child {child.chunk_id} references unknown parent {child.parent_id}"
                )
            payload: dict[str, Any] = {
                "chunk_id": child.chunk_id,
                "document_id": child.document_id,
                "child_text": child.text,
                "child_order": child.order,
                "parent_id": parent.chunk_id,
                "parent_text": parent.text,
                "parent_order": parent.order,
                "injection_quarantined": parent.injection_quarantined,
                "screener_id": parent.screener_id,
            }
            points.append(
                PointStruct(
                    id=child.chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=self._collection, points=points)
        logger.info(
            "qdrant.upsert.complete",
            n_points=len(points),
            collection=self._collection,
        )

    def search_dense(
        self,
        query: str,
        top_k: int,
        document_id: str | None = None,
    ) -> list[RetrievalHit]:
        """Dense (cosine) search for the top-K most similar child chunks."""
        if top_k <= 0:
            return []
        vector = self._embedder.embed_one(query)
        # `must_not` on a missing key keeps the point, so pre-screening
        # data stays retrievable and is re-screened at query time.
        query_filter = Filter(
            must_not=[FieldCondition(key="injection_quarantined", match=MatchValue(value=True))]
        )
        if document_id is not None:
            query_filter.must = [
                FieldCondition(key="document_id", match=MatchValue(value=document_id))
            ]
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        return [_point_to_hit(point) for point in response.points]

    def fetch_by_ids(self, chunk_ids: list[str]) -> dict[str, RetrievalHit]:
        """Fetch hits by chunk id. Used by hybrid fusion."""
        if not chunk_ids:
            return {}
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=cast("list[Any]", chunk_ids),
            with_payload=True,
        )
        return {str(point.id): _point_to_hit(point) for point in points}

    def parents_from_hits(self, hits: list[RetrievalHit]) -> list[ParentChunk]:
        """Return the parents of `hits`, deduplicated, in order of first appearance.

        One child per parent is read back in a single `retrieve` call, since
        the parent text lives in the child's payload.
        """
        first_child_by_parent: dict[str, str] = {}
        for hit in hits:
            first_child_by_parent.setdefault(hit.parent_id, hit.chunk_id)
        if not first_child_by_parent:
            return []
        points = self._client.retrieve(
            collection_name=self._collection,
            ids=cast("list[Any]", list(first_child_by_parent.values())),
            with_payload=True,
        )
        payload_by_child = {str(point.id): dict(point.payload or {}) for point in points}
        parents: list[ParentChunk] = []
        for child_id in first_child_by_parent.values():
            payload: dict[str, Any] | None = payload_by_child.get(child_id)
            if payload is None:
                continue
            parents.append(
                ParentChunk(
                    chunk_id=str(payload["parent_id"]),
                    document_id=str(payload["document_id"]),
                    text=str(payload["parent_text"]),
                    order=int(payload["parent_order"]),
                    injection_quarantined=bool(payload.get("injection_quarantined", False)),
                    screener_id=_optional_str(payload.get("screener_id")),
                )
            )
        return parents

    def iter_children_for_bm25(self) -> list[tuple[str, str]]:
        """Scroll every retrievable child and return `(chunk_id, child_text)` pairs.

        Children of quarantined parents are skipped so BM25 cannot surface them.
        """
        results: list[tuple[str, str]] = []
        offset: Any = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self._collection,
                limit=_BM25_SCROLL_BATCH,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload: dict[str, Any] = dict(point.payload or {})
                if payload.get("injection_quarantined") is True:
                    continue
                results.append((str(point.id), str(payload.get("child_text", ""))))
            if next_offset is None:
                break
            offset = next_offset
        return results

    def count(self) -> int:
        """Exact number of points (child chunks) in the collection."""
        return int(self._client.count(self._collection, exact=True).count)


def _optional_str(value: Any) -> str | None:
    """Keep a missing payload value as None instead of the string "None"."""
    return None if value is None else str(value)


def _point_to_hit(point: Any) -> RetrievalHit:
    """Turn a Qdrant point into a hit. `retrieve` records have no score, so they get 0."""
    payload: dict[str, Any] = dict(point.payload or {})
    score = float(getattr(point, "score", 0.0) or 0.0)
    return RetrievalHit(
        chunk_id=str(point.id),
        document_id=str(payload["document_id"]),
        parent_id=str(payload["parent_id"]),
        text=str(payload["child_text"]),
        order=int(payload["child_order"]),
        score=score,
        dense_score=score if score else None,
    )
