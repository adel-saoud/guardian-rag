"""Local end-to-end demo: ingest one document, run one query.

The wiring mirrors the FastAPI lifespan in `guardian_rag.api.app`, so it
exercises the real Presidio + GLiNER + DeBERTa + Qdrant + Ollama stack,
with the query and chunk injection screens set by
`GUARDIAN_QUERY_INJECTION_SCREENER` and `GUARDIAN_CHUNK_INJECTION_SCREENER`.
The reranker is left out. The Qdrant collection is recreated on every run.

Usage:
    docker compose up -d ollama qdrant
    docker compose exec ollama ollama pull qwen3.5:4b
    uv run python scripts/demo.py
"""

from __future__ import annotations

import asyncio
from functools import cache

import httpx
from qdrant_client import QdrantClient
from rich.console import Console

from guardian_rag.config import get_settings
from guardian_rag.generation.generator import Generator
from guardian_rag.generation.ollama import OllamaClient
from guardian_rag.ingestion.chunker import HierarchicalChunker
from guardian_rag.ingestion.models import Document
from guardian_rag.ingestion.pipeline import IngestionPipeline
from guardian_rag.ingestion.redaction_store import RedactionMapStore
from guardian_rag.injection.screener import (
    ClassifierScreener,
    build_screener,
    screen_parents_for_generation,
)
from guardian_rag.pii.audit import PostGenerationAuditor
from guardian_rag.pii.deberta_validator import DebertaValidator
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.gliner_recognizer import GlinerRecognizer
from guardian_rag.pii.presidio_engine import PresidioEngine
from guardian_rag.pii.redactor import Redactor
from guardian_rag.retrieval.embedder import Embedder
from guardian_rag.retrieval.hybrid import HybridRetriever
from guardian_rag.retrieval.store import QdrantStore
from guardian_rag.service_versions import (
    MIN_OLLAMA_VERSION,
    MIN_QDRANT_VERSION,
    require_min_version,
)

console = Console()


SAMPLE_DOC = """
HR Policy Memo, 2025-03-12.

The new hire programme welcomes Alice Bennett, joining the marketing team.
Her email is alice.bennett@example.com and her direct phone is +1-555-0142.
For payroll, salary deposits will be routed to BE71096123456769 starting
April. Benefits questions go to Bob Chen at bob.chen@example.com.
""".strip()


SAMPLE_QUERY = "Who handles benefits questions and what is their contact?"


async def main() -> None:
    """Ingest `SAMPLE_DOC`, then answer `SAMPLE_QUERY` through the query path."""
    settings = get_settings()
    console.print("[bold]Guardian RAG demo[/bold]")

    from guardian_rag.injection._real_backends import RealInjectionClassifierBackend
    from guardian_rag.pii._real_backends import (
        RealDebertaValidatorBackend,
        RealGlinerBackend,
    )
    from guardian_rag.retrieval._real_backends import RealEmbedderBackend

    detector = PiiDetector(
        presidio=PresidioEngine(),
        gliner=GlinerRecognizer(
            backend=RealGlinerBackend(settings.gliner_model),
            threshold=settings.gliner_threshold,
        ),
        validator=DebertaValidator(
            backend=RealDebertaValidatorBackend(settings.deberta_model),
            score_threshold=settings.deberta_threshold,
        ),
    )
    redactor = Redactor()

    @cache
    def injection_classifier() -> ClassifierScreener:
        """Cached so the query and chunk screeners share one model load."""
        return ClassifierScreener(
            backend=RealInjectionClassifierBackend(
                settings.injection_model, settings.injection_label
            ),
            model_name=settings.injection_model,
            threshold=settings.injection_threshold,
        )

    query_screener = build_screener(settings.query_injection_screener, injection_classifier)
    chunk_screener = build_screener(settings.chunk_injection_screener, injection_classifier)
    embedder = Embedder(
        backend=RealEmbedderBackend(settings.embedding_model),
        dim=settings.embedding_dim,
    )
    qdrant = QdrantClient(location=settings.qdrant_url, api_key=settings.qdrant_api_key)
    require_min_version("Qdrant", qdrant.info().version, MIN_QDRANT_VERSION)
    store = QdrantStore(
        client=qdrant,
        embedder=embedder,
        collection_name=settings.qdrant_collection,
        recreate=True,
    )
    chunker = HierarchicalChunker(
        parent_tokens=settings.parent_chunk_tokens,
        child_tokens=settings.child_chunk_tokens,
        child_overlap=settings.child_chunk_overlap,
    )
    pipeline = IngestionPipeline(
        detector=detector,
        redactor=redactor,
        chunker=chunker,
        redaction_store=RedactionMapStore(settings.redaction_map_dir),
        vector_sink=store,
        screener=chunk_screener,
    )

    console.print("\n[bold]Ingesting sample document[/bold]")
    result = pipeline.ingest(Document(document_id="hr-demo", text=SAMPLE_DOC, source="demo"))
    console.print(f"  entities detected: {result.n_entities}")
    console.print(f"  parents: {len(result.parent_chunks)}")
    console.print(f"  children: {len(result.child_chunks)}")
    console.print(f"  quarantined parents: {result.n_quarantined_parents}")
    console.print("[dim]redacted preview:[/dim]")
    console.print(result.redacted_text[:240])

    retriever = HybridRetriever(store, bm25_weight=settings.bm25_weight)
    retriever.refresh()

    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url,
        timeout=httpx.Timeout(settings.ollama_timeout_s),
    ) as http:
        ollama = OllamaClient(http=http, think=settings.ollama_think)
        require_min_version("Ollama", await ollama.version(), MIN_OLLAMA_VERSION)
        generator = Generator(ollama=ollama, model=settings.ollama_model)
        auditor = PostGenerationAuditor(detector=detector, redactor=redactor)

        # Same order as /v1/query: redact first, so neither the screener,
        # the retriever nor the LLM ever sees PII from the query.
        safe_query, _query_map = redactor.redact(
            SAMPLE_QUERY, detector.detect(SAMPLE_QUERY), "query"
        )
        console.print(f"\n[bold]Query[/bold] (redacted): {safe_query}")
        if query_screener.screen([safe_query])[0].flagged:
            console.print("Query blocked by the prompt-injection screen.")
            return
        retrieval = retriever.retrieve(safe_query, top_k=4)
        parents, dropped = screen_parents_for_generation(chunk_screener, retrieval.parents)
        console.print(f"  parents dropped by the injection screen: {len(dropped)}")
        grounded = await generator.generate(safe_query, parents)
        audit = auditor.audit(grounded.answer)

        console.print(f"\n[bold]Answer[/bold] (audit={'clean' if audit.clean else 'redacted'}):")
        console.print(audit.answer)


if __name__ == "__main__":
    asyncio.run(main())
