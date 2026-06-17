"""Local end-to-end demo: ingest one document, run one query.

The script reuses the same lifespan wiring the FastAPI app uses, so it
exercises the real Presidio + GLiNER + DeBERTa + Qdrant + Ollama stack.

Usage:
    docker compose up -d ollama qdrant
    docker compose exec ollama ollama pull llama3.2:3b
    uv run python scripts/demo.py
"""

from __future__ import annotations

import asyncio

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
from guardian_rag.pii.audit import PostGenerationAuditor
from guardian_rag.pii.deberta_validator import DebertaValidator
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.gliner_recognizer import GlinerRecognizer
from guardian_rag.pii.presidio_engine import PresidioEngine
from guardian_rag.pii.redactor import Redactor
from guardian_rag.retrieval.embedder import Embedder
from guardian_rag.retrieval.hybrid import HybridRetriever
from guardian_rag.retrieval.store import QdrantStore

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
    settings = get_settings()
    console.print("[bold]Guardian RAG demo[/bold]")

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
    embedder = Embedder(
        backend=RealEmbedderBackend(settings.embedding_model),
        dim=settings.embedding_dim,
    )
    qdrant = QdrantClient(url=settings.qdrant_url)
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
    )

    console.print("\n[bold]Ingesting sample document[/bold]")
    result = pipeline.ingest(Document(document_id="hr-demo", text=SAMPLE_DOC, source="demo"))
    console.print(f"  entities detected: {result.n_entities}")
    console.print(f"  parents: {len(result.parent_chunks)}")
    console.print(f"  children: {len(result.child_chunks)}")
    console.print("[dim]redacted preview:[/dim]")
    console.print(result.redacted_text[:240])

    retriever = HybridRetriever(store, bm25_weight=settings.bm25_weight)
    retriever.refresh()

    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url,
        timeout=httpx.Timeout(settings.ollama_timeout_s),
    ) as http:
        ollama = OllamaClient(http=http)
        generator = Generator(ollama=ollama, model=settings.ollama_model)
        auditor = PostGenerationAuditor(detector=detector, redactor=redactor)

        console.print(f"\n[bold]Query[/bold]: {SAMPLE_QUERY}")
        retrieval = retriever.retrieve(SAMPLE_QUERY, top_k=4)
        grounded = await generator.generate(SAMPLE_QUERY, retrieval.parents)
        audit = auditor.audit(grounded.answer)

        console.print(f"\n[bold]Answer[/bold] (audit={'clean' if audit.clean else 'redacted'}):")
        console.print(audit.answer)


if __name__ == "__main__":
    asyncio.run(main())
