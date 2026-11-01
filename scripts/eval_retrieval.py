"""Measure retrieval quality with and without the cross-encoder reranker.

Three datasets:

- `scifact` (default): BEIR SciFact test split, 5,183 abstracts and 300
  claims with relevance judgements. Out of domain for Guardian (science,
  not HR or contracts) but comparable with published BEIR numbers.
  Licensed CC-BY-SA-4.0, so it is downloaded at run time into
  `data/beir/` (gitignored) and never committed.
- `golden`: 30 hand-written queries over the 30-document synthetic golden
  corpus, redacted with the corpus's own entity spans so every chunk
  carries `[PERSON_1]`-style placeholders. A small in-domain smoke test,
  not a benchmark.
- `domain`: the synthetic in-domain benchmark, 270 multi-chunk HR,
  contract and support documents in 45 near-duplicate families
  (`guardian_rag.eval.domain_corpus`), redacted with their recorded PII
  spans, and 135 hand-written queries that each name one family member.
  Built to rank configurations where `golden` saturates.

Each configuration retrieves the same 30 child chunks per query, collapses
them to a document ranking, and scores nDCG@10, Recall@10 and MRR@10.

Usage:
    uv sync --extra ml --extra bench
    uv run python scripts/eval_retrieval.py
    uv run python scripts/eval_retrieval.py --dataset golden
    uv run python scripts/eval_retrieval.py --dataset domain
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table

from guardian_rag.config import Settings
from guardian_rag.eval.domain_corpus import (
    load_domain_queries,
    load_domain_topics,
    render_domain_corpus,
)
from guardian_rag.eval.loader import golden_entities_as_pii, load_golden_corpus
from guardian_rag.eval.ranking import (
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_ci,
    percentile,
    rank_documents,
    recall_at_k,
)
from guardian_rag.ingestion.chunker import HierarchicalChunker
from guardian_rag.ingestion.models import ChildChunk, ParentChunk
from guardian_rag.pii.redactor import Redactor
from guardian_rag.retrieval.embedder import Embedder
from guardian_rag.retrieval.hybrid import HybridRetriever
from guardian_rag.retrieval.reranker import Reranker
from guardian_rag.retrieval.store import QdrantStore

console = Console()

ROOT = Path(__file__).resolve().parent.parent
SCIFACT_REVISION = "b3b5335604bf5ee3c4447671af975ea25143d4f5"
SCIFACT_QRELS_REVISION = "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029"
"""Pinned Hugging Face dataset commits, so reruns score the same corpus and qrels."""
RANK_CUTOFF = 10
"""Cut-off for nDCG, Recall and MRR, the BEIR reporting standard."""
CHILDREN_PER_QUERY = 30
"""Child chunks retrieved per query. Every config ranks the same pool size,
and 30 children usually cover 10 distinct documents after collapsing."""
UPSERT_BATCH = 256
"""Documents chunked and embedded per Qdrant upsert, to bound memory."""

Corpus = dict[str, str]
Queries = dict[str, str]
Qrels = dict[str, dict[str, int]]


def _load_scifact(data_dir: Path) -> tuple[Corpus, Queries, Qrels]:
    """Download SciFact at the pinned revisions; keep only queries with test qrels."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    def fetch(repo_id: str, filename: str, revision: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id,
                filename,
                repo_type="dataset",
                revision=revision,
                local_dir=data_dir / repo_id,
            )
        )

    corpus_rows = pq.read_table(
        fetch("BeIR/scifact", "corpus/corpus-00000-of-00001.parquet", SCIFACT_REVISION)
    ).to_pylist()
    query_rows = pq.read_table(
        fetch("BeIR/scifact", "queries/queries-00000-of-00001.parquet", SCIFACT_REVISION)
    ).to_pylist()
    qrels_path = fetch("BeIR/scifact-qrels", "test.tsv", SCIFACT_QRELS_REVISION)

    qrels: Qrels = defaultdict(dict)
    with qrels_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qrels[str(row["query-id"])][str(row["corpus-id"])] = int(row["score"])

    corpus = {str(r["_id"]): f"{r['title']}\n{r['text']}".strip() for r in corpus_rows}
    queries = {str(r["_id"]): str(r["text"]) for r in query_rows if str(r["_id"]) in qrels}
    return corpus, queries, dict(qrels)


def _load_golden() -> tuple[Corpus, Queries, Qrels]:
    """Golden corpus redacted with its own spans, and the 30 smoke queries."""
    redactor = Redactor()
    corpus: Corpus = {}
    for doc in load_golden_corpus(ROOT / "eval" / "golden"):
        corpus[doc.document_id], _ = redactor.redact(
            doc.text, golden_entities_as_pii(doc), doc.document_id
        )

    queries: Queries = {}
    qrels: Qrels = {}
    path = ROOT / "eval" / "retrieval" / "golden_queries.jsonl"
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = f"g{i:03d}"
        queries[query_id] = row["query"]
        qrels[query_id] = {row["document_id"]: 1}
    return corpus, queries, qrels


def _load_domain() -> tuple[Corpus, Queries, Qrels]:
    """Render and redact the in-domain corpus; qrels come from the query file."""
    redactor = Redactor()
    documents = render_domain_corpus(load_domain_topics(ROOT / "eval" / "retrieval" / "domain"))
    corpus: Corpus = {
        doc.document_id: redactor.redact(doc.text, doc.entities, doc.document_id)[0]
        for doc in documents
    }
    rows = load_domain_queries(ROOT / "eval" / "retrieval" / "domain_queries.jsonl")
    queries: Queries = {row.query_id: row.query for row in rows}
    qrels: Qrels = {row.query_id: {row.document_id: 1} for row in rows}
    return corpus, queries, qrels


def _build_store(corpus: Corpus, settings: Settings) -> QdrantStore:
    """Chunk and embed the corpus into an in-memory Qdrant with the production settings."""
    from guardian_rag.retrieval._real_backends import RealEmbedderBackend

    embedder = Embedder(RealEmbedderBackend(settings.embedding_model), dim=settings.embedding_dim)
    store = QdrantStore(
        QdrantClient(location=":memory:"), embedder, collection_name="bench", recreate=True
    )
    chunker = HierarchicalChunker(
        parent_tokens=settings.parent_chunk_tokens,
        child_tokens=settings.child_chunk_tokens,
        child_overlap=settings.child_chunk_overlap,
    )
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    started = time.perf_counter()
    for i, (doc_id, text) in enumerate(corpus.items(), start=1):
        doc_parents, doc_children = chunker.split(text, doc_id)
        parents.extend(doc_parents)
        children.extend(doc_children)
        if i % UPSERT_BATCH == 0 or i == len(corpus):
            store.upsert(parents, children)
            parents, children = [], []
    console.print(
        f"Indexed {len(corpus)} documents, {store.count()} chunks "
        f"in {time.perf_counter() - started:.0f}s"
    )
    return store


def _evaluate(
    retriever: HybridRetriever, queries: Queries, qrels: Qrels
) -> tuple[dict[str, float], list[float]]:
    """Return mean metrics and latency percentiles, plus per-query nDCG for the bootstrap."""
    ndcg: list[float] = []
    recall: list[float] = []
    mrr: list[float] = []
    latency_ms: list[float] = []
    # One untimed call so model warm-up does not land in the latency tail.
    retriever.retrieve(next(iter(queries.values())), top_k=CHILDREN_PER_QUERY)
    for query_id, text in queries.items():
        started = time.perf_counter()
        result = retriever.retrieve(text, top_k=CHILDREN_PER_QUERY)
        latency_ms.append((time.perf_counter() - started) * 1000)
        ranked = rank_documents(result.children)
        ndcg.append(ndcg_at_k(ranked, qrels[query_id], RANK_CUTOFF))
        recall.append(recall_at_k(ranked, qrels[query_id], RANK_CUTOFF))
        mrr.append(mrr_at_k(ranked, qrels[query_id], RANK_CUTOFF))
    summary = {
        f"ndcg@{RANK_CUTOFF}": statistics.fmean(ndcg),
        f"recall@{RANK_CUTOFF}": statistics.fmean(recall),
        f"mrr@{RANK_CUTOFF}": statistics.fmean(mrr),
        "latency_p50_ms": percentile(latency_ms, 50),
        "latency_p95_ms": percentile(latency_ms, 95),
    }
    return summary, ndcg


def main() -> int:
    """Score dense, hybrid and hybrid + rerank on one dataset and write the JSON result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["scifact", "golden", "domain"], default="scifact")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "beir")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from guardian_rag.retrieval._real_backends import RealCrossEncoderBackend

    settings = Settings()
    loaders = {
        "scifact": lambda: _load_scifact(args.data_dir),
        "golden": _load_golden,
        "domain": _load_domain,
    }
    corpus, queries, qrels = loaders[args.dataset]()
    console.print(f"{args.dataset}: {len(corpus)} documents, {len(queries)} queries")

    store = _build_store(corpus, settings)
    reranker = Reranker(RealCrossEncoderBackend(settings.reranker_model), settings.reranker_model)
    configs = {
        "dense": HybridRetriever(store, bm25_weight=0.0),
        f"hybrid (bm25_weight={settings.bm25_weight})": HybridRetriever(
            store, bm25_weight=settings.bm25_weight
        ),
        "hybrid + rerank": HybridRetriever(
            store,
            bm25_weight=settings.bm25_weight,
            reranker=reranker,
            rerank_candidates=CHILDREN_PER_QUERY,
        ),
    }

    results: dict[str, dict[str, float]] = {}
    per_query_ndcg: dict[str, list[float]] = {}
    for name, retriever in configs.items():
        retriever.refresh()
        results[name], per_query_ndcg[name] = _evaluate(retriever, queries, qrels)

    hybrid_name, rerank_name = list(configs)[1:]
    delta, low, high = paired_bootstrap_ci(per_query_ndcg[hybrid_name], per_query_ndcg[rerank_name])
    rerank_delta = {f"ndcg@{RANK_CUTOFF}_delta": delta, "ci95_low": low, "ci95_high": high}
    console.print(
        f"Rerank vs hybrid, nDCG@{RANK_CUTOFF} delta {delta:+.3f} "
        f"(95% paired bootstrap CI {low:+.3f} to {high:+.3f})"
    )

    table = Table(title=f"Guardian RAG · retrieval eval · {args.dataset}")
    table.add_column("Config")
    metric_names = list(next(iter(results.values())))
    for metric in metric_names:
        table.add_column(metric, justify="right")
    for name, metrics in results.items():
        table.add_row(
            name,
            *(f"{v:.0f}" if k.startswith("latency") else f"{v:.3f}" for k, v in metrics.items()),
        )
    console.print(table)

    out = args.out or ROOT / "eval" / "results" / f"retrieval_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "n_documents": len(corpus),
                "n_queries": len(queries),
                "children_per_query": CHILDREN_PER_QUERY,
                "embedding_model": settings.embedding_model,
                "reranker_model": settings.reranker_model,
                "results": results,
                "rerank_vs_hybrid": rerank_delta,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
