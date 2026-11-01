"""Streamlit dashboard with three panels.

Every panel runs the detector from `dashboard.detector`: the three stages the
API runs, or Presidio alone (with a warning) when the `ml` extra is missing.

Excluded from pyright since `streamlit`, `plotly`, and `pandas` ship
without type stubs. Keep this file thin so the exclusion does not hide
real bugs in the rest of the codebase.

Run with:
    uv sync --extra ml --extra dashboard
    uv run streamlit run src/guardian_rag/dashboard/app.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from guardian_rag.config import Settings
from guardian_rag.dashboard.detector import build_dashboard_detector, ml_extra_installed
from guardian_rag.eval.loader import load_golden_corpus
from guardian_rag.eval.models import EvalReport, GoldenDocument
from guardian_rag.eval.runner import run_pii_eval
from guardian_rag.ingestion.chunker import HierarchicalChunker
from guardian_rag.ingestion.models import Document
from guardian_rag.ingestion.pipeline import IngestionPipeline, NullVectorSink
from guardian_rag.ingestion.redaction_store import RedactionMapStore
from guardian_rag.pii.detector import PiiDetector
from guardian_rag.pii.redactor import Redactor

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = PROJECT_ROOT / "eval" / "golden"
REDACTION_DIR = PROJECT_ROOT / "data" / "redaction_maps"


st.set_page_config(page_title="Guardian RAG", layout="wide")
st.title("Guardian RAG")
if not ml_extra_installed():
    st.warning(
        "Presidio stage only: names and other free-text PII are not detected. "
        "Run `uv sync --extra ml --extra dashboard` for the full detector."
    )


@st.cache_resource
def _detector() -> PiiDetector:
    """Loaded once per server: GLiNER and DeBERTa take seconds to load."""
    return build_dashboard_detector(Settings())


@st.cache_resource
def _pipeline() -> IngestionPipeline:
    """Ingestion that writes redaction maps and stores no vectors."""
    return IngestionPipeline(
        detector=_detector(),
        redactor=Redactor(),
        chunker=HierarchicalChunker(),
        redaction_store=RedactionMapStore(REDACTION_DIR),
        vector_sink=NullVectorSink(),
    )


@st.cache_resource
def _eval_report() -> EvalReport:
    """Computed once per server: Streamlit reruns every tab on each interaction."""
    return run_pii_eval(_load_corpus(), _detector())


@st.cache_data
def _load_corpus() -> list[GoldenDocument]:
    """Golden corpus, or an empty list before `seed_demo_corpus.py` has run."""
    if not GOLDEN_DIR.exists():
        return []
    return load_golden_corpus(GOLDEN_DIR)


def _load_redaction_maps() -> list[dict]:
    """Every readable redaction map on disk; corrupt files are skipped, not fatal."""
    if not REDACTION_DIR.exists():
        return []
    out = []
    for path in sorted(REDACTION_DIR.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


stats_tab, explorer_tab, eval_tab = st.tabs(
    ["PII Protection Stats", "Redaction Explorer", "Eval Results"]
)

with stats_tab:
    st.subheader("Aggregate PII protection")
    maps = _load_redaction_maps()
    n_docs = len(maps)
    entity_types = Counter()
    n_entities = 0
    for m in maps:
        for e in m.get("entities", []):
            entity_types[e["entity_type"]] += 1
            n_entities += 1

    documents_column, entities_column = st.columns(2)
    documents_column.metric("Documents ingested", n_docs)
    entities_column.metric("Entities detected", n_entities)

    if entity_types:
        df_types = pd.DataFrame(
            {"type": list(entity_types.keys()), "count": list(entity_types.values())}
        )
        fig = px.pie(df_types, values="count", names="type", hole=0.5)
        fig.update_layout(margin={"t": 10, "b": 10, "l": 10, "r": 10})
        st.plotly_chart(fig)
    else:
        st.info("Ingest a document to populate stats. Use the Redaction Explorer tab.")

with explorer_tab:
    st.subheader("Live redaction demo")
    st.caption(
        "Paste any text. Guardian runs the ingestion redaction pipeline "
        "and shows the result side by side."
    )
    sample = (
        "John Smith joined HR on 2025-03-12. "
        "His email is john.smith@example.com and his bank account is "
        "BE71096123456769. Manager: Bob Chen."
    )
    text = st.text_area("Input document", value=sample, height=160)
    document_id = st.text_input("Document id", value="demo-1")
    if st.button("Run redaction"):
        pipeline = _pipeline()
        doc = Document(document_id=document_id, text=text, source="dashboard")
        result = pipeline.ingest(doc)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original**")
            highlighted = text
            for entity in sorted(
                result.redaction_map.entities, key=lambda e: e.start, reverse=True
            ):
                highlighted = (
                    highlighted[: entity.start]
                    + f":red[{highlighted[entity.start : entity.end]}]"
                    + highlighted[entity.end :]
                )
            st.markdown(highlighted)
        with col2:
            st.markdown("**Redacted**")
            st.markdown(result.redacted_text)

        st.markdown("**Entities detected**")
        rows = [
            {
                "type": e.entity_type.value,
                "text": e.text,
                "start": e.start,
                "end": e.end,
                "stage": e.stage.value,
            }
            for e in result.redaction_map.entities
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.caption("No PII detected.")

with eval_tab:
    st.subheader("Eval results")
    corpus = _load_corpus()
    if not corpus:
        st.info(
            "Run `uv run python scripts/seed_demo_corpus.py` and "
            "`uv run python scripts/eval.py` to populate this panel."
        )
    else:
        report = _eval_report()
        recall_column, precision_column = st.columns(2)
        recall_column.metric("PII Recall", f"{report.overall.recall:.1%}")
        precision_column.metric("PII Precision", f"{report.overall.precision:.2f}")

        per_cat = [
            {
                "category": cat,
                "planted": c.planted,
                "detected": c.detected,
                "recall": f"{c.recall:.1%}",
                "precision": f"{c.precision:.2f}",
            }
            for cat, c in sorted(report.per_category.items())
        ]
        st.dataframe(pd.DataFrame(per_cat))
