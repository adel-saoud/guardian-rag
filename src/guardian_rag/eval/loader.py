"""Load the on-disk golden corpus.

Each document lives in `eval/golden/<id>.json`. The loader validates the
JSON with `GoldenDocument` so a malformed manifest fails fast instead of
producing silently wrong metrics.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from guardian_rag.eval.models import GoldenDocument
from guardian_rag.pii.models import PiiEntity, PiiStage

logger = structlog.get_logger(__name__)


def load_golden_corpus(root: Path | str) -> list[GoldenDocument]:
    """Load every `*.json` in `root`, sorted by document id for determinism."""
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(base)
    docs: list[GoldenDocument] = []
    for path in sorted(base.glob("*.json")):
        try:
            docs.append(GoldenDocument.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            logger.exception("golden.load.failed", path=str(path))
            raise
    logger.info("golden.load.complete", n_docs=len(docs), root=str(base))
    return docs


def golden_entities_as_pii(document: GoldenDocument) -> list[PiiEntity]:
    """Turn a golden document's planted entities into `PiiEntity` spans.

    The retrieval and injection benchmarks redact the golden corpus with
    its ground-truth spans, so their numbers do not depend on detector
    recall. The stage is nominal; nothing downstream reads it.
    """
    return [
        PiiEntity(
            entity_type=entity.entity_type,
            start=entity.start,
            end=entity.end,
            score=1.0,
            text=entity.text,
            stage=PiiStage.PRESIDIO,
        )
        for entity in document.entities
    ]
