"""Top-level PII detector orchestrating Presidio + GLiNER + DeBERTa.

This is the single entry point ingestion, the query path, and the audit all
share. Each layer (Presidio, GLiNER, optional DeBERTa) is injected so the
test suite can swap any of them for deterministic fakes.

The detector returns a flat `list[PiiEntity]` in source-document order. Use
the redactor to produce text + RedactionMap.
"""

from __future__ import annotations

import structlog

from guardian_rag.pii.deberta_validator import DebertaValidator
from guardian_rag.pii.gliner_recognizer import GlinerRecognizer
from guardian_rag.pii.models import PiiEntity
from guardian_rag.pii.presidio_engine import PresidioEngine

logger = structlog.get_logger(__name__)


class PiiDetector:
    """Run Presidio, GLiNER, and the optional DeBERTa validator in sequence.

    Parameters
    ----------
    presidio:
        Stage 1 engine for structured PII (regex). Required.
    gliner:
        Stage 2 recogniser for unstructured PII (NER). Optional. If absent,
        only Presidio entities are returned.
    validator:
        Stage 3 conditional DeBERTa validator. Optional. Only fires on
        low-confidence GLiNER spans.
    """

    def __init__(
        self,
        presidio: PresidioEngine,
        gliner: GlinerRecognizer | None = None,
        validator: DebertaValidator | None = None,
    ) -> None:
        self._presidio = presidio
        self._gliner = gliner
        self._validator = validator

    def detect(self, text: str) -> list[PiiEntity]:
        """Return all PII entities found in ``text``, in source order."""
        if not text:
            return []

        presidio_entities = self._presidio.detect(text)
        gliner_entities: list[PiiEntity] = []
        if self._gliner is not None:
            gliner_entities = self._gliner.detect(text)
            if self._validator is not None:
                gliner_entities = self._validator.validate(gliner_entities)

        merged = [*presidio_entities, *gliner_entities]
        merged.sort(key=lambda e: (e.start, e.end))
        logger.debug(
            "pii.detect.complete",
            n_presidio=len(presidio_entities),
            n_gliner=len(gliner_entities),
            n_total=len(merged),
        )
        return merged
