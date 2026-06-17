"""Production backends that wrap real HuggingFace models.

Imports of ``gliner`` and ``transformers`` are performed lazily inside the
constructors so the test suite, which substitutes deterministic fakes,
never has to install or download the heavy ML stack.

This module is excluded from pyright and coverage. Keep it small: every
line here is a line the type-checker cannot verify.
"""

from __future__ import annotations

from typing import Any, cast

from guardian_rag.pii.gliner_recognizer import GlinerSpan


class RealGlinerBackend:
    """Stage-2 backend backed by the official ``gliner`` package."""

    def __init__(self, model_name: str = "urchade/gliner_multi_pii-v1") -> None:
        from gliner import GLiNER

        self._model = GLiNER.from_pretrained(model_name)

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float,
    ) -> list[GlinerSpan]:
        spans = cast(
            "list[GlinerSpan]",
            self._model.predict_entities(text, labels, threshold=threshold),
        )
        return spans


class RealDebertaValidatorBackend:
    """Stage-3 backend backed by a HuggingFace zero-shot NLI pipeline."""

    HYPOTHESIS_TEMPLATE = "This text is {}."

    def __init__(
        self,
        model_name: str = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
    ) -> None:
        from transformers import pipeline

        self._pipeline = pipeline("zero-shot-classification", model=model_name)

    def score(self, text: str, hypothesis: str) -> float:
        result = cast(
            "dict[str, Any]",
            self._pipeline(
                text,
                candidate_labels=[hypothesis, "something else"],
                hypothesis_template=self.HYPOTHESIS_TEMPLATE,
            ),
        )
        labels = cast("list[str]", result["labels"])
        scores = cast("list[float]", result["scores"])
        for label, score in zip(labels, scores, strict=True):
            if label == hypothesis:
                return float(score)
        return 0.0
