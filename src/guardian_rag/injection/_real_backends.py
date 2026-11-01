"""Production backend for the injection classifier.

Excluded from pyright and coverage, like the other `_real_backends`
modules. `transformers` is imported lazily so the test suite never loads
model weights.
"""

from __future__ import annotations

from typing import Any, cast


class RealInjectionClassifierBackend:
    """HuggingFace text-classification pipeline returning P(injection).

    `top_k=None` returns every label's score, so the injection probability
    is read by label name instead of assuming the argmax label.
    """

    def __init__(
        self,
        model_name: str = "protectai/deberta-v3-base-prompt-injection-v2",
        injection_label: str = "INJECTION",
    ) -> None:
        from transformers import pipeline

        self._classifier = pipeline(
            "text-classification",
            model=model_name,
            truncation=True,
            max_length=512,
        )
        self._injection_label = injection_label

    def injection_probability(self, texts: list[str]) -> list[float]:
        """P(`injection_label`) per text. Raises if the model has no such label."""
        if not texts:
            return []
        outputs = cast("list[list[dict[str, Any]]]", self._classifier(texts, top_k=None))
        probabilities: list[float] = []
        for label_scores in outputs:
            by_label = {str(item["label"]): float(item["score"]) for item in label_scores}
            if self._injection_label not in by_label:
                raise KeyError(
                    f"label {self._injection_label!r} not in model labels {sorted(by_label)}"
                )
            probabilities.append(by_label[self._injection_label])
        return probabilities
