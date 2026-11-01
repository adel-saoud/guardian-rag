"""Stage 1 detector: Presidio regex recognisers for structured PII.

Presidio ships a battery of `EntityRecognizer` subclasses (email, IBAN,
credit card, IP, SSN, phone number). They do not need an NLP engine, so we
call them directly without loading spaCy or Stanza. This keeps Stage 1 in
the ~5ms latency budget the architecture decision committed to.

The `PresidioEngine` exposes a single `detect()` method that returns a list
of `PiiEntity` records in source-document order, ready to merge with the
GLiNER output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from guardian_rag.pii.models import PiiEntity, PiiEntityType, PiiStage

if TYPE_CHECKING:  # pragma: no cover
    from presidio_analyzer import EntityRecognizer


_PRESIDIO_TO_DOMAIN: dict[str, PiiEntityType] = {
    "EMAIL_ADDRESS": PiiEntityType.EMAIL,
    "PHONE_NUMBER": PiiEntityType.PHONE,
    "IBAN_CODE": PiiEntityType.BANK_ACCOUNT,
    "CREDIT_CARD": PiiEntityType.CREDIT_CARD,
    "IP_ADDRESS": PiiEntityType.IP_ADDRESS,
    "US_SSN": PiiEntityType.SOCIAL_SECURITY,
    "URL": PiiEntityType.URL,
}
"""Presidio entity names mapped onto the domain enum. Unmapped entities are dropped.

Presidio supports more types than this; the list is restricted to entities
that can be redacted with a typed placeholder and still read as RAG context.
"""


def _build_default_recognizers() -> list[EntityRecognizer]:
    """Instantiate the Presidio recognisers Guardian RAG enables by default.

    Import is lazy so that `from guardian_rag.pii import ...` stays cheap;
    the module only pays for Presidio when a `PresidioEngine` is built.
    """
    from presidio_analyzer.predefined_recognizers import (
        CreditCardRecognizer,
        EmailRecognizer,
        IbanRecognizer,
        IpRecognizer,
        PhoneRecognizer,
        UrlRecognizer,
        UsSsnRecognizer,
    )

    return [
        EmailRecognizer(),
        PhoneRecognizer(),
        IbanRecognizer(),
        CreditCardRecognizer(),
        IpRecognizer(),
        UsSsnRecognizer(),
        UrlRecognizer(),
    ]


class PresidioEngine:
    """Thin wrapper around a set of Presidio `EntityRecognizer`s.

    Parameters
    ----------
    recognizers:
        Presidio recognisers to run. If omitted, the default set registers
        EmailRecognizer, PhoneRecognizer, IbanRecognizer,
        CreditCardRecognizer, IpRecognizer, UsSsnRecognizer, UrlRecognizer,
        all with their English patterns. Other languages need their own
        recognisers, passed here.
    """

    def __init__(self, recognizers: list[EntityRecognizer] | None = None) -> None:
        self._recognizers = recognizers if recognizers is not None else _build_default_recognizers()

    def detect(self, text: str) -> list[PiiEntity]:
        """Run every recogniser against ``text`` and return merged entities."""
        if not text:
            return []

        results: list[PiiEntity] = []
        for recognizer in self._recognizers:
            for entity_name in recognizer.supported_entities:
                if entity_name not in _PRESIDIO_TO_DOMAIN:
                    continue
                # `nlp_artifacts` is optional for EntityRecognizers; passing
                # None skips the spaCy code path entirely.
                matches = recognizer.analyze(
                    text=text,
                    entities=[entity_name],
                    nlp_artifacts=None,  # type: ignore[arg-type]
                )
                for match in matches or []:
                    results.append(
                        PiiEntity(
                            entity_type=_PRESIDIO_TO_DOMAIN[entity_name],
                            start=int(match.start),
                            end=int(match.end),
                            score=float(match.score),
                            text=text[int(match.start) : int(match.end)],
                            stage=PiiStage.PRESIDIO,
                        )
                    )

        results.sort(key=lambda e: (e.start, e.end))
        return results
