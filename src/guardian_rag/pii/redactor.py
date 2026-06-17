"""Typed placeholder redaction.

Blank redaction destroys document structure and breaks retrieval. We replace
each PII span with a typed placeholder of the form `[PERSON_1]`,
`[EMAIL_2]`, etc. The same source span resolves to the same placeholder
within a document, so a chunk that mentions the same person twice remains
internally consistent.

`Redactor.redact()` returns the redacted text plus the `RedactionMap` that
records every placeholder. The map is stored separately and never embedded.
"""

from __future__ import annotations

from guardian_rag.pii.models import PiiEntity, PiiEntityType, RedactionMap


def _resolve_overlaps(entities: list[PiiEntity]) -> list[PiiEntity]:
    """Drop overlapping entities, keeping the higher-scoring one.

    Presidio and GLiNER can flag the same span twice when they agree. They
    can also disagree on boundaries. We resolve by sorting on (start, -score)
    and dropping anything that overlaps an entity already kept.
    """
    if not entities:
        return []
    ordered = sorted(entities, key=lambda e: (e.start, -e.score, e.end - e.start))
    kept: list[PiiEntity] = []
    for entity in ordered:
        if any(entity.overlaps(k) for k in kept):
            continue
        kept.append(entity)
    kept.sort(key=lambda e: e.start)
    return kept


class Redactor:
    """Replace PII spans in source text with typed placeholders.

    Parameters
    ----------
    placeholder_format:
        Format string with two placeholders: entity type and index. Defaults
        to ``"[{type}_{index}]"`` which produces e.g. ``[PERSON_1]``.
    """

    def __init__(self, placeholder_format: str = "[{type}_{index}]") -> None:
        self._format = placeholder_format

    def redact(
        self,
        text: str,
        entities: list[PiiEntity],
        document_id: str,
    ) -> tuple[str, RedactionMap]:
        """Redact ``entities`` from ``text`` and return the redacted text and map."""
        if not entities:
            return text, RedactionMap(document_id=document_id)

        deduped = _resolve_overlaps(entities)

        # Allocate stable placeholders per entity type per unique span text.
        # `[PERSON_1]` should refer to the same person every time it appears
        # in the document so the redacted chunk is still readable.
        index_by_type: dict[PiiEntityType, int] = {}
        token_by_span: dict[tuple[PiiEntityType, str], str] = {}
        placeholders: dict[str, str] = {}

        redacted_entities: list[PiiEntity] = []
        out: list[str] = []
        cursor = 0
        for entity in deduped:
            if entity.start < cursor:
                # Should not happen after `_resolve_overlaps`, but defend
                # the loop so a bad input cannot corrupt the output.
                continue
            key = (entity.entity_type, entity.text)
            token = token_by_span.get(key)
            if token is None:
                index_by_type[entity.entity_type] = index_by_type.get(entity.entity_type, 0) + 1
                token = self._format.format(
                    type=entity.entity_type.value,
                    index=index_by_type[entity.entity_type],
                )
                token_by_span[key] = token
                placeholders[token] = entity.text
            out.append(text[cursor : entity.start])
            out.append(token)
            cursor = entity.end
            redacted_entities.append(entity)
        out.append(text[cursor:])

        return "".join(out), RedactionMap(
            document_id=document_id,
            placeholders=placeholders,
            entities=redacted_entities,
        )
