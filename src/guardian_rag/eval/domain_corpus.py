"""Synthetic in-domain retrieval corpus: HR, contracts and support.

The golden smoke set has one short paragraph per topic, so every retrieval
configuration finds the right document and the set cannot rank them. This
corpus is built to be hard in the way enterprise text is hard: each topic
is a family of near-duplicate documents (the same policy for several
offices, the same contract template for several suppliers, the same
runbook for several product versions) that share most of their wording
and differ in a few facts. A query names one variant, so the retriever has
to pick it out of its siblings.

Topics live in `eval/retrieval/domain/*.json` as section templates plus
per-variant slot values. Rendering is deterministic for a given seed. PII
slots (`{person}`, `{email}`, `{phone}`, `{iban}`, `{address}` and their
`_2` forms) are filled from fixed fictional pools and their character spans
are recorded, so the benchmark can redact with `Redactor` exactly like
ingestion does and the retriever sees `[PERSON_1]`-style placeholders.

Relevance judgements come from the spec, never from a retrieval run: query
`q` is relevant to document `topic_id--variant_id` because its author wrote
it about that variant.
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from guardian_rag.pii.models import PiiEntity, PiiEntityType, PiiStage

_FIRST_NAMES = (
    "Amara", "Bastien", "Chloe", "Dmitri", "Elif", "Farid", "Greta", "Hiroshi", "Ines", "Jonas",
    "Kavya", "Lucas", "Maren", "Nadia", "Oskar", "Priya", "Quentin", "Rosa", "Samir", "Tamsin",
    "Ulrich", "Valentina", "Wiktor", "Yara", "Zoltan",
)  # fmt: skip
_LAST_NAMES = (
    "Abernathy", "Brandt", "Castellanos", "Duval", "Eriksen", "Fontaine", "Gallagher", "Haddad",
    "Ivanova", "Jaworski", "Kowalczyk", "Lindqvist", "Moreau", "Novak", "Okafor", "Petrovic",
    "Quinlan", "Rasmussen", "Silvestri", "Takahashi", "Underwood", "Vasquez", "Whitfield",
)  # fmt: skip
_STREETS = (
    "Harbour Road", "Linden Allee", "Rue des Tanneurs", "Calle Mayor", "Kingfisher Lane",
    "Via Garibaldi", "Oostkade", "Elm Street",
)  # fmt: skip
_CITIES = ("Leeds", "Lyon", "Bilbao", "Utrecht", "Turin", "Graz", "Porto", "Aarhus")

_PII_SLOTS: dict[str, PiiEntityType] = {
    "person": PiiEntityType.PERSON,
    "email": PiiEntityType.EMAIL,
    "phone": PiiEntityType.PHONE,
    "iban": PiiEntityType.BANK_ACCOUNT,
    "address": PiiEntityType.ADDRESS,
}
PII_SLOT_NAMES = frozenset(_PII_SLOTS) | {f"{name}_2" for name in _PII_SLOTS}
"""Template fields filled with fake PII; `_2` forms let one document name two people."""

DomainCategory = Literal["hr", "contracts", "support"]


def _template_fields(template: str) -> set[str]:
    """Return the `{field}` names used in a format template."""
    return {field for _, field, _, _ in string.Formatter().parse(template) if field}


class DomainVariant(BaseModel):
    """One member of a near-duplicate family.

    `slots` fills the non-PII placeholders of the topic's sections. The
    optional `extra_section` lets a variant carry text its siblings lack,
    as real per-site addenda do.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    variant_id: str = Field(pattern=r"^[a-z0-9-]+$")
    label: str
    slots: dict[str, str]
    extra_section: str | None = None


class DomainTopic(BaseModel):
    """A family of near-duplicate documents rendered from shared sections."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic_id: str = Field(pattern=r"^[a-z0-9-]+$")
    category: DomainCategory
    title: str
    sections: list[str] = Field(min_length=3)
    variants: list[DomainVariant] = Field(min_length=4)

    @model_validator(mode="after")
    def _check_slots(self) -> DomainTopic:
        """Every variant fills every slot, and siblings really differ."""
        shared = {f for s in self.sections for f in _template_fields(s)} - PII_SLOT_NAMES
        if len({v.variant_id for v in self.variants}) != len(self.variants):
            raise ValueError(f"{self.topic_id}: duplicate variant_id")
        for variant in self.variants:
            needed = shared | (_template_fields(variant.extra_section or "") - PII_SLOT_NAMES)
            missing = needed - set(variant.slots)
            unused = set(variant.slots) - needed
            if missing or unused:
                raise ValueError(
                    f"{self.topic_id}--{variant.variant_id}: missing slots {sorted(missing)}, "
                    f"unused slots {sorted(unused)}"
                )
        fingerprints = {tuple(sorted(v.slots.items())) for v in self.variants}
        if len(fingerprints) != len(self.variants):
            raise ValueError(f"{self.topic_id}: two variants have identical slot values")
        return self


class DomainDocument(BaseModel):
    """A rendered document with the exact spans of the PII it contains."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    category: DomainCategory
    topic_id: str
    text: str
    entities: list[PiiEntity]


class DomainQuery(BaseModel):
    """A hand-written query and the one document it was written about."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    query: str
    document_id: str


def load_domain_topics(spec_dir: Path) -> list[DomainTopic]:
    """Load every `*.json` topic list in `spec_dir`, sorted for determinism."""
    topics: list[DomainTopic] = []
    for path in sorted(spec_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        topics.extend(DomainTopic.model_validate(item) for item in raw)
    if len({t.topic_id for t in topics}) != len(topics):
        raise ValueError(f"duplicate topic_id in {spec_dir}")
    return topics


def load_domain_queries(path: Path) -> list[DomainQuery]:
    """Load the query file, one JSON object per line."""
    return [
        DomainQuery.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _pii_values(rng: random.Random) -> dict[str, str]:
    """Draw one fictional value per PII slot, in real formats (UK phone, GB IBAN)."""
    values: dict[str, str] = {}
    for suffix in ("", "_2"):
        first, last = rng.choice(_FIRST_NAMES), rng.choice(_LAST_NAMES)
        values[f"person{suffix}"] = f"{first} {last}"
        values[f"email{suffix}"] = f"{first}.{last}@example.com".lower()
        values[f"phone{suffix}"] = f"+44 20 7946 {rng.randrange(1000):04d}"
        digits = "".join(str(rng.randrange(10)) for _ in range(20))
        values[f"iban{suffix}"] = "GB" + " ".join(
            [f"{rng.randrange(10, 100)}", "NWBK", *(digits[i : i + 4] for i in range(0, 16, 4))]
        )
        values[f"address{suffix}"] = (
            f"{rng.randrange(1, 200)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"
        )
    return values


def _render(
    template: str, slots: dict[str, str], pii: dict[str, str], offset: int
) -> tuple[str, list[PiiEntity]]:
    """Fill `template`, recording each PII span at its offset in the document."""
    out: list[str] = []
    entities: list[PiiEntity] = []
    cursor = offset
    for literal, field, _, _ in string.Formatter().parse(template):
        out.append(literal)
        cursor += len(literal)
        if not field:
            continue
        if field in PII_SLOT_NAMES:
            value = pii[field]
            entities.append(
                PiiEntity(
                    entity_type=_PII_SLOTS[field.removesuffix("_2")],
                    start=cursor,
                    end=cursor + len(value),
                    score=1.0,
                    text=value,
                    stage=PiiStage.PRESIDIO,
                )
            )
        else:
            value = slots[field]
        out.append(value)
        cursor += len(value)
    return "".join(out), entities


def render_domain_corpus(topics: list[DomainTopic], seed: int = 0) -> list[DomainDocument]:
    """Render every variant of every topic into a document with PII spans.

    PII values are drawn from a generator seeded per document, so adding a
    topic does not change the names in any other document.
    """
    documents: list[DomainDocument] = []
    for topic in topics:
        for variant in topic.variants:
            document_id = f"{topic.topic_id}--{variant.variant_id}"
            pii = _pii_values(random.Random(f"{seed}:{document_id}"))  # noqa: S311, fake data
            parts = [f"{topic.title} ({variant.label})"]
            entities: list[PiiEntity] = []
            offset = len(parts[0])
            templates = [
                *topic.sections,
                *([variant.extra_section] if variant.extra_section else []),
            ]
            for template in templates:
                offset += 2
                text, found = _render(template, variant.slots, pii, offset)
                parts.append(text)
                entities.extend(found)
                offset += len(text)
            documents.append(
                DomainDocument(
                    document_id=document_id,
                    category=topic.category,
                    topic_id=topic.topic_id,
                    text="\n\n".join(parts),
                    entities=entities,
                )
            )
    return documents
