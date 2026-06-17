"""Citation extraction and validation.

The generator's prompt asks the model to cite parents using `[N]` where N
is the 1-based index of the parent block in the prompt. This module
extracts those citations and verifies each one references an actual
retrieved parent. A citation outside `[1, len(parents)]` is flagged as
ungrounded.

The validator does NOT attempt to verify factuality. That would need an
LLM-as-judge pass and lives in the eval harness instead.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from guardian_rag.ingestion.models import ParentChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")


class GroundedAnswer(BaseModel):
    """Answer text plus the parent chunks it actually cited.

    `cited_parent_ids` lists every parent referenced in citation order,
    deduplicated. `ungrounded_citations` lists `[N]` tokens whose N was
    out of range, which usually means the model invented a citation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: str
    cited_parent_ids: list[str] = Field(default_factory=lambda: [])
    ungrounded_citations: list[int] = Field(default_factory=lambda: [])

    @property
    def has_citations(self) -> bool:
        return bool(self.cited_parent_ids)

    @property
    def all_citations_grounded(self) -> bool:
        return not self.ungrounded_citations


def extract_citation_indices(answer: str) -> list[int]:
    """Return all `[N]` indices from `answer` in source order."""
    return [int(match.group(1)) for match in _CITATION_RE.finditer(answer)]


def verify_citations(answer: str, parents: list[ParentChunk]) -> GroundedAnswer:
    """Pair citations with parent ids, surface any that point outside the set."""
    indices = extract_citation_indices(answer)
    cited: list[str] = []
    seen: set[str] = set()
    ungrounded: list[int] = []
    for n in indices:
        if not 1 <= n <= len(parents):
            ungrounded.append(n)
            continue
        parent_id = parents[n - 1].chunk_id
        if parent_id in seen:
            continue
        seen.add(parent_id)
        cited.append(parent_id)
    return GroundedAnswer(
        answer=answer,
        cited_parent_ids=cited,
        ungrounded_citations=ungrounded,
    )
