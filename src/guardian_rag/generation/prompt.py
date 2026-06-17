"""Grounded prompt builder.

The prompt instructs the model to answer strictly from numbered context
blocks and to cite them via `[N]`. Citations are mapped back to parent
chunk ids by `citation.py`.

PII placeholders like `[PERSON_1]` are explicitly explained so the model
treats them as opaque tokens rather than guessing the original value.
"""

from __future__ import annotations

from guardian_rag.generation.ollama import ChatMessage
from guardian_rag.ingestion.models import ParentChunk

SYSTEM_PROMPT = (
    "You are Guardian, a privacy-safe assistant that answers questions strictly "
    "from the provided context. The context may contain redacted personal data "
    "shown as typed placeholders like [PERSON_1] or [EMAIL_1]. Treat placeholders "
    "as opaque tokens. Do not fabricate the underlying values. Quote placeholders "
    "literally when you reference them.\n\n"
    "Rules:\n"
    "1. Use only the numbered context blocks below.\n"
    "2. Cite each claim with [N] where N is the block number.\n"
    "3. If the context does not contain the answer, reply: "
    '"I do not know based on the provided context."\n'
    "4. Keep the answer concise. Do not invent facts."
)

REFUSAL_TEXT = "I do not know based on the provided context."


def _format_context(parents: list[ParentChunk]) -> str:
    """Number parents starting at 1 so citations align with the user view."""
    lines: list[str] = []
    for i, parent in enumerate(parents, start=1):
        lines.append(f"[{i}] {parent.text}")
    return "\n\n".join(lines)


def build_messages(query: str, parents: list[ParentChunk]) -> list[ChatMessage]:
    """Return the system + user messages for one grounded answer request."""
    context = _format_context(parents) if parents else "(no context available)"
    user = f"Context:\n{context}\n\nQuestion: {query}"
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=user),
    ]
