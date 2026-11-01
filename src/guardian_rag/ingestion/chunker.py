"""Hierarchical chunker: parent 1024 tokens, child 256 tokens with 10% overlap.

We split the document into parent chunks first, then split every parent
into children. Each child stores its parent's id so the retriever can
hand off the wider context window to the generator.

Token counting uses tiktoken `cl100k_base` because it is the default
encoding for the modern OpenAI-style tokenisers and is a reasonable proxy
for what most local models see. Chunk sizes are configured in tokens, not
characters, because PII-detection latency scales with tokens.
"""

from __future__ import annotations

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from guardian_rag.ingestion.models import ChildChunk, ParentChunk

TIKTOKEN_ENCODING = "cl100k_base"
"""Token counter for chunk sizes, a proxy for what local models see (see module docstring)."""


def _new_chunk_id() -> str:
    """Return a dashed UUID string: a valid Qdrant point id, so the store needs no rewrite."""
    return str(uuid.uuid4())


class HierarchicalChunker:
    """Split a redacted document into parent + child chunks.

    Parameters
    ----------
    parent_tokens:
        Target parent chunk size, in tokens. Default 1024.
    child_tokens:
        Target child chunk size, in tokens. Default 256.
    child_overlap:
        Token overlap between adjacent children inside the same parent.
        Default 25 (~10% of child size). Parent chunks do not overlap.
    """

    def __init__(
        self,
        parent_tokens: int = 1024,
        child_tokens: int = 256,
        child_overlap: int = 25,
    ) -> None:
        if parent_tokens <= 0 or child_tokens <= 0:
            raise ValueError("chunk sizes must be positive")
        if child_tokens > parent_tokens:
            raise ValueError("child_tokens must be <= parent_tokens")
        if child_overlap < 0 or child_overlap >= child_tokens:
            raise ValueError("child_overlap must be in [0, child_tokens)")

        self._parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=TIKTOKEN_ENCODING,
            chunk_size=parent_tokens,
            chunk_overlap=0,
        )
        self._child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=TIKTOKEN_ENCODING,
            chunk_size=child_tokens,
            chunk_overlap=child_overlap,
        )

    def split(
        self,
        text: str,
        document_id: str,
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Return parents and children with fresh ids; each child names its parent."""
        if not text.strip():
            return [], []

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        parent_texts: list[str] = self._parent_splitter.split_text(text)
        for parent_order, parent_text in enumerate(parent_texts):
            parent_id = _new_chunk_id()
            parents.append(
                ParentChunk(
                    chunk_id=parent_id,
                    document_id=document_id,
                    text=parent_text,
                    order=parent_order,
                )
            )
            child_texts: list[str] = self._child_splitter.split_text(parent_text)
            for child_order, child_text in enumerate(child_texts):
                children.append(
                    ChildChunk(
                        chunk_id=_new_chunk_id(),
                        document_id=document_id,
                        parent_id=parent_id,
                        text=child_text,
                        order=child_order,
                    )
                )

        return parents, children
