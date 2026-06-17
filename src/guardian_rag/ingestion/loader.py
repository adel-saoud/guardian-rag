"""Document loaders for text PDFs, markdown, and plain text.

Scanned PDFs are out of scope: text-only extraction means `pypdf` will
return empty strings for image-only pages. The loader logs a warning and
skips the page rather than raising, so a mixed-content document still
ingests cleanly.
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


_SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}


def load_path(path: Path) -> str:
    """Read a single supported file from disk and return its text.

    Raises `ValueError` for unsupported extensions. Returns an empty string
    only when the file itself is empty; missing files raise `FileNotFoundError`.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in _SUPPORTED_TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"unsupported file type: {suffix}")


def _load_pdf(path: Path) -> str:
    """Extract text from each page of a text-based PDF.

    Heavy import is local so loading a markdown file does not pay the
    pypdf import cost.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            logger.warning("ingestion.pdf.empty_page", path=str(path), page=i)
        pages.append(text)
    return "\n\n".join(pages)
