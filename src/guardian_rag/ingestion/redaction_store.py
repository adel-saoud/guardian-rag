"""Filesystem persistence for `RedactionMap` records.

The redaction map is never embedded into the vector store. We keep one
JSON file per document in `redaction_map_dir`. The audit endpoint reads
these to render the before/after view in the dashboard and to support
audit replay.

Storage is keyed on `document_id`. Writes are atomic (write to a temp
file in the same directory and rename) so a crash mid-write cannot
corrupt an existing map.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog

from guardian_rag.pii.models import RedactionMap

logger = structlog.get_logger(__name__)


class RedactionMapStore:
    """File-backed `RedactionMap` store, one JSON file per document."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, document_id: str) -> Path:
        if "/" in document_id or "\\" in document_id or document_id in {".", ".."}:
            raise ValueError(f"unsafe document_id: {document_id!r}")
        return self._root / f"{document_id}.json"

    def save(self, redaction_map: RedactionMap) -> Path:
        path = self._path_for(redaction_map.document_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            redaction_map.model_dump_json(indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)
        logger.debug(
            "redaction_map.saved",
            document_id=redaction_map.document_id,
            n_placeholders=len(redaction_map.placeholders),
            path=str(path),
        )
        return path

    def load(self, document_id: str) -> RedactionMap:
        path = self._path_for(document_id)
        if not path.exists():
            raise FileNotFoundError(path)
        return RedactionMap.model_validate_json(path.read_text(encoding="utf-8"))

    def exists(self, document_id: str) -> bool:
        return self._path_for(document_id).exists()

    def delete(self, document_id: str) -> bool:
        path = self._path_for(document_id)
        if not path.exists():
            return False
        path.unlink()
        return True
