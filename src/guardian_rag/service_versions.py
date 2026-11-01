"""Minimum versions of the external services, checked once at startup.

An old server otherwise fails later and less clearly: an Ollama without the
model family returns a 404 on the first query, and qdrant-client only logs
a warning when the server is out of its supported range.

- Ollama 0.17.5 is the first release with the small Qwen3.5 sizes,
  including `qwen3.5:4b`, the default generation and judge model, and the
  fix for Qwen3.5 repeating itself
  (https://github.com/ollama/ollama/releases/tag/v0.17.5).
- Qdrant 1.17.0: the Query API (`query_points`) needs 1.10, but the pinned
  qdrant-client 1.18 supports servers at most one minor version away.
"""

from __future__ import annotations

import re
from typing import Final

MIN_OLLAMA_VERSION: Final = "0.17.5"
MIN_QDRANT_VERSION: Final = "1.17.0"

_VERSION_RE: Final = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)")


class IncompatibleServiceError(RuntimeError):
    """An external service is older than the minimum version Guardian supports."""


def _parse(version: str) -> tuple[int, int, int] | None:
    """Read `major.minor.patch` from the start of `version`; None if it has no such prefix."""
    match = _VERSION_RE.match(version.strip())
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def require_min_version(service: str, found: str, minimum: str) -> None:
    """Raise `IncompatibleServiceError` naming `minimum` when `found` is older or unreadable."""
    parsed_found = _parse(found)
    parsed_minimum = _parse(minimum)
    if parsed_minimum is None:
        raise ValueError(f"minimum version {minimum!r} is not major.minor.patch")
    if parsed_found is None:
        raise IncompatibleServiceError(
            f"{service} reported version {found!r}, which is not readable; "
            f"Guardian needs {service} {minimum} or newer."
        )
    if parsed_found < parsed_minimum:
        raise IncompatibleServiceError(
            f"{service} {found} is older than {minimum}, the minimum Guardian supports. "
            f"Upgrade {service}."
        )
