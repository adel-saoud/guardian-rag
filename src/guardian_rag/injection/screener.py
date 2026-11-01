"""Prompt-injection screening for user queries and retrieved context.

Threat model. The LLM in this pipeline only ever sees redacted text, and
the redaction maps are never part of a prompt, so an injected instruction
cannot make it reveal original PII values. What an injection *can* do is
hijack the answer: ignore the context, invent facts, or push the user
somewhere else. Screening protects answer integrity. The PII guarantees
still come from the three-stage detector and the post-generation audit.

Two entry points are screened:

- the user query, after PII redaction and before retrieval (direct
  injection);
- every parent chunk, at ingestion time (indirect injection). The
  generator receives parent text, so a flagged chunk quarantines its whole
  parent, never just the child that matched.

Fail-closed. A screener that errors flags everything it was asked about:
the query is refused, the parent is quarantined. Disabling screening is an
explicit `off` mode, never a silent fallback.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import structlog

from guardian_rag.config import ScreenerMode
from guardian_rag.ingestion.models import ParentChunk
from guardian_rag.injection.models import InjectionVerdict

logger = structlog.get_logger(__name__)

SCREENER_ERROR_REASON = "screener_error"
"""`InjectionVerdict.reason` when a screener failed and flagged the text to stay fail-closed."""


@runtime_checkable
class InjectionScreener(Protocol):
    """Anything that returns one verdict per input text, in order."""

    @property
    def screener_id(self) -> str:
        """Stable id stored next to screened chunks, used to detect stale verdicts."""
        ...

    def screen(self, texts: list[str]) -> list[InjectionVerdict]:
        """Return one verdict per text. Must not raise; errors flag the text."""
        ...


@runtime_checkable
class InjectionClassifierBackend(Protocol):
    """A sequence classifier returning P(injection) per text."""

    def injection_probability(self, texts: list[str]) -> list[float]:
        """Return one probability in [0, 1] per text, in input order."""
        ...


def _windows(text: str, window_chars: int, overlap_chars: int) -> list[str]:
    """Split text into overlapping character windows.

    The classifier truncates at 512 tokens, so a long parent chunk would
    only be screened on its head. Overlap keeps a sentence that straddles a
    window boundary whole in at least one window, as long as it is shorter
    than the overlap.
    """
    if len(text) <= window_chars:
        return [text]
    step = window_chars - overlap_chars
    return [text[start : start + window_chars] for start in range(0, len(text), step)]


class ClassifierScreener:
    """Screen texts with a local injection classifier.

    Parameters
    ----------
    backend:
        The classifier. Production wires the HuggingFace model.
    model_name:
        Model id, recorded in `screener_id`.
    threshold:
        P(injection) at or above which a text is flagged.
    window_chars, overlap_chars:
        Character windowing for texts longer than the classifier input.
        1500 characters stays under 512 DeBERTa tokens for English prose.
    """

    def __init__(
        self,
        backend: InjectionClassifierBackend,
        model_name: str,
        threshold: float = 0.5,
        window_chars: int = 1500,
        overlap_chars: int = 200,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if not 0 <= overlap_chars < window_chars:
            raise ValueError("overlap_chars must be in [0, window_chars)")
        self._backend = backend
        self._model_name = model_name
        self._threshold = threshold
        self._window_chars = window_chars
        self._overlap_chars = overlap_chars

    @property
    def screener_id(self) -> str:
        """Model and threshold, so changing either marks stored verdicts stale."""
        return f"classifier:{self._model_name}@{self._threshold}"

    def screen(self, texts: list[str]) -> list[InjectionVerdict]:
        """Score every window of every text; a text's score is its worst window."""
        if not texts:
            return []
        windows_per_text = [
            _windows(text, self._window_chars, self._overlap_chars) for text in texts
        ]
        flat = [window for windows in windows_per_text for window in windows]
        try:
            probabilities = self._backend.injection_probability(flat)
            if len(probabilities) != len(flat):
                raise RuntimeError(
                    f"classifier returned {len(probabilities)} scores for {len(flat)} inputs"
                )
        except Exception:
            logger.exception("injection.classifier.failed", n_texts=len(texts))
            return [
                InjectionVerdict(flagged=True, score=1.0, reason=SCREENER_ERROR_REASON)
                for _ in texts
            ]

        verdicts: list[InjectionVerdict] = []
        cursor = 0
        for windows in windows_per_text:
            score = max(probabilities[cursor : cursor + len(windows)])
            cursor += len(windows)
            flagged = score >= self._threshold
            verdicts.append(
                InjectionVerdict(
                    flagged=flagged,
                    score=score,
                    reason="classifier" if flagged else None,
                )
            )
        return verdicts


_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_previous",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b"
            r"(previous|prior|above|earlier|preceding|all|system)\b[^.\n]{0,30}\b"
            r"(instructions?|prompts?|rules|directions|context|guidelines)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "new_instructions",
        re.compile(r"\b(new|updated|real|actual)\s+instructions?\s*:", re.IGNORECASE),
    ),
    (
        "role_override",
        re.compile(
            r"\b(you are now|from now on,? you|act as|pretend (to be|you are))\b", re.IGNORECASE
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(reveal|print|show|repeat|output|disclose|leak)\b[^.\n]{0,40}\b"
            r"(system prompt|your instructions|hidden instructions|redaction map|"
            r"original (names?|values?|data))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "chat_template_tokens",
        re.compile(r"(<\|im_start\|>|<\|system\|>|\[INST\]|<<SYS>>|###\s*system\s*:)", re.I),
    ),
)
"""Named patterns for instruction-override phrasing seen in public injection corpora.

Regex is brittle against paraphrase (60% on the own indirect set, 10% on the
deepset test split, see ADR 0009), so rules complement the classifier. They
are the chunk default only because they produced no false positive on the
golden corpus, where the classifier flagged 3 of 30 benign documents.
"""


class RuleScreener:
    """Flag texts matching known instruction-override patterns."""

    @property
    def screener_id(self) -> str:
        """Bump the version when `_RULES` changes, so stored verdicts are re-screened."""
        return "rules:v1"

    def screen(self, texts: list[str]) -> list[InjectionVerdict]:
        """Flag a text on the first matching rule and name it in `reason`."""
        verdicts: list[InjectionVerdict] = []
        for text in texts:
            matched = next((name for name, pattern in _RULES if pattern.search(text)), None)
            verdicts.append(
                InjectionVerdict(
                    flagged=matched is not None,
                    score=1.0 if matched else 0.0,
                    reason=f"rule:{matched}" if matched else None,
                )
            )
        return verdicts


class CompositeScreener:
    """Flag a text when any member screener flags it."""

    def __init__(self, screeners: list[InjectionScreener]) -> None:
        if not screeners:
            raise ValueError("CompositeScreener needs at least one screener")
        self._screeners = screeners

    @property
    def screener_id(self) -> str:
        """Member ids joined with `+`, in member order."""
        return "+".join(s.screener_id for s in self._screeners)

    def screen(self, texts: list[str]) -> list[InjectionVerdict]:
        """Return the first flagging member's verdict, else a clean verdict with the top score."""
        per_screener = [s.screen(texts) for s in self._screeners]
        combined: list[InjectionVerdict] = []
        for verdicts in zip(*per_screener, strict=True):
            flagged = [v for v in verdicts if v.flagged]
            combined.append(
                flagged[0]
                if flagged
                else InjectionVerdict(flagged=False, score=max(v.score for v in verdicts))
            )
        return combined


class NullScreener:
    """Explicit `off` mode. Never flags. Logged loudly at startup."""

    @property
    def screener_id(self) -> str:
        """Stamped on parents ingested with screening off, so they are re-screened later."""
        return "off"

    def screen(self, texts: list[str]) -> list[InjectionVerdict]:
        """Clean verdict for every text."""
        return [InjectionVerdict(flagged=False, score=0.0) for _ in texts]


def build_screener(
    mode: ScreenerMode,
    classifier_factory: Callable[[], ClassifierScreener],
) -> InjectionScreener:
    """Build the screener for a `GUARDIAN_*_INJECTION_SCREENER` mode.

    The classifier is built through a factory so the heavy model only loads
    in the modes that use it. Callers that build several screeners should
    pass a caching factory so the model loads once.
    """
    if mode == "off":
        logger.warning("injection.screening.disabled")
        return NullScreener()
    if mode == "rules":
        return RuleScreener()
    if mode == "classifier":
        return classifier_factory()
    return CompositeScreener([classifier_factory(), RuleScreener()])


def screen_parents_for_ingestion(
    screener: InjectionScreener, parents: list[ParentChunk]
) -> list[ParentChunk]:
    """Screen parents at ingestion and stamp the verdict on each.

    Every parent gets the current `screener_id`, so a later screener change
    is detectable at query time.
    """
    if not parents:
        return []
    verdicts = screener.screen([parent.text for parent in parents])
    marked: list[ParentChunk] = []
    for parent, verdict in zip(parents, verdicts, strict=True):
        if verdict.flagged:
            logger.warning(
                "injection.parent.quarantined",
                document_id=parent.document_id,
                parent_id=parent.chunk_id,
                reason=verdict.reason,
                score=verdict.score,
            )
        marked.append(
            parent.model_copy(
                update={
                    "injection_quarantined": verdict.flagged,
                    "screener_id": screener.screener_id,
                }
            )
        )
    return marked


def screen_parents_for_generation(
    screener: InjectionScreener, parents: list[ParentChunk]
) -> tuple[list[ParentChunk], list[ParentChunk]]:
    """Return `(kept, dropped)` parents for one query's context.

    Parents already quarantined are dropped. Parents screened by a
    different screener (or never screened, for data ingested before
    screening existed) are re-screened now, since their stored verdict
    does not describe the current policy.
    """
    stale = [
        p for p in parents if not p.injection_quarantined and p.screener_id != screener.screener_id
    ]
    stale_verdicts = dict(
        zip((p.chunk_id for p in stale), screener.screen([p.text for p in stale]), strict=True)
    )
    kept: list[ParentChunk] = []
    dropped: list[ParentChunk] = []
    for parent in parents:
        verdict = stale_verdicts.get(parent.chunk_id)
        if parent.injection_quarantined or (verdict is not None and verdict.flagged):
            dropped.append(parent)
        else:
            kept.append(parent)
    if dropped:
        logger.warning(
            "injection.context.dropped",
            parent_ids=[p.chunk_id for p in dropped],
            n_rescreened=len(stale),
        )
    return kept, dropped
