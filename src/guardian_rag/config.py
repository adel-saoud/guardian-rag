"""Runtime settings, environment-driven, validated by pydantic-settings.

All configuration is exposed via `GUARDIAN_*` environment variables. Tests can
override fields by instantiating `Settings(**overrides)` directly.

Unknown `GUARDIAN_*` names, in the environment or in `.env`, stop startup
with `UnknownSettingError`. pydantic-settings would otherwise ignore them,
so a misspelled or renamed variable would silently keep its default.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from typing import Final, Literal

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

ScreenerMode = Literal["classifier", "rules", "classifier+rules", "off"]
"""Injection screener choice. `off` is explicit: disabling screening is never a fallback."""


ENV_PREFIX: Final = "GUARDIAN_"

_CLOSE_NAME_CUTOFF: Final = 0.6
"""difflib ratio below which a suggested name would mislead more than help."""


class UnknownSettingError(ValueError):
    """A `GUARDIAN_*` variable that matches no setting, with the closest valid name."""


def reject_unknown_env_names(names: Iterable[str], known: Iterable[str]) -> None:
    """Raise `UnknownSettingError` if a `GUARDIAN_*` name is not a known setting.

    Names are compared upper-cased because settings are case-insensitive.
    Variables without the prefix are not Guardian's business and are skipped.
    """
    valid = sorted({name.upper() for name in known})
    unknown = sorted(
        {
            upper
            for name in names
            if (upper := name.upper()).startswith(ENV_PREFIX) and upper not in valid
        }
    )
    if not unknown:
        return
    # Compare the part after the prefix: the shared prefix alone would make
    # every pair look similar.
    suffixes = [name.removeprefix(ENV_PREFIX) for name in valid]
    hints: list[str] = []
    for name in unknown:
        close = difflib.get_close_matches(
            name.removeprefix(ENV_PREFIX), suffixes, n=1, cutoff=_CLOSE_NAME_CUTOFF
        )
        hint = f"did you mean {ENV_PREFIX}{close[0]}?" if close else "no setting with a close name"
        hints.append(f"{name} ({hint})")
    raise UnknownSettingError(
        f"unknown {ENV_PREFIX}* variable(s): {', '.join(hints)}. "
        "Valid names are listed in docs/configuration.md."
    )


class Settings(BaseSettings):
    """Process-wide configuration for Guardian RAG."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Keep the default sources; first reject unknown names in the environment and `.env`.

        This hook sees the same `.env` file pydantic-settings loads, including
        an `_env_file` passed at construction.
        """
        seen = [
            name
            for source in (env_settings, dotenv_settings)
            if isinstance(source, EnvSettingsSource)
            for name in source.env_vars
        ]
        reject_unknown_env_names(
            seen, (f"{ENV_PREFIX}{field}" for field in settings_cls.model_fields)
        )
        return init_settings, env_settings, dotenv_settings, file_secret_settings

    # PII detection
    gliner_model: str = Field(
        default="urchade/gliner_multi_pii-v1",
        description="HF model id for the GLiNER PII recogniser.",
    )
    gliner_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Minimum span score for GLiNER to surface a candidate entity.",
    )
    deberta_model: str = Field(
        default="MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        description="HF model id for the DeBERTa zero-shot validator.",
    )
    deberta_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "GLiNER span score below which DeBERTa validates the candidate. "
            "Lowering this turns DeBERTa into a hot path. Raising it narrows the safety net."
        ),
    )

    # Chunking
    child_chunk_tokens: int = Field(
        default=256,
        ge=32,
        description="Child chunk size in tokens. Children are embedded and retrieved.",
    )
    parent_chunk_tokens: int = Field(
        default=1024,
        ge=128,
        description="Parent chunk size in tokens. Parents are the generation context.",
    )
    child_chunk_overlap: int = Field(
        default=25,
        ge=0,
        description="Token overlap between adjacent children, about 10% of the child size.",
    )

    # Embedding
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HF model id for retrieval embeddings.",
    )
    embedding_dim: int = Field(
        default=384,
        ge=1,
        description="Vector size of the Qdrant collection. Must match `embedding_model`.",
    )

    # Qdrant
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Passed to `QdrantClient(location=...)`: a server URL, or `:memory:`.",
    )
    qdrant_collection: str = Field(default="guardian_chunks")
    qdrant_api_key: str | None = Field(default=None)

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen3.5:4b", description="Ollama generation model.")
    ollama_judge_model: str = Field(
        default="qwen3.5:4b",
        description="Ollama model for the faithfulness judge in `scripts/eval_faithfulness.py`.",
    )
    ollama_think: bool = Field(
        default=False,
        description=(
            "Ollama `think` request field. Qwen3.5 thinks by default; off keeps "
            "answers short and the judge's yes/no reply parsable."
        ),
    )
    ollama_timeout_s: float = Field(
        default=60.0,
        gt=0.0,
        description="Per-request timeout in seconds. CPU generation with a 4B model is slow.",
    )

    # Retrieval
    bm25_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="BM25 share of the fused score; the dense share is `1 - bm25_weight`.",
    )

    # Reranking
    reranker_enabled: bool = Field(
        default=False,
        description="Rerank the fused hybrid pool with a local cross-encoder.",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        description=(
            "HF cross-encoder id. English-only by default; "
            "BAAI/bge-reranker-v2-m3 is the multilingual option."
        ),
    )
    rerank_candidates: int = Field(
        default=30,
        ge=1,
        description="Fused hybrid candidates handed to the reranker.",
    )

    # Queries and chunks get separate modes because the classifier flags
    # benign support tickets (ADR 0009): tolerable for a refused query, not
    # for a document silently quarantined at ingestion.
    query_injection_screener: ScreenerMode = Field(
        default="classifier+rules",
        description=(
            "Screen redacted user queries. `off` disables screening and is "
            "logged as a warning at startup."
        ),
    )
    chunk_injection_screener: ScreenerMode = Field(
        default="rules",
        description=(
            "Screen parent chunks at ingestion (and stale verdicts at query "
            "time). `off` disables screening and is logged as a warning."
        ),
    )
    injection_model: str = Field(
        default="protectai/deberta-v3-base-prompt-injection-v2",
        description="HF sequence classifier for prompt injection. English-only.",
    )
    injection_label: str = Field(
        default="INJECTION",
        description="Label name of the injection class in the classifier's id2label.",
    )
    injection_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="P(injection) at or above which a query or chunk is flagged.",
    )

    # Storage
    redaction_map_dir: str = Field(
        default="data/redaction_maps",
        description="Filesystem directory where per-document redaction maps live.",
    )

    # Audit behaviour
    audit_fail_closed: bool = Field(
        default=True,
        description=(
            "If the post-generation audit raises, redact the answer instead of returning it."
        ),
    )


def get_settings() -> Settings:
    """Return a freshly-constructed `Settings` instance.

    Kept as a function (not a cached singleton) so tests can monkey-patch
    environment variables and observe the change without import-order tricks.
    """
    return Settings()
