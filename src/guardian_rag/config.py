"""Runtime settings, environment-driven, validated by pydantic-settings.

All configuration is exposed via `GUARDIAN_*` environment variables. Tests can
override fields by instantiating `Settings(**overrides)` directly.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration for Guardian RAG."""

    model_config = SettingsConfigDict(
        env_prefix="GUARDIAN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

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
    languages: tuple[str, ...] = Field(
        default=("en",),
        description="Languages to enable in Presidio (ISO 639-1 codes).",
    )

    # Chunking
    child_chunk_tokens: int = Field(default=256, ge=32)
    parent_chunk_tokens: int = Field(default=1024, ge=128)
    child_chunk_overlap: int = Field(default=25, ge=0)

    # Embedding
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HF model id for retrieval embeddings.",
    )
    embedding_dim: int = Field(default=384, ge=1)

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="guardian_chunks")
    qdrant_api_key: str | None = Field(default=None)

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:3b")
    ollama_timeout_s: float = Field(default=60.0, gt=0.0)

    # Retrieval
    top_k: int = Field(default=8, ge=1)
    parent_top_k: int = Field(default=4, ge=1)
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)

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
