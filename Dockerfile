# syntax=docker/dockerfile:1.7

# Builder
FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.5.0 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON=python3.11

WORKDIR /app

COPY pyproject.toml uv.lock* ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra ml --no-dev


# Runtime
FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 guardian

WORKDIR /app
COPY --from=builder --chown=guardian:guardian /app /app
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/guardian/.cache/huggingface

USER guardian

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "guardian_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
