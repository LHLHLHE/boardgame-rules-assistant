FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

COPY boardgame_rules_backend ./boardgame_rules_backend

COPY alembic.ini ./
COPY alembic ./alembic

RUN uv sync --no-dev

CMD ["uv", "run", "python", "-m", "boardgame_rules_backend.main"]
