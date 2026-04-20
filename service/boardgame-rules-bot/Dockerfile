FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv sync --no-dev

COPY boardgame_rules_bot ./boardgame_rules_bot

CMD ["uv", "run", "python", "-m", "boardgame_rules_bot.main"]
