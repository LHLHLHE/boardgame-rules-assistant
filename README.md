# Boardgame Rules Assistant

RAG-система для ответов на вопросы о правилах настольных игр. Репозиторий объединяет исследовательский контур для подготовки корпуса и подбора RAG-пайплайна, а также сервис с backend API, Telegram-ботом и административной панелью.

Векторный поиск построен на Qdrant, обработка документов и RAG-логика используют LlamaIndex. В исследовательских экспериментах LLM обычно запускается через Ollama, а сервисный контур рассчитан на OpenAI-compatible endpoints, включая vLLM.

## Структура проекта

| Раздел                      | Назначение                                                                                      |
|-----------------------------|-------------------------------------------------------------------------------------------------|
| `service/`                  | Сервис: FastAPI backend, Telegram-бот, React admin panel, Celery worker и Docker-инфраструктура |
| `research/data_work/`       | Подготовка корпуса: парсинг правил, извлечение и очистка текстов, EDA                           |
| `research/rag_experiments/` | RAG-эксперименты: индексация в Qdrant, retrieval/generation evaluation, подбор конфигурации     |
| `data/`                     | Данные, которые появляются после выполнения скриптов подготовки данных                          |
| `manifests/`                | CSV-манифесты документов и subset-архивов для индексации                                        |

> Корпус правил и производные файлы в `data/` не входят в репозиторий. Их нужно собрать локально с помощью скриптов из `research/data_work/` или подготовить самостоятельно в формате, описанном в [research/rag_experiments/README.md](research/rag_experiments/README.md).

## Основные компоненты

- **Backend API** - FastAPI-приложение для управления играми, документами, пользователями, фоновыми задачами и вопросами к RAG.
- **Telegram-бот** - пользовательский интерфейс для выбора игры и серии вопросов по правилам.
- **Admin Panel** - React/Vite-интерфейс для администрирования контента и пользователей.
- **RAG-пайплайн** - подготовка чанков, dense/hybrid retrieval, metadata filter, генерация ответов и оценка качества.
- **Инфраструктура** - PostgreSQL, Redis, Qdrant, MinIO, Nginx, Celery, dev/prod Docker Compose и Minikube-манифесты.

## Требования

Требования различаются для исследовательского и сервисного контуров:

| Контур            | Требования                                                                                      |
|-------------------|-------------------------------------------------------------------------------------------------|
| `research/`       | Python `3.11.*`, `uv`, Docker/Docker Compose для Qdrant, Ollama для локальных LLM-экспериментов |
| `service/`        | Python `3.12+`, `uv`, Docker/Docker Compose, Node.js `20+` и npm для сборки admin panel         |
| Прод/GPU-сценарии | NVIDIA driver/runtime для vLLM-сервисов, если они запускаются на целевой машине                 |

Подробные зависимости зафиксированы в `research/pyproject.toml`, `service/pyproject.toml`, `service/boardgame-rules-backend/pyproject.toml`, `service/boardgame-rules-bot/pyproject.toml` и `service/admin-panel/package.json`.

## Быстрый старт

### RAG-эксперименты

```bash
cd research
uv sync

cd rag_experiments
docker compose up -d
uv run python -m scripts.index_documents
```

Индексация предполагает, что уже подготовлены очищенные тексты в `data/rules_texts_cleaned_good/` и манифест `manifests/index_manifest.csv`.
Для генерации датасетов, оценки retrieval/generation и Hydra overrides см. [research/rag_experiments/README.md](research/rag_experiments/README.md).

### Runtime-сервис в Docker

```bash
cd service/infra-dev
cp .env.dev.example .env
```

В `.env` нужно указать рабочие OpenAI-compatible endpoints и имена моделей:

- `RAG_LLM__API_BASE`
- `RAG_EMBEDDING__API_BASE`
- `RAG_LLM__MODEL`
- `RAG_EMBEDDING__MODEL`

Затем соберите admin panel и поднимите dev-стек:

```bash
cd ../admin-panel
npm ci
npm run build

cd ../infra-dev
docker compose up -d --build
```

После запуска доступны:

- admin panel: `http://localhost/`
- Swagger UI: `http://localhost:8000/docs`
- healthcheck: `http://localhost:8000/api/health`

Полный сценарий разработки, production Docker Compose, Minikube-деплой, webhook Telegram и тесты описаны в `service/README.md`.

## Документация по разделам

- [service/README.md](service/README.md) - запуск и эксплуатация сервиса, dev/prod-инфраструктура, API и Telegram-сценарии.
- [research/data_work/README.md](research/data_work/README.md) - подготовка данных, очистка текстов правил, идентификаторы и хэши документов.
- [research/rag_experiments/README.md](research/rag_experiments/README.md) - RAG-пайплайны, конфигурация Hydra, Qdrant, метрики и результаты экспериментов.

## Лицензия

Проект распространяется по лицензии GNU AGPLv3.
