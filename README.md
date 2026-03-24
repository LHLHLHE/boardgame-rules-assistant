# boardgame-rules-assistant

RAG-система для ответов на вопросы о правилах настольных игр с Telegram-ботом и backend API:
Qdrant для векторного поиска, LlamaIndex для обработки документов и интеграции с LLM.

## Структура проекта

| Раздел                     | Описание                                                |
|----------------------------|---------------------------------------------------------|
| `service/`                 | Runtime-сервис: Backend API, Telegram Bot, Admin Panel  |
| `research/data_work`       | Подготовка данных: парсинг правил, очистка текстов, EDA |
| `research/rag_experiments` | RAG-пайплайн: индексация, оценка, эксперименты          |

## Требования

- Python 3.11+
- [Qdrant](https://qdrant.tech/) (Docker)
- [Ollama](https://ollama.ai/) (для LLM)

## Быстрый старт

1. Запуск Qdrant: `docker-compose up -d` (в `research/rag_experiments`).
2. Индексация и запуск экспериментов — см. [research/rag_experiments/README.md](research/rag_experiments/README.md).
3. Запуск runtime-сервисов (backend, bot, infra) — см. [service/README.md](service/README.md).
