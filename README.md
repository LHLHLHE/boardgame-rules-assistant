# boardgame-rules-assistant

RAG-система для ответов на вопросы о правилах настольных игр: Qdrant для векторного поиска, LlamaIndex для обработки документов и интеграции с LLM.

## Структура проекта

| Раздел                     | Описание                                                |
|----------------------------|---------------------------------------------------------|
| `research/data_work`       | Подготовка данных: парсинг правил, очистка текстов, EDA |
| `research/rag_experiments` | RAG-пайплайн: индексация, оценка, эксперименты          |

## Требования

- Python 3.11+
- [Qdrant](https://qdrant.tech/) (Docker)
- [Ollama](https://ollama.ai/) (для LLM)

## Быстрый старт

1. Запуск Qdrant: `docker-compose up -d` (в `research/rag_experiments`).
2. Индексация и запуск экспериментов — см. [research/rag_experiments/README.md](research/rag_experiments/README.md).
