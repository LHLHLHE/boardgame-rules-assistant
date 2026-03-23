# Сервис Boardgame Rules Assistant

## Состав

| Компонент                   | Назначение                                                        |
|-----------------------------|-------------------------------------------------------------------|
| **boardgame-rules-backend** | REST API                                                          |
| **admin-panel**             | Административная панель для управления контентом и пользователями |
| **Celery worker**           | Выполнение фоновых задач                                          |
| **PostgreSQL**              | БД для таблтчных данных сервиса                                   |
| **Redis**                   | Брокер и бэкенд для Celery                                        |
| **Qdrant**                  | Векторное хранилище                                               |
| **MinIO**                   | S3-совместимое хранилище файлов                                   |
| **Nginx**                   | Reverse-proxy и раздача статики                                   |

## Технологии

**Backend**

- Python 3.12, FastAPI, Uvicorn  
- SQLAlchemy 2 + Alembic (миграции), PostgreSQL, Qdrant, MinIO 
- Celery + Redis  
- LlamaIndex

**Административная панель**

- React 19, TypeScript, Vite
- React Router, TanStack Query

## Требования

- Docker и Docker Compose  
- Для сборки админки: Node.js 20+ и npm  
- Доступные порты по умолчанию в dev: `80` (Nginx), `8000` (прямой backend, если открыт), `5432`, `6379`, `6333`, `9000` / `9001` (MinIO API и консоль)

## Запуск в режиме разработки (Docker)

Рекомендуемый сценарий: каталог [`infra-dev`](infra-dev).

1. Скопируйте файл окружения и при необходимости отредактируйте значения (особенно URL vLLM для LLM и эмбеддингов):

   ```bash
   cd service/infra-dev
   cp .env.dev.example .env
   ```

2. Укажите рабочие эндпоинты моделей в `.env` (пример из шаблона):

   - `RAG_LLM__API_BASE` - базовый URL API чата (формат OpenAI, обычно с суффиксом `/v1`)  
   - `RAG_EMBEDDING__API_BASE` - базовый URL API эмбеддингов  
   - `RAG_LLM__MODEL`, `RAG_EMBEDDING__MODEL` - имена моделей на стороне сервера инференса

3. Соберите фронтенд админки (Nginx отдаёт статику из `admin-panel/dist`):

   ```bash
   cd ../admin-panel
   npm ci
   npm run build
   ```

4. Поднимите стек:

   ```bash
   cd ../infra-dev
   docker compose up -d --build
   ```

5. Создайте первого администратора (миграции выполняются контейнером `migrations` при старте):

   ```bash
   docker compose exec backend uv run python -m boardgame_rules_backend.cli create-admin <username>
   ```

После этого:

- Админка: **http://localhost/**.  
- API и Swagger UI: **http://localhost:8000**. Документация: **http://localhost:8000/docs**, схема OpenAPI: **http://localhost:8000/openapi.json**. Через Nginx на порту 80 эти пути не проксируются (прокси только `/api/`), поэтому для Swagger удобнее открыть порт 8000.  
- Проверка здоровья: `GET /api/health` (например **http://localhost:8000/api/health**).

Консоль MinIO обычно доступна на порту **9001** (логин/пароль из `.env`: `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`). Бакет для правил задаётся `S3_BUCKET` (по умолчанию `boardgame-rules`); при первом использовании его может понадобиться создать вручную в консоли, если приложение не создаёт его само.

## Взаимодействие с API

1. **Авторизация** - `POST /api/v1/auth/login` (получение JWT), далее заголовок `Authorization: Bearer <token>` для защищённых маршрутов.  
2. **Игры и правила** - префикс `/api/v1/games` (CRUD игр, загрузка PDF и т.д.).  
3. **Вопросы по правилам** - `/api/v1/questions` (RAG по выбранной игре).  
4. **Пользователи** - `/api/v1/users` (админские операции).  
5. **Фоновые задачи** - `/api/v1/background-tasks` (статус индексации и обработки файлов).

Админ-панель использует относительный базовый путь `/api/v1` и ожидает, что страница открыта с того же origin, что и Nginx (проксирование `/api/`).

---

Корневой репозиторий также содержит каталоги `research/` (эксперименты и подготовка данных) - они не являются частью runtime этого сервиса.
