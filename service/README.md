# Сервис Boardgame Rules Assistant

## Состав

| Компонент                   | Назначение                                                        |
|-----------------------------|-------------------------------------------------------------------|
| **boardgame-rules-backend** | REST API                                                          |
| **boardgame-rules-bot**     | Telegram-бот для вопросов по правилам                             |
| **admin-panel**             | Административная панель для управления контентом и пользователями |
| **Celery worker**           | Выполнение фоновых задач                                          |
| **PostgreSQL**              | БД для табличных данных сервиса                                   |
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

**Telegram-бот**

- Python 3.12, aiogram 3  
- Inline mode Telegram (`/setinline` в BotFather)  
- FSM-контекст серии вопросов по выбранной игре

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

## Прод-развертывание на одной VM (Minikube)

Ниже описан сценарий для одной ВМ.
Манифесты находятся в `service/infra/k8s`.

### 1) Подготовить VM

- Ubuntu 22.04+ (или совместимая Linux), Docker, `kubectl`, `minikube`.
- NVIDIA Driver + NVIDIA Container Toolkit (если `vLLM` запускается с GPU).
- DNS A-запись домена должна указывать на публичный IP VM.

Проверки:

```bash
nvidia-smi
minikube version
kubectl version --client
```

### 2) Запустить Minikube и необходимые аддоны

Пример старта:

```bash
minikube start --driver=docker
minikube addons enable ingress
```

Установить cert-manager:

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml
kubectl wait --for=condition=Available deployment --all -n cert-manager --timeout=180s
```

Важно для GPU в Minikube: на ноде должен быть доступен ресурс `nvidia.com/gpu` (через NVIDIA device plugin/runtime).

```bash
kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}'; echo
```

### 3) Собрать образы backend, bot и admin-panel в Docker-демон Minikube

```bash
eval "$(minikube -p minikube docker-env)"
docker build -t boardgame-backend:latest ../boardgame-rules-backend
docker build -t boardgame-bot:latest ../boardgame-rules-bot
docker build -t boardgame-admin-panel:latest ../admin-panel
```

### 4) Подготовить конфиг перед применением манифестов

1. Отредактируйте `service/infra/k8s/02-secret.example.yaml` (токены, пароли, `JWT_SECRET`).
2. Отредактируйте `service/infra/k8s/01-configmap.yaml`:
   - `WEBHOOK_HOST=https://<ваш-домен>`
3. Отредактируйте `service/infra/k8s/30-cluster-issuer.yaml`:
   - `email: <ваш-email>`
4. Отредактируйте `service/infra/k8s/31-ingress.yaml`:
   - `host: <ваш-домен>` и `tls.hosts`.

### 5) Деплой по фазам (с миграциями до backend/celery)

```bash
cd service/infra/k8s
kubectl apply -f 00-namespace.yaml
kubectl apply -f 01-configmap.yaml -f 02-secret.example.yaml
kubectl apply -f 10-postgres.yaml -f 11-redis.yaml -f 12-qdrant.yaml -f 13-minio.yaml -f 14-vllm.yaml
kubectl apply -f 20-migrations-job.yaml
kubectl wait --for=condition=complete job/migrations -n boardgame-prod --timeout=300s
kubectl apply -f 21-backend.yaml -f 22-celery.yaml -f 23-bot.yaml -f 24-admin-frontend.yaml
kubectl apply -f 30-cluster-issuer.yaml -f 31-ingress.yaml
```

Если миграции нужно выполнить повторно:

```bash
kubectl delete job migrations -n boardgame-prod --ignore-not-found
kubectl apply -f 20-migrations-job.yaml
kubectl wait --for=condition=complete job/migrations -n boardgame-prod --timeout=300s
```

### 6) Проверка готовности

```bash
kubectl get pods -n boardgame-prod
kubectl get pvc -n boardgame-prod
kubectl get ingress -n boardgame-prod
```

Проверка API по домену:

```bash
curl -fsS https://<ваш-домен>/api/health
```

Ожидаемый ответ:

```json
{"status":"ok"}
```

### 7) Обновления и rollback

- Обновление backend/bot/admin: пересобрать image (`boardgame-backend:latest` / `boardgame-bot:latest` / `boardgame-admin-panel:latest`) и перезапустить:

  ```bash
  kubectl rollout restart deployment/backend deployment/celery deployment/bot deployment/admin-frontend -n boardgame-prod
  kubectl rollout status deployment/backend -n boardgame-prod
  ```

- Быстрый rollback:

  ```bash
  kubectl rollout undo deployment/backend -n boardgame-prod
  kubectl rollout undo deployment/bot -n boardgame-prod
  kubectl rollout undo deployment/admin-frontend -n boardgame-prod
  ```

Примечание: в `k8s`-схеме маршрутизацию выполняет `Ingress` (`/api`, `/webhook`, `/`), а `admin-frontend` раздаёт собранную статику `admin-panel`.

## Тесты backend (unit + integration)

Тесты находятся в `service/boardgame-rules-backend/tests` и разделены маркерами:

- `unit` - быстрые изолированные тесты с моками зависимостей;
- `integration` - API-тесты с реальной инфраструктурой.

### Основной сценарий: запуск внутри контейнера

```bash
cd service/infra-test
cp .env.test.example .env.test
docker compose up -d --build
```

Ниже команды запускаются на хосте и исполняют `pytest` внутри контейнера `backend`.

Unit внутри контейнера:

```bash
docker compose exec backend uv run python -m pytest -m unit
```

Integration внутри контейнера:

```bash
docker compose exec backend uv run python -m pytest -m integration
```

Все тесты внутри контейнера:

```bash
docker compose exec backend uv run python -m pytest
```

Остановить тестовую инфраструктуру:

```bash
docker compose down -v
```

### Локальный альтернативный сценарий (только unit, без infra)

```bash
cd service/boardgame-rules-backend
uv sync
uv run --project . python -m pytest -m unit
```

Примечание по PostgreSQL: внутри docker-сети используется `POSTGRES_HOST=postgres` и
`POSTGRES_PORT=5432`; host-порт задаётся отдельно через `POSTGRES_HOST_PORT`.
Дефолт для integration в `.env.test`: `RUN_INTEGRATION_TESTS=1`,
`INTEGRATION_BASE_URL=http://backend:8000`.

## Telegram-бот: сценарий использования

1. Включите inline mode у бота в BotFather: `/setinline`.
2. В чате с ботом нажмите кнопку **Задать вопрос** (или используйте `/ask`).
3. Нажмите **Найти игру** и введите название в строке ввода Telegram.
4. Выберите игру из выпадающего списка и задайте вопрос.
5. Для серии вопросов по той же игре продолжайте писать вопросы в чат.
6. Для смены игры используйте кнопку **Сменить игру**.
7. Для остановки сценария используйте кнопку **Отмена** или команду `/cancel`.

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
