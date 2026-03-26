import logging.config
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from boardgame_rules_backend.api.v1.auth import router as auth_router
from boardgame_rules_backend.api.v1.background_tasks import router as background_tasks_router
from boardgame_rules_backend.api.v1.games import router as games_router
from boardgame_rules_backend.api.v1.questions import router as questions_router
from boardgame_rules_backend.api.v1.users import router as users_router
from boardgame_rules_backend.connectors import (get_qdrant_async_client, get_qdrant_client,
                                                get_qdrant_vector_store, get_redis_client,
                                                get_s3_client)
from boardgame_rules_backend.exception_handlers import register_exception_handlers
from boardgame_rules_backend.rag import Generator, Retriever
from boardgame_rules_backend.settings import app_config, build_log_config, logging_settings


@asynccontextmanager
async def lifespan(application: FastAPI):
    qdrant_client = get_qdrant_client()
    qdrant_async_client = get_qdrant_async_client()
    qdrant_vector_store = get_qdrant_vector_store(qdrant_client, qdrant_async_client)

    redis_client = get_redis_client()
    s3_client = get_s3_client()

    application.state.qdrant_client = qdrant_client
    application.state.vector_store = qdrant_vector_store

    application.state.redis_client = redis_client
    application.state.s3_client = s3_client
    application.state.retriever = Retriever(vector_store=qdrant_vector_store)
    application.state.generator = Generator()
    try:
        yield
    finally:
        qdrant_client.close()
        await qdrant_async_client.close()
        await redis_client.aclose()
        s3_client.close()


app = FastAPI(
    title="Boardgame Rules Assistant API",
    description=(
        "API для управления играми, загрузки правил и ответов "
        "на вопросы по правилам через RAG."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if app_config.is_prod and not app_config.enable_docs_in_prod else "/docs",
    redoc_url=None if app_config.is_prod and not app_config.enable_docs_in_prod else "/redoc",
    openapi_url=(
        None if app_config.is_prod and not app_config.enable_docs_in_prod else "/openapi.json"
    ),
)

if app_config.is_prod:
    if app_config.cors_allow_origins_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=app_config.cors_allow_origins_list,
            allow_credentials=False,
            allow_methods=app_config.cors_allow_methods_list or ["*"],
        )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
    )

register_exception_handlers(app)

app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(
    background_tasks_router,
    prefix="/api/v1/background-tasks",
    tags=["background-tasks"],
)
app.include_router(games_router, prefix="/api/v1/games", tags=["games"])
app.include_router(questions_router, prefix="/api/v1/questions", tags=["questions"])
app.include_router(users_router, prefix="/api/v1/users", tags=["users"])


@app.get(
    "/api/health",
    summary="Проверка доступности",
    description="Возвращает статус сервиса. Используется для health-check и мониторинга.",
)
async def health():
    return {"status": "ok"}


log_config_dict = build_log_config(logging_settings)
logging.config.dictConfig(log_config_dict)

if __name__ == "__main__":
    uvicorn.run(
        "boardgame_rules_backend.main:app",
        host="0.0.0.0",
        port=8000,
        proxy_headers=True,
        log_config=log_config_dict,
        log_level=logging_settings.log_level.lower(),
    )
