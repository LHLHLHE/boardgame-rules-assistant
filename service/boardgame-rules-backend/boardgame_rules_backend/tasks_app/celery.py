from celery import Celery

from boardgame_rules_backend.settings import app_config

celery_app = Celery(
    "boardgame_rules",
    broker=app_config.redis_url,
    backend=app_config.redis_url,
    include=["boardgame_rules_backend.tasks_app.tasks.process_rules"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)

# Register Celery signal handlers (background task tracking).
import boardgame_rules_backend.tasks_app.signals  # noqa: E402, F401
