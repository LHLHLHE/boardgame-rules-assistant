from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class BackgroundTaskRead(BaseModel):
    """Фоновая задача Celery (журнал для админки)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    celery_task_id: str = Field(..., description="Идентификатор задачи Celery")
    task_name: str = Field(..., description="Полное имя задачи")
    state: str = Field(..., description="STARTED, SUCCESS, FAILURE, RETRY, REVOKED")
    started_at: datetime
    finished_at: datetime | None = None
    error_message: str | None = None
    result_summary: str | None = None
    related_entity_type: str | None = None
    related_entity_id: int | None = None
    game_title: str | None = Field(
        None,
        description="Игра (ссылка) или подпись «Манифест (N документов)» для manifest_index_batch",
    )
    game_id: int | None = Field(None, description="ID игры для ссылки в админке")
    doc_id: str | None = Field(None, description="Хэш документа правил, если применимо")
