from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GameBase(BaseModel):
    title: str = Field(..., min_length=1, description="Название игры")
    source_doc_url: str | None = Field(None, description="Ссылка на источник правил (URL)")

    @field_validator("title")
    @classmethod
    def strip_title(cls, v: str) -> str:
        s = v.strip()
        if not s:
            msg = "title cannot be empty"
            raise ValueError(msg)
        return s


class GameCreate(GameBase):
    """Схема создания игры."""

    pass


class GameUpdate(BaseModel):
    title: str | None = Field(None, description="Название игры")
    source_doc_url: str | None = Field(None, description="Ссылка на источник правил (URL)")

    @field_validator("title")
    @classmethod
    def strip_title_if_set(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if not s:
            msg = "title cannot be empty"
            raise ValueError(msg)
        return s

    @model_validator(mode="after")
    def reject_explicit_null_title(self) -> Self:
        if "title" in self.model_fields_set and self.title is None:
            msg = "title cannot be null"
            raise ValueError(msg)
        return self


class GameRead(GameBase):
    """Схема чтения игры."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="ID игры")
    created_at: datetime = Field(..., description="Дата создания")
    updated_at: datetime = Field(..., description="Дата обновления")


class GameListRead(BaseModel):
    """Список игр с общим количеством (пагинация)."""

    items: list[GameRead] = Field(..., description="Страница записей")
    total: int = Field(..., ge=0, description="Всего записей с учётом search")


class RulesDocumentRead(BaseModel):
    """Схема документа с правилами."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="ID документа")
    game_id: int = Field(..., description="ID игры")
    doc_id: str = Field(..., description="Хэш документа (SHA-256)")
    storage_path: str = Field(..., description="Путь в S3")
    lang: str = Field(..., description="Код языка (ru, en и т.д.)")
    status: str = Field(..., description="Статус: pending, processing, indexed, failed")
    created_at: datetime = Field(..., description="Дата создания")


class UploadRulesResponse(BaseModel):
    """Ответ при загрузке правил."""

    rules_document: RulesDocumentRead = Field(..., description="Созданный документ с правилами")
    task_queued: bool = Field(True, description="Задача поставлена в очередь обработки")
    tasks_url: str = Field(
        "/api/v1/background-tasks",
        description="Путь к странице статусов задач",
    )


class CreateGameWithRulesResponse(BaseModel):
    game: GameRead
    rules_document: RulesDocumentRead | None = None
    task_queued: bool = False
    tasks_url: str = Field(
        "/api/v1/background-tasks",
        description="Путь к странице статусов задач",
    )
