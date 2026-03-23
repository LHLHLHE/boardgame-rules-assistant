from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Запрос вопроса по правилам игры."""

    game_id: int = Field(..., description="Идентификатор игры (как в GET /api/v1/games)")
    query: str = Field(..., description="Вопрос по правилам")


class QuestionResponse(BaseModel):
    """Ответ на вопрос по правилам."""

    answer: str = Field(..., description="Ответ, сформированный RAG на основе правил")
