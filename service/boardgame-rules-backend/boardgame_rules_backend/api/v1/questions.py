from typing import Annotated

from fastapi import APIRouter, Depends

from boardgame_rules_backend.dependencies import get_qa_service, require_bot_api_token
from boardgame_rules_backend.schemas import QuestionRequest, QuestionResponse
from boardgame_rules_backend.service import QAService

router = APIRouter()


@router.post(
    "/ask",
    dependencies=[Depends(require_bot_api_token)],
    response_model=QuestionResponse,
    summary="Задать вопрос по правилам",
    description=(
        "Отвечает на вопрос по правилам игры. RAG: поиск в индексе + LLM. "
        "game_id — как в списке игр. Правила должны быть indexed. "
        "Только для телеграм-бота: заголовок X-Bot-Token."
    ),
    responses={
        200: {"description": "Ответ сгенерирован"},
        401: {"description": "Нет или неверный X-Bot-Token"},
        503: {"description": "BOT_API_TOKEN не задан на сервере"},
        404: {"description": "Игра не найдена или правила не проиндексированы"},
    },
)
async def ask_question(
    payload: QuestionRequest,
    rag_service: Annotated[QAService, Depends(get_qa_service)],
) -> QuestionResponse:
    answer = await rag_service.get_answer(payload.game_id, payload.query, payload.history)
    return QuestionResponse(answer=answer)
