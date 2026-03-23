from fastapi import APIRouter, Depends, Query

from boardgame_rules_backend.dependencies import get_background_task_service, require_moderator
from boardgame_rules_backend.schemas import AuthUser, BackgroundTaskRead
from boardgame_rules_backend.service import BackgroundTaskService

router = APIRouter()


@router.get(
    "",
    response_model=list[BackgroundTaskRead],
    summary="Журнал фоновых задач Celery",
    description=(
        "Возвращает записи о запусках задач (сигналы task_*). "
        "Сортировка по времени старта (новые первые). "
        "Для задач обработки правил добавляются название игры и doc_id."
    ),
    responses={200: {"description": "Список задач"}},
)
async def list_background_tasks(
    skip: int = Query(0, ge=0, description="Пропустить N записей"),
    limit: int = Query(100, ge=1, le=200, description="Макс. количество записей"),
    _auth: AuthUser = Depends(require_moderator),
    task_service: BackgroundTaskService = Depends(get_background_task_service),
):
    return await task_service.list_tasks(skip=skip, limit=limit)
