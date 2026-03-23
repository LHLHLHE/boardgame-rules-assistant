from boardgame_rules_backend.repository import BackgroundTaskRepository
from boardgame_rules_backend.schemas.background_tasks import BackgroundTaskRead

MANIFEST_INDEX_BATCH_TYPE = "manifest_index_batch"


class BackgroundTaskService:
    def __init__(self, task_repo: BackgroundTaskRepository):
        self.task_repo = task_repo

    async def list_tasks(self, skip: int = 0, limit: int = 100) -> list[BackgroundTaskRead]:
        rows = await self.task_repo.list_with_rules_context(skip=skip, limit=limit)
        out: list[BackgroundTaskRead] = []
        for task, doc_id, game_title, game_id in rows:
            gt, gid, did = game_title, game_id, doc_id
            if (
                task.related_entity_type == MANIFEST_INDEX_BATCH_TYPE
                and task.related_entity_id is not None
            ):
                n = task.related_entity_id
                gt = f"Манифест ({n} документов)"
                gid = None
                did = None
            out.append(
                BackgroundTaskRead(
                    id=task.id,
                    celery_task_id=task.celery_task_id,
                    task_name=task.task_name,
                    state=task.state.value,
                    started_at=task.started_at,
                    finished_at=task.finished_at,
                    error_message=task.error_message,
                    result_summary=task.result_summary,
                    related_entity_type=task.related_entity_type,
                    related_entity_id=task.related_entity_id,
                    game_title=gt,
                    game_id=gid,
                    doc_id=did,
                )
            )
        return out
