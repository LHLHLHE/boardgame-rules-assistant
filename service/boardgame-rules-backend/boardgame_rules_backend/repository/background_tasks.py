from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from boardgame_rules_backend.models import BackgroundTask, Game, RulesDocument


class BackgroundTaskRepository:
    def __init__(self, pg_db_session: AsyncSession):
        self.pg_db_session = pg_db_session

    async def list_with_rules_context(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> list[tuple[BackgroundTask, str | None, str | None, int | None]]:
        """
        Return (BackgroundTask, doc_id, game_title, game_id) for rules_document links; else Nones.

        Sorted by started_at descending.
        """
        stmt = (
            select(BackgroundTask, RulesDocument.doc_id, Game.title, RulesDocument.game_id)
            .outerjoin(
                RulesDocument,
                (BackgroundTask.related_entity_type == "rules_document")
                & (BackgroundTask.related_entity_id == RulesDocument.id),
            )
            .outerjoin(Game, RulesDocument.game_id == Game.id)
            .order_by(BackgroundTask.started_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.pg_db_session.execute(stmt)
        return list(result.all())
