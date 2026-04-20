import logging

from boardgame_rules_backend.exceptions import GameNotFound
from boardgame_rules_backend.rag import Generator, Retriever
from boardgame_rules_backend.repository import GameRepository
from boardgame_rules_backend.settings import app_config, truncate_for_log

logger = logging.getLogger(__name__)


class QAService:
    def __init__(self, retriever: Retriever, generator: Generator, game_repo: GameRepository):
        self.retriever = retriever
        self.generator = generator
        self.game_repo = game_repo

    async def get_answer(self, game_id: int, query: str, history: str | None = None) -> str:
        game = await self.game_repo.get_game_by_id(game_id)
        if not game:
            raise GameNotFound()

        max_c = app_config.rag_log_max_chars
        if app_config.rag_debug_log:
            logger.info(
                "[RAG] request game_id=%s title=%r query_preview=%r",
                game.id,
                game.title,
                truncate_for_log(query, max_c),
            )

        context = await self.retriever.retrieve_context(
            game_id=game.id,
            display_title=game.title,
            query=query,
        )

        if app_config.rag_debug_log:
            placeholder = context.startswith("Информация не найдена.")
            logger.info(
                "[RAG] context chars=%s placeholder=%s preview=%r",
                len(context),
                placeholder,
                truncate_for_log(context, max_c),
            )

        answer = await self.generator.generate(query=query, context=context, history=history)

        if app_config.rag_debug_log:
            logger.info(
                "[RAG] answer chars=%s preview=%r",
                len(answer),
                truncate_for_log(answer, max_c),
            )

        return answer
