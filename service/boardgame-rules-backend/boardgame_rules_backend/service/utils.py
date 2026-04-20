from pathlib import Path

from boardgame_rules_backend.database.postgres import PGAsyncSessionFactory
from boardgame_rules_backend.repository import GameRepository
from boardgame_rules_backend.service.games import GameService


async def run_load_initial_data(
    base_path: Path,
    index: bool,
    limit: int | None = None,
) -> tuple[int, int]:
    """Load games and rules from manifest. Returns (games_created, docs_created)."""
    async with PGAsyncSessionFactory() as session:
        game_repo = GameRepository(session)
        game_service = GameService(game_repo)
        return await game_service.initialize_from_manifest(
            base_path,
            index=index,
            limit=limit,
        )
