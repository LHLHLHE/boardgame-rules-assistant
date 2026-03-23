from collections.abc import Iterable
from typing import Any, TypedDict

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from boardgame_rules_backend.exceptions import DuplicateGameTitle
from boardgame_rules_backend.models import Game, RulesDocument, RulesDocumentStatus


def is_unique_violation(exc: IntegrityError) -> bool:
    orig = getattr(exc, "orig", None)
    if orig is not None:
        pgcode = getattr(orig, "pgcode", None)
        if pgcode == "23505":
            return True
    msg = str(orig if orig is not None else exc).lower()
    return "unique" in msg or "duplicate key" in msg


class ManifestItem(TypedDict):
    game_id: int
    lang: str
    storage_path: str
    doc_id: str


class GameRepository:
    def __init__(self, pg_db_session: AsyncSession):
        self.pg_db_session = pg_db_session

    async def get_games(
        self, skip: int = 0, limit: int = 100, search: str | None = None
    ) -> list[Game]:
        stmt = select(Game).order_by(Game.title)
        if search and search.strip():
            pattern = f"%{search.strip()}%"
            stmt = stmt.where(Game.title.ilike(pattern))

        stmt = stmt.offset(skip).limit(limit)
        result = await self.pg_db_session.execute(stmt)
        return list(result.scalars().all())

    async def get_game_by_id(self, game_id: int) -> Game | None:
        result = await self.pg_db_session.execute(
            select(Game).where(Game.id == game_id)
        )
        return result.scalar_one_or_none()

    async def has_any_games(self) -> bool:
        result = await self.pg_db_session.execute(select(Game).limit(1))
        return result.scalar_one_or_none() is not None

    async def get_or_create_game_by_title(
        self,
        title: str,
        source_doc_url: str | None = None,
    ) -> tuple[Game, bool]:
        """Return (game, created). created is True if the game was newly created."""
        normalized_title = title.strip() if title else ""
        result = await self.pg_db_session.execute(
            select(Game).where(func.lower(Game.title) == normalized_title.lower())
        )
        game = result.scalar_one_or_none()
        if game:
            return game, False

        try:
            async with self.pg_db_session.begin_nested():
                new_game = await self.create_game(
                    title=normalized_title,
                    source_doc_url=source_doc_url,
                    commit=False,
                    suppress_integrity_error=True,
                )
                return new_game, True
        except IntegrityError as exc:
            if not is_unique_violation(exc):
                raise

        result = await self.pg_db_session.execute(
            select(Game).where(func.lower(Game.title) == normalized_title.lower())
        )
        game = result.scalar_one_or_none()
        if game is None:
            msg = "Expected existing game after unique title conflict"
            raise RuntimeError(msg) from None
        return game, False

    async def get_rules_doc_by_game_and_doc_id(
        self, game_id: int, doc_id: str
    ) -> RulesDocument | None:
        result = await self.pg_db_session.execute(
            select(RulesDocument).where(
                RulesDocument.game_id == game_id,
                RulesDocument.doc_id == doc_id,
            )
        )
        return result.scalar_one_or_none()

    async def delete_all_games(self) -> int:
        result = await self.pg_db_session.execute(delete(Game))
        count = result.rowcount
        await self.pg_db_session.commit()
        return count

    async def delete_game_by_id(self, game_id: int) -> bool:
        """Delete one game; rules_documents rows are removed by ON DELETE CASCADE."""
        result = await self.pg_db_session.execute(delete(Game).where(Game.id == game_id))
        await self.pg_db_session.commit()
        return bool(result.rowcount)

    async def create_game(
        self,
        title: str,
        source_doc_url: str | None = None,
        commit: bool = True,
        suppress_integrity_error: bool = False,
    ) -> Game:
        game = Game(
            title=title,
            source_doc_url=source_doc_url,
        )
        self.pg_db_session.add(game)
        try:
            await self.pg_db_session.flush()
            await self.pg_db_session.refresh(game)
        except IntegrityError:
            if suppress_integrity_error:
                raise
            await self.pg_db_session.rollback()
            raise DuplicateGameTitle from None
        if commit:
            await self.pg_db_session.commit()

        return game

    async def update_game(self, game: Game, updates: dict[str, Any]) -> Game:
        if not updates:
            return game
        if "title" in updates:
            game.title = updates["title"]
        if "source_doc_url" in updates:
            game.source_doc_url = updates["source_doc_url"]

        try:
            await self.pg_db_session.flush()
            await self.pg_db_session.refresh(game)
            await self.pg_db_session.commit()
        except IntegrityError:
            await self.pg_db_session.rollback()
            raise DuplicateGameTitle from None
        return game

    async def get_rules_by_game_id(self, game_id: int) -> list[RulesDocument]:
        result = await self.pg_db_session.execute(
            select(RulesDocument).where(RulesDocument.game_id == game_id)
        )
        return list(result.scalars().all())

    async def count_rules_documents_same_storage_other_games(
        self, storage_path: str, game_id: int
    ) -> int:
        """Rows referencing this S3 key from games other than ``game_id`` (for refcount)."""
        result = await self.pg_db_session.execute(
            select(func.count())
            .select_from(RulesDocument)
            .where(
                RulesDocument.storage_path == storage_path,
                RulesDocument.game_id != game_id,
            )
        )
        return int(result.scalar_one())

    async def delete_rules_documents_for_game(self, game_id: int) -> None:
        """Remove all rules_documents rows for this game (after Qdrant/S3 cleanup)."""
        await self.pg_db_session.execute(
            delete(RulesDocument).where(RulesDocument.game_id == game_id)
        )
        await self.pg_db_session.commit()

    async def create_rules_document(
        self,
        game_id: int,
        doc_id: str,
        storage_path: str,
        lang: str = "ru",
        commit: bool = False,
    ) -> RulesDocument:
        doc = RulesDocument(
            game_id=game_id,
            doc_id=doc_id,
            storage_path=storage_path,
            lang=lang,
            status=RulesDocumentStatus.pending,
        )
        self.pg_db_session.add(doc)
        await self.pg_db_session.flush()
        await self.pg_db_session.refresh(doc)
        if commit:
            await self.pg_db_session.commit()

        return doc

    async def batch_create_from_manifest(
        self, items: Iterable[ManifestItem]
    ) -> tuple[int, list[int]]:
        docs_created = 0
        created_doc_ids: list[int] = []
        for item in items:
            if await self.get_rules_doc_by_game_and_doc_id(item["game_id"], item["doc_id"]):
                continue

            doc = await self.create_rules_document(
                game_id=item["game_id"],
                doc_id=item["doc_id"],
                storage_path=item["storage_path"],
                lang=item["lang"],
            )
            docs_created += 1
            created_doc_ids.append(doc.id)

        await self.pg_db_session.commit()
        return docs_created, created_doc_ids
