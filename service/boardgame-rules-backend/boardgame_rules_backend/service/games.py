import asyncio
import base64
import csv
import hashlib
import io
from pathlib import Path

import aiofiles

from boardgame_rules_backend.connectors import (delete_all_objects_under_prefix_best_effort,
                                                delete_points_by_rules_document_id,
                                                delete_qdrant_collection_best_effort,
                                                delete_s3_objects_best_effort, download_rules_file,
                                                processed_rules_key, source_content_type,
                                                upload_rules_file, upload_source_file)
from boardgame_rules_backend.connectors.s3 import RULES_S3_PREFIX
from boardgame_rules_backend.exceptions import (EmptyFileError, GameNotFound,
                                                RulesProcessingInProgress, RulesSourceNotFound)
from boardgame_rules_backend.models import RulesDocument, RulesDocumentStatus
from boardgame_rules_backend.repository import GameRepository
from boardgame_rules_backend.schemas.games import (CreateGameWithRulesResponse, GameCreate,
                                                   GameListRead, GameRead, GameUpdate,
                                                   RulesDocumentRead)
from boardgame_rules_backend.schemas.mappers import to_game_read, to_rules_document_read
from boardgame_rules_backend.tasks_app import process_manifest_index_batch, process_rules_document
from boardgame_rules_backend.utils.filenames import build_rules_source_filename

MANIFEST_FILENAME = "index_manifest.csv"
REQUIRED_COLUMNS = {"game_title", "lang", "text_path"}
SUPPORTED_UPLOAD_EXTENSIONS = ("pdf", "txt")


class GameService:
    def __init__(self, game_repo: GameRepository):
        self.game_repo = game_repo

    async def has_any_games(self) -> bool:
        return await self.game_repo.has_any_games()

    async def list_games(
        self,
        skip: int = 0,
        limit: int = 100,
        search: str | None = None,
    ) -> GameListRead:
        games = await self.game_repo.get_games(skip=skip, limit=limit, search=search)
        total = await self.game_repo.count_games(search=search)
        items = [to_game_read(game) for game in games]
        return GameListRead(items=items, total=total)

    async def get_game(self, game_id: int) -> GameRead:
        game = await self.game_repo.get_game_by_id(game_id)
        if not game:
            raise GameNotFound()
        return to_game_read(game)

    async def create_game(self, payload: GameCreate) -> GameRead:
        game = await self.game_repo.create_game(
            title=payload.title,
            source_doc_url=payload.source_doc_url,
        )
        return to_game_read(game)

    async def create_game_with_rules(
        self,
        title: str,
        source_doc_url: str | None = None,
        content: bytes | None = None,
        filename: str | None = None,
        lang: str = "ru",
    ) -> CreateGameWithRulesResponse:
        """
        Create game and optionally enqueue rules for processing.

        Returns (game, rules_doc?, task_queued).
        """
        should_upload_rules = content is not None or filename is not None
        if should_upload_rules:
            if content is None or not content.strip():
                raise EmptyFileError()
            safe_filename = (filename or "rules.txt").strip() or "rules.txt"
            ext = (
                safe_filename.rsplit(".", 1)[-1] if "." in safe_filename else "txt"
            ).lower()
            if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
                raise ValueError(f"Unsupported file format. Use PDF or TXT, got: {ext}")
            filename = safe_filename

        game = await self.game_repo.create_game(
            title=title.strip(),
            source_doc_url=source_doc_url,
        )

        rules_doc: RulesDocumentRead | None = None
        task_queued = False

        if should_upload_rules and content is not None and filename is not None:
            rules_doc = await self.upload_rules(
                game_id=game.id,
                content=content,
                filename=filename,
                lang=lang,
            )
            task_queued = True

        return CreateGameWithRulesResponse(
            game=to_game_read(game),
            rules_document=rules_doc,
            task_queued=task_queued,
        )

    async def update_game(self, game_id: int, payload: GameUpdate) -> GameRead:
        game = await self.game_repo.get_game_by_id(game_id)
        if not game:
            raise GameNotFound()
        updates = payload.model_dump(exclude_unset=True)
        if not updates:
            return to_game_read(game)
        updated = await self.game_repo.update_game(game, updates)
        return to_game_read(updated)

    async def list_game_rules(self, game_id: int) -> list[RulesDocumentRead]:
        game = await self.game_repo.get_game_by_id(game_id)
        if not game:
            raise GameNotFound()
        docs = await self.game_repo.get_rules_by_game_id(game_id)
        return [to_rules_document_read(doc) for doc in docs]

    async def delete_game(self, game_id: int) -> None:
        """Remove game rules from Qdrant/S3, then delete the game row (CASCADE rules_documents)."""
        game = await self.game_repo.get_game_by_id(game_id)
        if not game:
            raise GameNotFound()
        existing = await self.game_repo.get_rules_by_game_id(game_id)
        if any(
            d.status in (RulesDocumentStatus.pending, RulesDocumentStatus.processing)
            for d in existing
        ):
            raise RulesProcessingInProgress()
        await self._cleanup_qdrant_and_s3_for_game_rules(game_id, documents=existing)
        deleted = await self.game_repo.delete_game_by_id(game_id)
        if not deleted:
            raise GameNotFound()

    async def _cleanup_qdrant_and_s3_for_game_rules(
        self,
        game_id: int,
        documents: list[RulesDocument] | None = None,
    ) -> None:
        """
        Remove Qdrant points and unreferenced S3 objects for this game's rules.

        Does not change Postgres (call delete_rules_documents_for_game or delete_game separately).
        """
        docs = documents if documents is not None else await self.game_repo.get_rules_by_game_id(
            game_id
        )
        if not docs:
            return
        for doc in docs:
            await asyncio.to_thread(delete_points_by_rules_document_id, doc.id)
        seen_paths: set[str] = set()
        for doc in docs:
            p = doc.storage_path
            if p in seen_paths:
                continue
            seen_paths.add(p)
            others = await self.game_repo.count_rules_documents_same_storage_other_games(p, game_id)
            if others == 0:
                await asyncio.to_thread(delete_s3_objects_best_effort, [p])
        seen_source_paths: set[str] = set()
        for doc in docs:
            p = doc.source_storage_path
            if not p or p in seen_source_paths:
                continue
            seen_source_paths.add(p)
            others = await self.game_repo.count_rules_documents_same_source_storage_other_games(
                p,
                game_id,
            )
            if others == 0:
                await asyncio.to_thread(delete_s3_objects_best_effort, [p])

    async def upload_rules(
        self,
        game_id: int,
        content: bytes,
        filename: str,
        lang: str = "ru",
    ) -> RulesDocumentRead:
        """Enqueue rules for background processing."""
        game = await self.game_repo.get_game_by_id(game_id)
        if not game:
            raise GameNotFound()

        existing = await self.game_repo.get_rules_by_game_id(game_id)
        if any(
            d.status in (RulesDocumentStatus.pending, RulesDocumentStatus.processing)
            for d in existing
        ):
            raise RulesProcessingInProgress()

        if not content.strip():
            raise EmptyFileError()

        safe_filename = filename.strip() or "rules.txt"
        ext = (safe_filename.rsplit(".", 1)[-1] if "." in safe_filename else "txt").lower()
        if ext not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise ValueError(f"Unsupported file format. Use PDF or TXT, got: {ext}")
        source_filename = build_rules_source_filename(game.title, ext, game_id=game.id)

        if existing:
            await self._cleanup_qdrant_and_s3_for_game_rules(game_id, documents=existing)
            await self.game_repo.delete_rules_documents_for_game(game_id)

        doc_id = hashlib.sha256(content).hexdigest()
        storage_path = processed_rules_key(doc_id, "txt")
        source_storage_path, _ = await asyncio.to_thread(
            upload_source_file,
            content,
            source_filename,
        )

        doc = await self.game_repo.create_rules_document(
            game_id=game_id,
            doc_id=doc_id,
            storage_path=storage_path,
            source_storage_path=source_storage_path,
            source_filename=source_filename,
            lang=lang,
            commit=True,
        )

        content_b64 = base64.b64encode(content).decode("ascii")
        process_rules_document.delay(doc.id, content_base64=content_b64, filename=safe_filename)

        return to_rules_document_read(doc)

    async def get_rules_source(self, game_id: int) -> tuple[bytes, str, str]:
        game = await self.game_repo.get_game_by_id(game_id)
        if not game:
            raise GameNotFound()

        docs = await self.game_repo.get_rules_by_game_id(game_id)
        source_docs = [
            doc for doc in docs if doc.source_storage_path and doc.source_filename
        ]
        if not source_docs:
            raise RulesSourceNotFound()

        latest_doc = max(source_docs, key=lambda doc: (doc.created_at, doc.id))
        if not latest_doc.source_storage_path or not latest_doc.source_filename:
            raise RulesSourceNotFound()

        source_bytes = await asyncio.to_thread(
            download_rules_file,
            latest_doc.source_storage_path,
        )
        return (
            source_bytes,
            latest_doc.source_filename,
            source_content_type(latest_doc.source_filename),
        )

    async def initialize_from_manifest(
        self,
        base_path: Path,
        index: bool = False,
        limit: int | None = None,
    ) -> tuple[int, int]:
        """
        Load games and rules from manifest CSV. Texts are stored as-is; no preprocessing.

        Manifest files are expected to be already pre-cleaned at dataset preparation stage.
        """
        manifest_path = base_path / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        async with aiofiles.open(manifest_path, encoding="utf-8", newline="") as f:
            content = await f.read()

        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)

        if not rows:
            return 0, 0

        missing = REQUIRED_COLUMNS - set(rows[0].keys())
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")

        games_created = 0
        items: list[dict] = []
        uploaded_s3_keys: list[str] = []

        try:
            for row in rows:
                game_title = (row.get("game_title") or "").strip()
                lang = (row.get("lang") or "ru").strip()
                text_path_str = (row.get("text_path") or "").strip()
                source_path_str = (row.get("source_path") or "").strip()
                source_sha256 = (row.get("source_sha256") or "").strip().lower()
                source_mime = (row.get("source_mime") or "").strip().lower()

                if not game_title or not text_path_str:
                    continue

                file_path = (base_path / text_path_str).resolve()
                try:
                    file_path.relative_to(base_path.resolve())
                except ValueError:
                    continue
                if not file_path.exists() or not file_path.is_file():
                    continue

                try:
                    async with aiofiles.open(file_path, "rb") as fp:
                        file_content = await fp.read()
                except OSError:
                    continue

                if not file_content.strip():
                    continue

                game, game_was_created = await self.game_repo.get_or_create_game_by_title(
                    game_title
                )
                if game_was_created:
                    games_created += 1

                source_storage_path: str | None = None
                source_filename: str | None = None
                if source_path_str:
                    source_file_path = (base_path / source_path_str).resolve()
                    try:
                        source_file_path.relative_to(base_path.resolve())
                    except ValueError:
                        continue
                    if source_file_path.exists() and source_file_path.is_file():
                        try:
                            async with aiofiles.open(source_file_path, "rb") as fp:
                                source_content = await fp.read()
                        except OSError:
                            source_content = b""

                        source_ext = source_file_path.suffix.lower().lstrip(".")
                        if source_content and source_ext in SUPPORTED_UPLOAD_EXTENSIONS:
                            if source_mime and source_mime not in {
                                "application/pdf",
                                "text/plain",
                                "text/plain; charset=utf-8",
                            }:
                                raise ValueError(
                                    f"Unsupported source_mime for {source_path_str}: {source_mime}"
                                )
                            source_hash = hashlib.sha256(source_content).hexdigest()
                            if source_sha256 and source_sha256 != source_hash:
                                raise ValueError(
                                    f"source_sha256 mismatch for {source_path_str}: "
                                    f"expected {source_sha256}, got {source_hash}"
                                )
                            source_storage_path, _ = await asyncio.to_thread(
                                upload_source_file,
                                source_content,
                                source_file_path.name,
                            )
                            source_filename = build_rules_source_filename(
                                game.title,
                                source_ext,
                                game_id=game.id,
                            )
                            uploaded_s3_keys.append(source_storage_path)

                # No preprocessing; manifest texts are expected to be pre-cleaned.
                s3_key, doc_id = upload_rules_file(file_content, file_path.name)
                uploaded_s3_keys.append(s3_key)
                items.append({
                    "game_id": game.id,
                    "lang": lang,
                    "storage_path": s3_key,
                    "doc_id": doc_id,
                    "source_storage_path": source_storage_path,
                    "source_filename": source_filename,
                })
                if limit is not None and games_created >= limit:
                    break

            if not items:
                return games_created, 0

            docs_created, created_doc_ids = (
                await self.game_repo.batch_create_from_manifest(items)
            )
        except Exception:
            delete_s3_objects_best_effort(uploaded_s3_keys)
            raise

        if index and created_doc_ids:
            process_manifest_index_batch.delay(list(created_doc_ids))

        return games_created, docs_created

    async def clear_all_games(self) -> int:
        """Remove all games and wipe related Qdrant + S3 rules objects."""
        await asyncio.to_thread(delete_qdrant_collection_best_effort)
        await asyncio.to_thread(delete_all_objects_under_prefix_best_effort, RULES_S3_PREFIX)
        return await self.game_repo.delete_all_games()
