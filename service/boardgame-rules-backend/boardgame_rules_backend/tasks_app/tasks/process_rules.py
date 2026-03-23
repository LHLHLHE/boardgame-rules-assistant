import base64
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from boardgame_rules_backend.connectors import download_rules_file, put_rules_content
from boardgame_rules_backend.database import get_sync_pg_db_session
from boardgame_rules_backend.models import Game, RulesDocument, RulesDocumentStatus
from boardgame_rules_backend.rag import Indexer
from boardgame_rules_backend.tasks_app.celery import celery_app
from boardgame_rules_backend.utils.preprocessing import extract_text_from_pdf, preprocess_rules_text


def index_rules_document_from_s3_sync(
    session: Session,
    rules_document_id: int,
    skip_if_indexed: bool = True,
) -> dict:
    """
    Download preprocessed rules from S3, chunk, embed to Qdrant, set status to indexed.

    Used for manifest flow and after upload once raw text has been written to S3.
    """
    result = session.execute(
        select(RulesDocument, Game).join(Game, RulesDocument.game_id == Game.id).where(
            RulesDocument.id == rules_document_id
        )
    )
    row = result.one_or_none()
    if not row:
        return {
            "status": "failed",
            "error": "Document not found",
            "rules_document_id": rules_document_id,
        }

    doc, game = row
    if skip_if_indexed and doc.status == RulesDocumentStatus.indexed:
        return {
            "status": "skipped",
            "reason": "already_indexed",
            "rules_document_id": rules_document_id,
        }

    storage_path = doc.storage_path
    doc_id = doc.doc_id
    lang = doc.lang

    doc.status = RulesDocumentStatus.processing
    session.commit()

    try:
        preprocessed = download_rules_file(storage_path).decode("utf-8")

        indexer = Indexer()
        chunks = indexer.chunk_text(
            preprocessed,
            doc_id,
            doc.id,
            game.id,
            lang,
        )
        indexer.embed_and_upsert(chunks)

        doc = session.get(RulesDocument, rules_document_id)
        if doc:
            doc.status = RulesDocumentStatus.indexed
        session.commit()

        return {"status": "indexed", "chunks": len(chunks), "rules_document_id": rules_document_id}
    except Exception:
        session.rollback()
        doc = session.get(RulesDocument, rules_document_id)
        if doc:
            doc.status = RulesDocumentStatus.failed
        session.commit()
        raise


@celery_app.task(bind=True)
def process_rules_document(
    self,
    rules_document_id: int,
    content_base64: str | None = None,
    filename: str | None = None,
) -> dict:
    """
    Process a rules document in background.

    If content_base64 and filename are provided (upload flow): extract text, preprocess,
    save to S3, then chunk and index. Raw is never saved.

    If not provided (manifest flow): download preprocessed text from S3 and index only.
    """
    is_upload = content_base64 is not None and filename is not None

    with get_sync_pg_db_session() as session:
        result = session.execute(
            select(RulesDocument, Game).join(Game, RulesDocument.game_id == Game.id).where(
                RulesDocument.id == rules_document_id
            )
        )
        row = result.one_or_none()
        if not row:
            return {"status": "failed", "error": "Document not found"}

        doc, _game = row
        storage_path = doc.storage_path

        if is_upload:
            doc.status = RulesDocumentStatus.processing
            session.commit()
            try:
                content = base64.b64decode(content_base64)
                ext = (Path(filename).suffix or ".txt").lstrip(".").lower()

                if ext == "pdf":
                    raw_text = extract_text_from_pdf(content)
                else:
                    raw_text = content.decode("utf-8", errors="replace")

                preprocessed = preprocess_rules_text(raw_text)
                if not preprocessed.strip():
                    raise ValueError("Preprocessed text is empty")

                put_rules_content(storage_path, preprocessed.encode("utf-8"))
            except Exception:
                session.rollback()
                doc = session.get(RulesDocument, rules_document_id)
                if doc:
                    doc.status = RulesDocumentStatus.failed
                session.commit()
                raise

        return index_rules_document_from_s3_sync(
            session,
            rules_document_id,
            skip_if_indexed=not is_upload,
        )


@celery_app.task(
    bind=True,
    soft_time_limit=60 * 60,
    time_limit=60 * 60,
)
def process_manifest_index_batch(self, rules_document_ids: list[int]) -> dict:
    """
    Index many rules documents from S3 in one Celery task (manifest initialization).

    Per-document errors are recorded; processing continues for remaining ids.
    Already indexed documents are skipped (idempotent retry).
    """
    results: list[dict] = []
    for rules_document_id in rules_document_ids:
        with get_sync_pg_db_session() as session:
            try:
                out = index_rules_document_from_s3_sync(
                    session,
                    rules_document_id,
                    skip_if_indexed=True,
                )
                results.append(out)
            except Exception as exc:
                results.append(
                    {
                        "status": "failed",
                        "rules_document_id": rules_document_id,
                        "error": str(exc),
                    }
                )
    return {
        "status": "ok",
        "count": len(results),
        "results": results,
    }
