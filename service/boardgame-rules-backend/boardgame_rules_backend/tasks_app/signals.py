import json
import logging
from datetime import datetime, timezone
from typing import Any

from celery import signals
from celery.exceptions import Retry
from sqlalchemy import select

from boardgame_rules_backend.database import get_sync_pg_db_session
from boardgame_rules_backend.models import BackgroundTask, BackgroundTaskState

logger = logging.getLogger(__name__)

MAX_KWARGS_SNAPSHOT = 4000
MAX_RESULT = 4000
MAX_ERROR = 8000

PROCESS_RULES_SUFFIX = "process_rules_document"
PROCESS_MANIFEST_BATCH_SUFFIX = "process_manifest_index_batch"


def task_id_from_signal(task_id: str | None, sender: Any) -> str | None:
    if task_id:
        return str(task_id)
    if sender is None:
        return None
    req = getattr(sender, "request", None)
    if req is not None:
        tid = getattr(req, "id", None)
        if tid is not None:
            return str(tid)
    return None


def sanitize_kwargs(kwargs: dict[str, Any] | None) -> dict[str, Any]:
    if not kwargs:
        return {}
    out: dict[str, Any] = {}
    for key, val in kwargs.items():
        if key == "content_base64":
            out[key] = "<omitted>"
        else:
            out[key] = val
    return out


def kwargs_snapshot(kwargs: dict[str, Any] | None) -> str | None:
    safe = sanitize_kwargs(kwargs)
    try:
        raw = json.dumps(safe, default=str, ensure_ascii=False)
    except TypeError:
        raw = str(safe)
    if len(raw) > MAX_KWARGS_SNAPSHOT:
        return raw[:MAX_KWARGS_SNAPSHOT] + "…"
    return raw


def related_entity(
    task_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> tuple[str | None, int | None]:
    if PROCESS_MANIFEST_BATCH_SUFFIX in task_name and args:
        first = args[0]
        if isinstance(first, list) and first:
            return "manifest_index_batch", len(first)
    if PROCESS_RULES_SUFFIX in task_name and args:
        try:
            return "rules_document", int(args[0])
        except (TypeError, ValueError):
            return None, None
    jc = (kwargs or {}).get("job_context")
    if isinstance(jc, dict):
        kind = jc.get("kind")
        entity_id = jc.get("id")
        if isinstance(kind, str) and isinstance(entity_id, int):
            return kind, entity_id
    return None, None


def serialize_result(result: Any) -> str:
    try:
        raw = json.dumps(result, default=str, ensure_ascii=False)
    except TypeError:
        raw = str(result)
    if len(raw) > MAX_RESULT:
        return raw[:MAX_RESULT] + "…"
    return raw


def serialize_error(exc: BaseException | None, einfo: Any) -> str:
    if einfo is not None:
        tb = getattr(einfo, "traceback", None)
        if tb is not None:
            s = str(tb)
            return s if len(s) <= MAX_ERROR else s[:MAX_ERROR] + "…"
    if exc is not None:
        s = str(exc)
        return s if len(s) <= MAX_ERROR else s[:MAX_ERROR] + "…"
    return "unknown error"


@signals.task_prerun.connect
def on_task_prerun(
    sender: Any = None,
    task_id: str | None = None,
    task: Any = None,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    **_extra: Any,
) -> None:
    tid = task_id or (getattr(getattr(task, "request", None), "id", None) if task else None)
    if not tid:
        return
    tid = str(tid)
    task_name = ""
    if task is not None:
        task_name = getattr(task, "name", None) or ""
    if not task_name and sender is not None:
        task_name = getattr(sender, "name", "") or ""
    args = args or ()
    kwargs = kwargs or {}
    rel_type, rel_id = related_entity(task_name, args, kwargs)
    snap = kwargs_snapshot(kwargs)
    try:
        with get_sync_pg_db_session() as session:
            row = BackgroundTask(
                celery_task_id=tid,
                task_name=task_name,
                state=BackgroundTaskState.STARTED,
                kwargs_snapshot=snap,
                related_entity_type=rel_type,
                related_entity_id=rel_id,
            )
            session.add(row)
            session.commit()
    except Exception:
        logger.exception("background_tasks: failed to record task_prerun for %s", tid)


@signals.task_success.connect
def on_task_success(sender: Any = None, result: Any = None, **kwargs: Any) -> None:
    # task_success officially provides only `result`; task_id comes from sender.request.id.
    tid = task_id_from_signal(kwargs.get("task_id"), sender)
    if not tid:
        return
    summary = serialize_result(result)
    try:
        with get_sync_pg_db_session() as session:
            row = session.execute(
                select(BackgroundTask).where(BackgroundTask.celery_task_id == tid)
            ).scalar_one_or_none()
            if row is None:
                logger.warning(
                    "background_tasks: no row for celery_task_id=%s (task_success); "
                    "task_prerun may have failed",
                    tid,
                )
                return
            row.state = BackgroundTaskState.SUCCESS
            row.result_summary = summary
            row.finished_at = datetime.now(timezone.utc)
            session.commit()
    except Exception:
        logger.exception("background_tasks: failed to record task_success for %s", tid)


@signals.task_failure.connect
def on_task_failure(
    sender: Any = None,
    task_id: str | None = None,
    exception: BaseException | None = None,
    einfo: Any = None,
    **kwargs: Any,
) -> None:
    if isinstance(exception, Retry):
        # Retries are recorded by task_retry; avoid marking FAILURE before a retry run.
        return
    tid = task_id_from_signal(task_id, sender)
    if not tid:
        return
    err = serialize_error(exception, einfo)
    try:
        with get_sync_pg_db_session() as session:
            row = session.execute(
                select(BackgroundTask).where(BackgroundTask.celery_task_id == tid)
            ).scalar_one_or_none()
            if row is None:
                logger.warning(
                    "background_tasks: no row for celery_task_id=%s (task_failure); "
                    "task_prerun may have failed",
                    tid,
                )
                return
            row.state = BackgroundTaskState.FAILURE
            row.error_message = err
            row.finished_at = datetime.now(timezone.utc)
            session.commit()
    except Exception:
        logger.exception("background_tasks: failed to record task_failure for %s", tid)


@signals.task_retry.connect
def on_task_retry(
    sender: Any = None,
    request: Any = None,
    **kwargs: Any,
) -> None:
    tid = None
    if request is not None:
        tid = getattr(request, "id", None)
    tid = str(tid) if tid else task_id_from_signal(None, sender)
    if not tid:
        return
    try:
        with get_sync_pg_db_session() as session:
            row = session.execute(
                select(BackgroundTask).where(BackgroundTask.celery_task_id == tid)
            ).scalar_one_or_none()
            if row is None:
                logger.warning(
                    "background_tasks: no row for celery_task_id=%s (task_retry); "
                    "task_prerun may have failed",
                    tid,
                )
                return
            row.state = BackgroundTaskState.RETRY
            session.commit()
    except Exception:
        logger.exception("background_tasks: failed to record task_retry for %s", tid)


@signals.task_revoked.connect
def on_task_revoked(
    sender: Any = None,
    request: Any = None,
    terminated: bool = False,
    signum: Any = None,
    expired: bool = False,
    **kwargs: Any,
) -> None:
    tid = None
    if request is not None:
        tid = getattr(request, "id", None)
    tid = tid or kwargs.get("task_id")
    if not tid:
        return
    tid = str(tid)
    try:
        with get_sync_pg_db_session() as session:
            row = session.execute(
                select(BackgroundTask).where(BackgroundTask.celery_task_id == tid)
            ).scalar_one_or_none()
            if row is None:
                logger.warning(
                    "background_tasks: no row for celery_task_id=%s (task_revoked); "
                    "task_prerun may have failed",
                    tid,
                )
                return
            row.state = BackgroundTaskState.REVOKED
            row.finished_at = datetime.now(timezone.utc)
            row.error_message = (
                f"revoked (terminated={terminated}, expired={expired}, signum={signum})"
            )
            session.commit()
    except Exception:
        logger.exception("background_tasks: failed to record task_revoked for %s", tid)
