import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException
from fastapi import Path as PathParam
from fastapi import Query, UploadFile

from boardgame_rules_backend.api.utils import extract_zip_safely
from boardgame_rules_backend.dependencies import (get_game_service, require_admin,
                                                  require_bot_or_moderator, require_moderator)
from boardgame_rules_backend.exceptions import EmptyFileError, GameNotFound
from boardgame_rules_backend.schemas import (AuthUser, CreateGameWithRulesResponse, GameCreate,
                                             GameListRead, GameRead, GameUpdate, RulesDocumentRead,
                                             UploadRulesResponse)
from boardgame_rules_backend.service import GameService

router = APIRouter()
TASKS_URL = "/api/v1/background-tasks"


@router.post(
    "/initialize",
    summary="Инициализация из манифеста",
    description=(
        "Загружает игры и правила из CSV-манифеста и ZIP-архива. "
        "manifest: CSV с game_title, lang, text_path (идентификатор документа — хэш содержимого). "
        "archive: ZIP. limit: макс. новых игр (опционально)."
    ),
    responses={
        200: {"description": "Загрузка выполнена"},
        400: {"description": "Неверный формат (не CSV/ZIP) или ошибка чтения архива"},
        409: {"description": "БД не пуста и limit не указан — нужно очистить или указать limit"},
    },
)
async def initialize_from_upload(
    manifest: UploadFile = File(
        ..., description="CSV-файл манифеста (index_manifest.csv)"
    ),
    archive: UploadFile = File(..., description="ZIP-архив с текстами правил"),
    limit: int | None = Query(
        None, description="Макс. новых игр (без limit — только при пустой БД)"
    ),
    _auth: AuthUser = Depends(require_admin),
    game_service: GameService = Depends(get_game_service),
):
    if not manifest.filename or not manifest.filename.lower().endswith(".csv"):
        raise HTTPException(400, "manifest must be a CSV file")
    if not archive.filename or not archive.filename.lower().endswith(".zip"):
        raise HTTPException(400, "archive must be a ZIP file")

    if limit is None and await game_service.has_any_games():
        raise HTTPException(
            409,
            "Database already has games. Clear data first if you want to re-initialize.",
        )

    tmpdir = Path(tempfile.mkdtemp(prefix="admin_initialize_"))
    try:
        manifest_path = tmpdir / "index_manifest.csv"
        manifest_content = await manifest.read()
        manifest_path.write_bytes(manifest_content)

        archive_content = await archive.read()
        extract_zip_safely(archive_content, tmpdir)

        games_created, docs_created = await game_service.initialize_from_manifest(
            tmpdir,
            index=True,
            limit=limit,
        )
        return {
            "status": "ok",
            "games_created": games_created,
            "rules_documents_created": docs_created,
        }
    except zipfile.BadZipFile as e:
        raise HTTPException(400, f"Invalid ZIP archive: {e}") from e
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@router.post(
    "/clear-games",
    summary="Удаление всех игр",
    description=(
        "Удаляет все игры и связанные документы правил в PostgreSQL (CASCADE), "
        "очищает коллекцию Qdrant с правилами и объекты в S3 под префиксом rules/."
    ),
    responses={200: {"description": "Игры удалены"}},
)
async def clear_games(
    _auth: AuthUser = Depends(require_admin),
    game_service: GameService = Depends(get_game_service),
):
    deleted = await game_service.clear_all_games()
    return {"status": "ok", "games_deleted": deleted}


@router.get(
    "",
    response_model=GameListRead,
    summary="Список игр",
    description=(
        "Возвращает список игр с пагинацией. Параметр search - поиск по названию. "
        "Доступ: заголовок X-Bot-Token (телеграм-бот) или JWT модератора/админа (админ-панель)."
    ),
    responses={
        200: {"description": "Список игр"},
        401: {"description": "Нет X-Bot-Token и нет/неверный JWT"},
        403: {"description": "JWT без прав модератора"},
    },
)
async def list_games(
    skip: int = Query(0, ge=0, description="Пропустить N записей"),
    limit: int = Query(100, ge=1, le=200, description="Макс. количество записей"),
    search: str | None = Query(None, description="Поиск по названию (подстрока)"),
    _auth: AuthUser | None = Depends(require_bot_or_moderator),
    game_service: GameService = Depends(get_game_service),
):
    return await game_service.list_games(skip=skip, limit=limit, search=search)


@router.post(
    "",
    response_model=GameRead,
    summary="Создание игры",
    description="Создаёт игру. Правила — отдельно или через POST /with-rules.",
    responses={
        200: {"description": "Игра создана"},
        409: {"description": "Игра с таким названием уже есть (без учёта регистра)"},
    },
)
async def create_game(
    payload: GameCreate,
    _auth: AuthUser = Depends(require_moderator),
    game_service: GameService = Depends(get_game_service),
):
    return await game_service.create_game(payload)


@router.post(
    "/with-rules",
    response_model=CreateGameWithRulesResponse,
    summary="Создание игры с правилами",
    description="""
Создаёт игру и при наличии файла загружает правила в одном запросе. multipart/form-data.
При загрузке файла ставит задачу в очередь; ответ содержит task_queued и tasks_url.
""",
    responses={
        200: {"description": "Игра создана, при файле — задача в очереди"},
        400: {"description": "Пустой файл или неподдерживаемый формат (только PDF/TXT)"},
        409: {"description": "Игра с таким названием уже есть (без учёта регистра)"},
    },
)
async def create_game_with_rules(
    title: str = Form(..., description="Название игры"),
    source_doc_url: str | None = Form(None, description="Ссылка на источник"),
    file: UploadFile | None = File(None, description="Файл правил (.pdf или .txt, опционально)"),
    lang: str = Form("ru", description="Код языка"),
    _auth: AuthUser = Depends(require_moderator),
    game_service: GameService = Depends(get_game_service),
):
    content = None
    filename = None
    if file is not None:
        content = await file.read()
        filename = file.filename or "rules.txt"

    try:
        return await game_service.create_game_with_rules(
            title=title,
            source_doc_url=source_doc_url or None,
            content=content,
            filename=filename,
            lang=lang,
        )
    except EmptyFileError:
        raise HTTPException(status_code=400, detail="Empty file")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get(
    "/{game_id}",
    response_model=GameRead,
    summary="Получение игры",
    description="Возвращает игру по ID.",
    responses={
        200: {"description": "Игра найдена"},
        404: {"description": "Игра не найдена"},
    },
)
async def get_game(
    game_id: int = PathParam(..., description="ID игры"),
    _auth: AuthUser | None = Depends(require_bot_or_moderator),
    game_service: GameService = Depends(get_game_service),
):
    try:
        return await game_service.get_game(game_id)
    except GameNotFound:
        raise HTTPException(status_code=404, detail="Game not found")


@router.patch(
    "/{game_id}",
    response_model=GameRead,
    summary="Обновление игры",
    description="Обновляет название и/или ссылку на источник. Передавать только изменяемые поля.",
    responses={
        200: {"description": "Игра обновлена"},
        404: {"description": "Игра не найдена"},
        409: {"description": "Другое название уже занято (без учёта регистра)"},
    },
)
async def update_game(
    game_id: int = PathParam(..., description="ID игры"),
    payload: GameUpdate = Body(...),
    _auth: AuthUser = Depends(require_moderator),
    game_service: GameService = Depends(get_game_service),
):
    try:
        return await game_service.update_game(game_id, payload)
    except GameNotFound:
        raise HTTPException(status_code=404, detail="Game not found")


@router.delete(
    "/{game_id}",
    summary="Удаление игры",
    description=(
        "Удаляет игру и связанные документы правил в PostgreSQL (CASCADE), "
        "точки Qdrant и объекты S3 для файлов, которые больше не ссылаются на другие игры."
    ),
    responses={
        200: {"description": "Игра удалена"},
        404: {"description": "Игра не найдена"},
        409: {"description": "Правила в очереди или индексируются"},
    },
)
async def delete_game(
    game_id: int = PathParam(..., description="ID игры"),
    _auth: AuthUser = Depends(require_moderator),
    game_service: GameService = Depends(get_game_service),
):
    await game_service.delete_game(game_id)
    return {"status": "ok"}


@router.post(
    "/{game_id}/rules",
    response_model=UploadRulesResponse,
    summary="Загрузить правила",
    description=(
        "Загружает файл правил (PDF/TXT). Обработка в фоне: "
        "извлечение текста, препроцессинг, индексация."
    ),
    responses={
        200: {"description": "Файл принят, задача поставлена в очередь"},
        400: {"description": "Пустой файл или неподдерживаемый формат"},
        404: {"description": "Игра не найдена"},
        409: {
            "description": (
                "Правила в очереди на индексацию или обрабатываются; "
                "дождитесь статуса «Проиндексировано» или «Ошибка»"
            ),
        },
    },
)
async def upload_rules(
    game_id: int = PathParam(..., description="ID игры"),
    file: UploadFile = File(..., description="Файл правил (.pdf или .txt)"),
    lang: str = Query("ru", description="Код языка"),
    _auth: AuthUser = Depends(require_moderator),
    game_service: GameService = Depends(get_game_service),
):
    try:
        content = await file.read()
        doc = await game_service.upload_rules(
            game_id=game_id,
            content=content,
            filename=file.filename or "rules.txt",
            lang=lang,
        )
        return UploadRulesResponse(rules_document=doc, task_queued=True, tasks_url=TASKS_URL)
    except EmptyFileError:
        raise HTTPException(status_code=400, detail="Empty file")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except GameNotFound:
        raise HTTPException(status_code=404, detail="Game not found")


@router.get(
    "/{game_id}/rules",
    response_model=list[RulesDocumentRead],
    summary="Список правил игры",
    description="Возвращает все загруженные документы с правилами для указанной игры.",
    responses={
        200: {"description": "Список документов с правилами"},
        404: {"description": "Игра не найдена"},
    },
)
async def list_game_rules(
    game_id: int = PathParam(..., description="ID игры"),
    _auth: AuthUser = Depends(require_moderator),
    game_service: GameService = Depends(get_game_service),
):
    try:
        return await game_service.list_game_rules(game_id)
    except GameNotFound:
        raise HTTPException(status_code=404, detail="Game not found")
