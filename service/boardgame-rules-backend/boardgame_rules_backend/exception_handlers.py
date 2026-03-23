from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette import status
from starlette.requests import Request

from boardgame_rules_backend.exceptions import (AuthServiceError, DuplicateGameTitle, GameNotFound,
                                                RulesProcessingInProgress)


def game_not_found_handler(request: Request, exc: GameNotFound) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": exc.detail},
    )


def duplicate_game_title_handler(request: Request, exc: DuplicateGameTitle) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": exc.detail},
    )


def rules_processing_in_progress_handler(
    request: Request,
    exc: RulesProcessingInProgress
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"detail": exc.detail},
    )


def auth_service_error_handler(request: Request, exc: AuthServiceError) -> JSONResponse:
    headers = {}
    if exc.status_code == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(GameNotFound, game_not_found_handler)
    app.add_exception_handler(DuplicateGameTitle, duplicate_game_title_handler)
    app.add_exception_handler(RulesProcessingInProgress, rules_processing_in_progress_handler)
    app.add_exception_handler(AuthServiceError, auth_service_error_handler)
