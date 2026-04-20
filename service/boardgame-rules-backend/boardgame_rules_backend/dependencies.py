from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from boardgame_rules_backend.database import get_pg_db_session
from boardgame_rules_backend.rag import Generator, Retriever
from boardgame_rules_backend.repository import (BackgroundTaskRepository, GameRepository,
                                                UserRepository)
from boardgame_rules_backend.schemas.auth import AuthUser
from boardgame_rules_backend.service import (AuthService, BackgroundTaskService, GameService,
                                             QAService)
from boardgame_rules_backend.settings import app_config

security = HTTPBearer(auto_error=False)


def bot_token_valid(received: str | None) -> bool:
    expected = (app_config.bot_api_token or "").strip()
    if not expected:
        return False
    return (received or "").strip() == expected


def require_bot_api_token(x_bot_token: Annotated[str | None, Header()] = None) -> None:
    """Require valid X-Bot-Token."""
    if not (app_config.bot_api_token or "").strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bot API token is not configured on the server",
        )
    if not bot_token_valid(x_bot_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Bot-Token",
        )


def get_game_repository(
    pg_db_session: AsyncSession = Depends(get_pg_db_session),
) -> GameRepository:
    return GameRepository(pg_db_session=pg_db_session)


def get_background_task_repository(
    pg_db_session: AsyncSession = Depends(get_pg_db_session),
) -> BackgroundTaskRepository:
    return BackgroundTaskRepository(pg_db_session=pg_db_session)


def get_user_repository(
    pg_db_session: AsyncSession = Depends(get_pg_db_session),
) -> UserRepository:
    return UserRepository(pg_db_session=pg_db_session)


def get_game_service(
    game_repo: GameRepository = Depends(get_game_repository),
) -> GameService:
    return GameService(game_repo=game_repo)


def get_background_task_service(
    task_repo: BackgroundTaskRepository = Depends(get_background_task_repository),
) -> BackgroundTaskService:
    return BackgroundTaskService(task_repo=task_repo)


def get_auth_service(
    user_repo: UserRepository = Depends(get_user_repository),
) -> AuthService:
    return AuthService(user_repo=user_repo)


async def require_bot_or_moderator(
    x_bot_token: Annotated[str | None, Header()] = None,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(security)
    ] = None,
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthUser | None:
    if bot_token_valid(x_bot_token):
        return None
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Send X-Bot-Token or Authorization: Bearer (moderator)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = await auth_service.get_user_from_access_token(credentials.credentials)
    if not (user.is_staff or user.is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Moderator or admin access required",
        )
    return user


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(security)
    ],
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthUser:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await auth_service.get_user_from_access_token(credentials.credentials)


def require_moderator(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if not (user.is_staff or user.is_admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Moderator or admin access required",
        )
    return user


def require_admin(user: AuthUser = Depends(get_current_user)) -> AuthUser:
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_generator(request: Request) -> Generator:
    return request.app.state.generator


def get_qa_service(
    retriever: Retriever = Depends(get_retriever),
    generator: Generator = Depends(get_generator),
    game_repo: GameRepository = Depends(get_game_repository),
) -> QAService:
    return QAService(retriever=retriever, generator=generator, game_repo=game_repo)
