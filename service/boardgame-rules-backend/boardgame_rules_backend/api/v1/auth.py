from fastapi import APIRouter, Depends

from boardgame_rules_backend.dependencies import get_auth_service, get_current_user
from boardgame_rules_backend.schemas import AuthUser, LoginRequest, MeResponse, TokenResponse
from boardgame_rules_backend.service import AuthService

router = APIRouter()


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Вход",
    description="Выдаёт JWT при успешной аутентификации (только staff или admin).",
)
async def login(
    payload: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
):
    token = await auth_service.login(payload.username, payload.password)
    return TokenResponse(access_token=token)


@router.get(
    "/me",
    response_model=MeResponse,
    summary="Текущий пользователь",
)
async def me(current: AuthUser = Depends(get_current_user)):
    return MeResponse.model_validate(current)
