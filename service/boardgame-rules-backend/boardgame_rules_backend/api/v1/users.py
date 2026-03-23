from fastapi import APIRouter, Depends

from boardgame_rules_backend.dependencies import get_auth_service, require_admin
from boardgame_rules_backend.schemas import AuthUser, UserCreate, UserRead, UserUpdate
from boardgame_rules_backend.service import AuthService

router = APIRouter()


@router.get("", response_model=list[UserRead])
async def list_users(
    _auth: AuthUser = Depends(require_admin),
    auth_service: AuthService = Depends(get_auth_service),
):
    return await auth_service.list_users()


@router.post("", response_model=UserRead, status_code=201)
async def create_user(
    payload: UserCreate,
    _auth: AuthUser = Depends(require_admin),
    auth_service: AuthService = Depends(get_auth_service),
):
    return await auth_service.create_user(payload)


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(
    user_id: int,
    payload: UserUpdate,
    _auth: AuthUser = Depends(require_admin),
    auth_service: AuthService = Depends(get_auth_service),
):
    return await auth_service.update_user(user_id, payload)


@router.delete("/{user_id}", status_code=204)
async def delete_user(
    user_id: int,
    _auth: AuthUser = Depends(require_admin),
    auth_service: AuthService = Depends(get_auth_service),
):
    await auth_service.delete_user(user_id)
