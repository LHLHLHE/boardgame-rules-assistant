from datetime import datetime, timedelta, timezone

from jose import JWTError

from boardgame_rules_backend.auth.jwt import decode_token, encode_token, user_id_from_payload
from boardgame_rules_backend.auth.utils import hash_password_bcrypt, verify_password_bcrypt
from boardgame_rules_backend.exceptions import (AdminUserNotFound, AuthInvalidCredentials,
                                                AuthPanelAccessDenied,
                                                InitialAdminAlreadyExistsError,
                                                InvalidOrExpiredTokenError, LastAdminRemovalError,
                                                TokenSubjectUserMissingError, UsernameAlreadyExists)
from boardgame_rules_backend.models import User
from boardgame_rules_backend.repository import UserRepository
from boardgame_rules_backend.schemas.auth import AuthUser
from boardgame_rules_backend.schemas.mappers import to_auth_user, to_user_read
from boardgame_rules_backend.schemas.users import UserCreate, UserRead, UserUpdate
from boardgame_rules_backend.settings import app_config


class AuthService:
    def __init__(self, user_repo: UserRepository):
        self._user_repo = user_repo

    @staticmethod
    def _user_role(user: User) -> str:
        if user.is_admin:
            return "admin"
        if user.is_staff:
            return "moderator"
        return "none"

    def _create_access_token(self, user: User) -> str:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=app_config.jwt_expire_minutes
        )
        payload = {
            "sub": str(user.id),
            "username": user.username,
            "role": self._user_role(user),
            "exp": int(expire.timestamp()),
        }
        return encode_token(payload)

    @staticmethod
    def _hash_password(plain: str) -> str:
        return hash_password_bcrypt(plain)

    @staticmethod
    def _verify_password(plain: str, hashed: str) -> bool:
        return verify_password_bcrypt(plain, hashed)

    async def get_user_from_access_token(self, token: str) -> AuthUser:
        try:
            payload = decode_token(token)
            user_id = user_id_from_payload(payload)
        except JWTError:
            raise InvalidOrExpiredTokenError() from None
        user = await self._user_repo.get_user_by_id(user_id)
        if user is None:
            raise TokenSubjectUserMissingError()
        return to_auth_user(user)

    async def login(self, username: str, password: str) -> str:
        user = await self._user_repo.get_user_by_username(username)
        if user is None or not self._verify_password(password, user.password):
            raise AuthInvalidCredentials()
        if not (user.is_staff or user.is_admin):
            raise AuthPanelAccessDenied()
        return self._create_access_token(user)

    async def list_users(self) -> list[UserRead]:
        users = await self._user_repo.get_users_list()
        return [to_user_read(user) for user in users]

    async def create_user(self, payload: UserCreate) -> UserRead:
        if await self._user_repo.get_user_by_username(payload.username):
            raise UsernameAlreadyExists()
        user = await self._user_repo.create_user(
            username=payload.username,
            password_hash=self._hash_password(payload.password),
            email=payload.email,
            is_admin=payload.is_admin,
            is_staff=payload.is_staff,
        )
        return to_user_read(user)

    async def update_user(self, user_id: int, payload: UserUpdate) -> UserRead:
        user = await self._user_repo.get_user_by_id(user_id)
        if user is None:
            raise AdminUserNotFound()

        data = payload.model_dump(exclude_unset=True)
        if data.get("is_admin") is False and user.is_admin:
            admins = await self._user_repo.count_admins()
            if admins <= 1:
                raise LastAdminRemovalError()

        updates: dict[str, object] = {}
        if "password" in data and data["password"] is not None:
            updates["password"] = self._hash_password(data["password"])
        if "email" in data:
            updates["email"] = data["email"]
        if "is_admin" in data:
            updates["is_admin"] = data["is_admin"]
        if "is_staff" in data:
            updates["is_staff"] = data["is_staff"]

        updated = await self._user_repo.update_user_by_id(user_id, updates)
        if updated is None:
            raise AdminUserNotFound()
        return to_user_read(updated)

    async def delete_user(self, user_id: int) -> None:
        user = await self._user_repo.get_user_by_id(user_id)
        if user is None:
            raise AdminUserNotFound()

        if user.is_admin:
            admins = await self._user_repo.count_admins()
            if admins <= 1:
                raise LastAdminRemovalError()

        await self._user_repo.delete_user(user)

    async def create_initial_admin(
        self,
        username: str,
        password: str,
        force: bool = False,
    ) -> UserRead:
        if await self._user_repo.count_admins() > 0 and not force:
            raise InitialAdminAlreadyExistsError()
        if await self._user_repo.get_user_by_username(username):
            raise UsernameAlreadyExists()
        user = await self._user_repo.create_user(
            username=username,
            password_hash=self._hash_password(password),
            is_admin=True,
            is_staff=True,
        )
        return to_user_read(user)
