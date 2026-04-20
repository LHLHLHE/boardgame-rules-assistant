from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from boardgame_rules_backend.models import User

USER_MUTABLE_FIELDS = frozenset({"password", "email", "is_admin", "is_staff"})


class UserRepository:
    def __init__(self, pg_db_session: AsyncSession):
        self.pg_db_session = pg_db_session

    async def get_user_by_id(self, user_id: int) -> User | None:
        result = await self.pg_db_session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_user_by_username(self, username: str) -> User | None:
        result = await self.pg_db_session.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_users_list(self, skip: int = 0, limit: int = 100) -> list[User]:
        result = await self.pg_db_session.execute(
            select(User).order_by(User.username).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def count_admins(self) -> int:
        result = await self.pg_db_session.execute(
            select(func.count()).select_from(User).where(User.is_admin.is_(True))
        )
        return int(result.scalar_one() or 0)

    async def create_user(
        self,
        username: str,
        password_hash: str,
        email: str | None = None,
        is_admin: bool = False,
        is_staff: bool = False,
    ) -> User:
        user = User(
            username=username,
            password=password_hash,
            email=email,
            is_admin=is_admin,
            is_staff=is_staff,
        )
        self.pg_db_session.add(user)
        await self.pg_db_session.flush()
        await self.pg_db_session.refresh(user)
        await self.pg_db_session.commit()
        return user

    async def update_user_by_id(
        self,
        user_id: int,
        updates: dict[str, Any],
    ) -> User | None:
        user = await self.get_user_by_id(user_id)
        if user is None:
            return None

        for key, value in updates.items():
            if key not in USER_MUTABLE_FIELDS:
                msg = f"unsupported user field: {key!r}"
                raise ValueError(msg)
            setattr(user, key, value)

        await self.pg_db_session.flush()
        await self.pg_db_session.refresh(user)
        await self.pg_db_session.commit()
        return user

    async def delete_user(self, user: User) -> None:
        await self.pg_db_session.execute(delete(User).where(User.id == user.id))
        await self.pg_db_session.commit()
