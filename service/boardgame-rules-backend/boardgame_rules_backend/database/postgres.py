from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, declared_attr, sessionmaker

from boardgame_rules_backend.settings import app_config

pg_engine = create_async_engine(
    app_config.pg_db_url,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
)
PGAsyncSessionFactory = async_sessionmaker(pg_engine, autoflush=False, expire_on_commit=False)

pg_sync_engine = create_engine(
    app_config.pg_sync_db_url,
    pool_size=3,
    max_overflow=5,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
)
PGSyncSessionFactory = sessionmaker(pg_sync_engine, autoflush=False, expire_on_commit=False)


@contextmanager
def get_sync_pg_db_session() -> Generator[Session, None, None]:
    with PGSyncSessionFactory() as session:
        yield session


async def get_pg_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with PGAsyncSessionFactory() as session:
        yield session


class PGBase(DeclarativeBase):
    id: Any

    __allow_unmapped__ = True

    @declared_attr
    def __tablename__(self) -> str:
        return self.__name__.lower()
