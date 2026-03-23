import datetime as dt
import enum

from sqlalchemy import DateTime, Enum, ForeignKey, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from boardgame_rules_backend.database.postgres import PGBase


class RulesDocumentStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    indexed = "indexed"
    failed = "failed"


class Game(PGBase):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(100))
    source_doc_url: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    __table_args__ = (
        Index(
            "ix_games_title_lower",
            func.lower(title),
            unique=True,
        ),
    )

    rules_documents: Mapped[list["RulesDocument"]] = relationship(
        "RulesDocument",
        back_populates="game"
    )


class RulesDocument(PGBase):
    __tablename__ = "rules_documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id", ondelete="CASCADE"))
    doc_id: Mapped[str] = mapped_column(String(64))
    storage_path: Mapped[str] = mapped_column(String(1024))
    lang: Mapped[str] = mapped_column(String(16), default="ru")
    status: Mapped[RulesDocumentStatus] = mapped_column(
        Enum(RulesDocumentStatus, name="rules_document_status"),
        default=RulesDocumentStatus.pending,
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )

    game: Mapped["Game"] = relationship("Game", back_populates="rules_documents")
