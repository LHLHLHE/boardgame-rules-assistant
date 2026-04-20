from boardgame_rules_backend.models import Game, RulesDocument, User
from boardgame_rules_backend.schemas.auth import AuthUser
from boardgame_rules_backend.schemas.games import GameRead, RulesDocumentRead
from boardgame_rules_backend.schemas.users import UserRead


def to_auth_user(user: User) -> AuthUser:
    return AuthUser.model_validate(user)


def to_user_read(user: User) -> UserRead:
    return UserRead.model_validate(user)


def to_game_read(game: Game) -> GameRead:
    return GameRead.model_validate(game)


def to_rules_document_read(doc: RulesDocument) -> RulesDocumentRead:
    return RulesDocumentRead.model_validate(doc)
