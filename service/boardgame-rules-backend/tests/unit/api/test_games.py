from unittest.mock import AsyncMock

import pytest

from boardgame_rules_backend.dependencies import (get_game_service, require_admin,
                                                  require_bot_or_moderator, require_moderator)
from boardgame_rules_backend.exceptions import DuplicateGameTitle, EmptyFileError, GameNotFound

pytestmark = pytest.mark.unit


def _override_auth_dependencies(fastapi_app, auth_user_factory):
    moderator = auth_user_factory(user_id=2, username="mod", is_admin=False, is_staff=True)
    admin = auth_user_factory(user_id=1, username="admin", is_admin=True, is_staff=True)
    fastapi_app.dependency_overrides[require_moderator] = lambda: moderator
    fastapi_app.dependency_overrides[require_bot_or_moderator] = lambda: moderator
    fastapi_app.dependency_overrides[require_admin] = lambda: admin


@pytest.mark.asyncio
async def test_list_games_success(api_client, fastapi_app, auth_user_factory):
    _override_auth_dependencies(fastapi_app, auth_user_factory)
    game_service = AsyncMock()
    game_service.list_games = AsyncMock(
        return_value=[
            {
                "id": 1,
                "title": "Catan",
                "source_doc_url": None,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        ]
    )
    fastapi_app.dependency_overrides[get_game_service] = lambda: game_service

    response = await api_client.get("/api/v1/games?skip=0&limit=50&search=cat")

    assert response.status_code == 200
    assert response.json()[0]["title"] == "Catan"
    game_service.list_games.assert_awaited_once_with(skip=0, limit=50, search="cat")


@pytest.mark.asyncio
async def test_create_game_duplicate_title_returns_409(api_client, fastapi_app, auth_user_factory):
    _override_auth_dependencies(fastapi_app, auth_user_factory)
    game_service = AsyncMock()
    game_service.create_game = AsyncMock(side_effect=DuplicateGameTitle())
    fastapi_app.dependency_overrides[get_game_service] = lambda: game_service

    response = await api_client.post(
        "/api/v1/games",
        json={"title": "Catan", "source_doc_url": None},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "A game with this title already exists"


@pytest.mark.asyncio
async def test_get_game_not_found_returns_404(api_client, fastapi_app, auth_user_factory):
    _override_auth_dependencies(fastapi_app, auth_user_factory)
    game_service = AsyncMock()
    game_service.get_game = AsyncMock(side_effect=GameNotFound())
    fastapi_app.dependency_overrides[get_game_service] = lambda: game_service

    response = await api_client.get("/api/v1/games/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Game not found"


@pytest.mark.asyncio
async def test_create_game_with_rules_empty_file_returns_400(
    api_client, fastapi_app, auth_user_factory
):
    _override_auth_dependencies(fastapi_app, auth_user_factory)
    game_service = AsyncMock()
    game_service.create_game_with_rules = AsyncMock(side_effect=EmptyFileError())
    fastapi_app.dependency_overrides[get_game_service] = lambda: game_service

    response = await api_client.post(
        "/api/v1/games/with-rules",
        data={"title": "Catan", "lang": "ru"},
        files={"file": ("rules.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Empty file"


@pytest.mark.asyncio
async def test_initialize_requires_csv_manifest(api_client, fastapi_app, auth_user_factory):
    _override_auth_dependencies(fastapi_app, auth_user_factory)
    game_service = AsyncMock()
    fastapi_app.dependency_overrides[get_game_service] = lambda: game_service

    response = await api_client.post(
        "/api/v1/games/initialize",
        files={
            "manifest": ("manifest.txt", b"bad", "text/plain"),
            "archive": ("archive.zip", b"PK\x03\x04", "application/zip"),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "manifest must be a CSV file"
    assert game_service.initialize_from_manifest.await_count == 0
