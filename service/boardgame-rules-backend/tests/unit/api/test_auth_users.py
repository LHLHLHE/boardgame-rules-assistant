import datetime as dt
from unittest.mock import AsyncMock

import pytest

from boardgame_rules_backend.dependencies import get_auth_service, get_current_user, require_admin
from boardgame_rules_backend.exceptions import AuthInvalidCredentials, UsernameAlreadyExists
from boardgame_rules_backend.schemas import AuthUser

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_login_success(api_client, fastapi_app):
    auth_service = AsyncMock()
    auth_service.login = AsyncMock(return_value="jwt-token")
    fastapi_app.dependency_overrides[get_auth_service] = lambda: auth_service

    response = await api_client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "secret123"},
    )

    assert response.status_code == 200
    assert response.json()["access_token"] == "jwt-token"
    assert response.json()["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_invalid_credentials(api_client, fastapi_app):
    auth_service = AsyncMock()
    auth_service.login = AsyncMock(side_effect=AuthInvalidCredentials())
    fastapi_app.dependency_overrides[get_auth_service] = lambda: auth_service

    response = await api_client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "wrong"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid username or password"


@pytest.mark.asyncio
async def test_me_returns_current_user(api_client, fastapi_app, auth_user_factory):
    current_user = auth_user_factory(user_id=7, username="mod", is_admin=False, is_staff=True)
    fastapi_app.dependency_overrides[get_current_user] = lambda: current_user

    response = await api_client.get("/api/v1/auth/me")

    assert response.status_code == 200
    assert response.json()["id"] == 7
    assert response.json()["username"] == "mod"
    assert response.json()["is_staff"] is True


def _sample_user_payload(user_id: int, username: str) -> dict:
    now = dt.datetime.now(dt.UTC).isoformat()
    return {
        "id": user_id,
        "username": username,
        "email": f"{username}@example.com",
        "is_admin": False,
        "is_staff": True,
        "created_at": now,
        "updated_at": now,
    }


@pytest.mark.asyncio
async def test_list_users_success(api_client, fastapi_app):
    auth_service = AsyncMock()
    auth_service.list_users = AsyncMock(return_value=[_sample_user_payload(2, "mod")])
    fastapi_app.dependency_overrides[get_auth_service] = lambda: auth_service
    fastapi_app.dependency_overrides[require_admin] = lambda: AuthUser(
        id=1,
        username="admin",
        email=None,
        is_admin=True,
        is_staff=True,
        created_at=dt.datetime.now(dt.UTC),
        updated_at=dt.datetime.now(dt.UTC),
    )

    response = await api_client.get("/api/v1/users")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["username"] == "mod"


@pytest.mark.asyncio
async def test_create_user_conflict(api_client, fastapi_app):
    auth_service = AsyncMock()
    auth_service.create_user = AsyncMock(side_effect=UsernameAlreadyExists())
    fastapi_app.dependency_overrides[get_auth_service] = lambda: auth_service
    fastapi_app.dependency_overrides[require_admin] = lambda: AuthUser(
        id=1,
        username="admin",
        email=None,
        is_admin=True,
        is_staff=True,
        created_at=dt.datetime.now(dt.UTC),
        updated_at=dt.datetime.now(dt.UTC),
    )

    response = await api_client.post(
        "/api/v1/users",
        json={
            "username": "mod",
            "password": "secret123",
            "is_staff": True,
            "is_admin": False,
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Username already exists"
