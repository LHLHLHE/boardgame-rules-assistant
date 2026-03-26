import datetime as dt
import os
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from boardgame_rules_backend.main import app
from boardgame_rules_backend.schemas import AuthUser


@pytest.fixture
def fastapi_app():
    """Shared app fixture with guaranteed cleanup of dependency overrides."""
    app.dependency_overrides.clear()
    try:
        yield app
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
async def api_client(request, fastapi_app) -> AsyncGenerator[AsyncClient, None]:
    """
    Provide API-client.

    Unit tests run in-process via ASGITransport.
    Integration tests call a real backend container over HTTP.
    """
    if request.node.get_closest_marker("integration"):
        if os.getenv("RUN_INTEGRATION_TESTS", "0") != "1":
            pytest.skip("Set RUN_INTEGRATION_TESTS=1 to run integration tests.")
        base_url = os.getenv("INTEGRATION_BASE_URL", "http://localhost:8001")
        async with AsyncClient(base_url=base_url) as client:
            yield client
        return

    transport = ASGITransport(app=fastapi_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.fixture
def auth_user_factory():
    def _make_auth_user(
        user_id: int = 1,
        username: str = "admin",
        *,
        is_admin: bool = True,
        is_staff: bool = True,
    ) -> AuthUser:
        now = dt.datetime.now(dt.UTC)
        return AuthUser(
            id=user_id,
            username=username,
            email=f"{username}@example.com",
            is_admin=is_admin,
            is_staff=is_staff,
            created_at=now,
            updated_at=now,
        )

    return _make_auth_user
