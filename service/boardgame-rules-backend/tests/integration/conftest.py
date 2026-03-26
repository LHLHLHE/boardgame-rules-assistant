import os
import time
import uuid

import pytest

from boardgame_rules_backend.auth.utils import hash_password_bcrypt
from boardgame_rules_backend.database.postgres import PGAsyncSessionFactory
from boardgame_rules_backend.rag import Indexer
from boardgame_rules_backend.repository import UserRepository


@pytest.fixture(autouse=True)
def integration_guard():
    if os.getenv("RUN_INTEGRATION_TESTS", "0") != "1":
        pytest.skip("Set RUN_INTEGRATION_TESTS=1 to run integration tests.")


@pytest.fixture
async def backend_is_alive(api_client):
    response = await api_client.get("/api/health")
    assert response.status_code == 200, response.text


@pytest.fixture
async def integration_admin_credentials():
    username = os.getenv("INTEGRATION_ADMIN_USERNAME", "integration_admin")
    password = os.getenv("INTEGRATION_ADMIN_PASSWORD", "integration_admin_password")

    async with PGAsyncSessionFactory() as session:
        user_repo = UserRepository(session)
        existing = await user_repo.get_user_by_username(username)
        if existing is None:
            await user_repo.create_user(
                username=username,
                password_hash=hash_password_bcrypt(password),
                is_admin=True,
                is_staff=True,
            )

    return {"username": username, "password": password}


@pytest.fixture
async def admin_auth_headers(api_client, integration_admin_credentials):
    # Fail fast with clear diagnostics if auth is broken.
    login_response = await api_client.post(
        "/api/v1/auth/login",
        json=integration_admin_credentials,
    )
    assert login_response.status_code == 200, login_response.text

    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def bot_headers():
    token = os.getenv("INTEGRATION_BOT_TOKEN", "integration-bot-token")
    return {"X-Bot-Token": token}


@pytest.fixture
async def staff_auth_headers(api_client, admin_auth_headers):
    suffix = str(int(time.time() * 1000))
    username = f"staff_{suffix}"
    password = "integration_staff_password"

    create_response = await api_client.post(
        "/api/v1/users",
        headers=admin_auth_headers,
        json={
            "username": username,
            "password": password,
            "is_staff": True,
            "is_admin": False,
        },
    )
    assert create_response.status_code == 201, create_response.text

    login_response = await api_client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert login_response.status_code == 200, login_response.text
    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def indexed_game(api_client, admin_auth_headers):
    game_title = f"Indexed Integration Game {uuid.uuid4().hex[:8]}"
    create_response = await api_client.post(
        "/api/v1/games",
        headers=admin_auth_headers,
        json={"title": game_title},
    )
    assert create_response.status_code == 200, create_response.text
    game_id = create_response.json()["id"]

    indexer = Indexer()
    rules_document_id = int(time.time() * 1000)
    chunks = indexer.chunk_text(
        text=(
            "В этой игре побеждает игрок, набравший 10 очков победы. "
            "Очки даются за контроль территорий и выполнение задач."
        ),
        doc_id=uuid.uuid4().hex,
        rules_document_id=rules_document_id,
        game_id=game_id,
        lang="ru",
    )
    indexer.embed_and_upsert(chunks)
    return {"game_id": game_id, "title": game_title}
