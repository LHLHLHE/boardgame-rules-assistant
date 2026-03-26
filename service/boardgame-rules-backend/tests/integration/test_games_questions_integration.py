import time

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_games_crud_flow(api_client, admin_auth_headers):
    suffix = str(int(time.time() * 1000))
    title = f"Integration Game {suffix}"

    create_response = await api_client.post(
        "/api/v1/games",
        headers=admin_auth_headers,
        json={"title": title},
    )
    assert create_response.status_code == 200, create_response.text

    game = create_response.json()
    game_id = game["id"]
    assert game["title"] == title

    get_response = await api_client.get(f"/api/v1/games/{game_id}", headers=admin_auth_headers)
    assert get_response.status_code == 200
    assert get_response.json()["id"] == game_id

    update_response = await api_client.patch(
        f"/api/v1/games/{game_id}",
        headers=admin_auth_headers,
        json={"title": f"{title} Updated"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["title"].endswith("Updated")

    list_response = await api_client.get("/api/v1/games", headers=admin_auth_headers)
    assert list_response.status_code == 200

    ids = [item["id"] for item in list_response.json()]
    assert game_id in ids

    delete_response = await api_client.delete(
        f"/api/v1/games/{game_id}",
        headers=admin_auth_headers,
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_games_duplicate_title_returns_409(api_client, admin_auth_headers):
    suffix = str(int(time.time() * 1000))
    title = f"Duplicate Integration Game {suffix}"
    create_response = await api_client.post(
        "/api/v1/games",
        headers=admin_auth_headers,
        json={"title": title},
    )
    assert create_response.status_code == 200

    duplicate_response = await api_client.post(
        "/api/v1/games",
        headers=admin_auth_headers,
        json={"title": title},
    )
    assert duplicate_response.status_code == 409


@pytest.mark.asyncio
async def test_games_get_not_found_returns_404(api_client, admin_auth_headers):
    response = await api_client.get("/api/v1/games/999999", headers=admin_auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_games_create_requires_auth_returns_401(api_client):
    response = await api_client.post("/api/v1/games", json={"title": "No Auth Game"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_questions_ask_real_flow_returns_answer(api_client, indexed_game, bot_headers):
    ask_response = await api_client.post(
        "/api/v1/questions/ask",
        headers=bot_headers,
        json={"game_id": indexed_game["game_id"], "query": "Кто побеждает в этой игре?"},
    )
    assert ask_response.status_code == 200
    answer = ask_response.json()["answer"]
    assert answer.startswith("CONTEXT_OK"), answer


@pytest.mark.asyncio
async def test_questions_ask_invalid_bot_token_returns_401(api_client, indexed_game):
    response = await api_client.post(
        "/api/v1/questions/ask",
        headers={"X-Bot-Token": "invalid-token"},
        json={"game_id": indexed_game["game_id"], "query": "Q"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_questions_ask_game_not_found_returns_404(api_client, bot_headers):
    response = await api_client.post(
        "/api/v1/questions/ask",
        headers=bot_headers,
        json={"game_id": 999999, "query": "Q"},
    )
    assert response.status_code == 404
