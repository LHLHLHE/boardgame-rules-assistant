from unittest.mock import AsyncMock

import pytest

from boardgame_rules_backend.dependencies import get_qa_service
from boardgame_rules_backend.exceptions import GameNotFound
from boardgame_rules_backend.settings import app_config

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_ask_question_success(api_client, fastapi_app, monkeypatch):
    monkeypatch.setattr(app_config, "bot_api_token", "test-bot-token")
    qa_service = AsyncMock()
    qa_service.get_answer = AsyncMock(return_value="answer text")
    fastapi_app.dependency_overrides[get_qa_service] = lambda: qa_service

    response = await api_client.post(
        "/api/v1/questions/ask",
        json={"game_id": 10, "query": "Как победить?", "history": "Контекст"},
        headers={"X-Bot-Token": "test-bot-token"},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "answer text"}
    qa_service.get_answer.assert_awaited_once_with(10, "Как победить?", "Контекст")


@pytest.mark.asyncio
async def test_ask_question_invalid_bot_token_returns_401(api_client, monkeypatch):
    monkeypatch.setattr(app_config, "bot_api_token", "test-bot-token")

    response = await api_client.post(
        "/api/v1/questions/ask",
        json={"game_id": 10, "query": "Q"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing X-Bot-Token"


@pytest.mark.asyncio
async def test_ask_question_game_not_found_returns_404(api_client, fastapi_app, monkeypatch):
    monkeypatch.setattr(app_config, "bot_api_token", "test-bot-token")
    qa_service = AsyncMock()
    qa_service.get_answer = AsyncMock(side_effect=GameNotFound())
    fastapi_app.dependency_overrides[get_qa_service] = lambda: qa_service

    response = await api_client.post(
        "/api/v1/questions/ask",
        json={"game_id": 999, "query": "Q"},
        headers={"X-Bot-Token": "test-bot-token"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Game not found"
