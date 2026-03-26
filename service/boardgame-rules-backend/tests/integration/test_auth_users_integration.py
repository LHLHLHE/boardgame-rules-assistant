import time

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_auth_login_and_me(api_client, integration_admin_credentials, backend_is_alive):
    login_response = await api_client.post(
        "/api/v1/auth/login",
        json=integration_admin_credentials,
    )
    assert login_response.status_code == 200

    token = login_response.json()["access_token"]
    me_response = await api_client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_response.status_code == 200

    me_data = me_response.json()
    assert me_data["username"] == integration_admin_credentials["username"]
    assert me_data["is_admin"] is True


@pytest.mark.asyncio
async def test_auth_login_invalid_credentials_returns_401(
    api_client,
    integration_admin_credentials
):
    login_response = await api_client.post(
        "/api/v1/auth/login",
        json={
            "username": integration_admin_credentials["username"],
            "password": "wrong-password",
        },
    )
    assert login_response.status_code == 401
    assert "Invalid username or password" in login_response.text


@pytest.mark.asyncio
async def test_users_crud(api_client, admin_auth_headers):
    suffix = str(int(time.time() * 1000))
    username = f"it_user_{suffix}"

    create_response = await api_client.post(
        "/api/v1/users",
        headers=admin_auth_headers,
        json={
            "username": username,
            "password": "integration_secret_123",
            "is_staff": True,
            "is_admin": False,
        },
    )
    assert create_response.status_code == 201

    list_response = await api_client.get("/api/v1/users", headers=admin_auth_headers)
    assert list_response.status_code == 200

    usernames = [item["username"] for item in list_response.json()]
    assert username in usernames

    user_id = create_response.json()["id"]
    patch_response = await api_client.patch(
        f"/api/v1/users/{user_id}",
        headers=admin_auth_headers,
        json={"is_staff": False},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["is_staff"] is False

    delete_response = await api_client.delete(
        f"/api/v1/users/{user_id}",
        headers=admin_auth_headers,
    )
    assert delete_response.status_code == 204


@pytest.mark.asyncio
async def test_users_requires_auth_returns_401(api_client):
    response = await api_client.get("/api/v1/users")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_users_admin_only_returns_403(api_client, staff_auth_headers):
    response = await api_client.get("/api/v1/users", headers=staff_auth_headers)
    assert response.status_code == 403
