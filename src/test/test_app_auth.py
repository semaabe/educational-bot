"""Authentication wiring of the real backend.app.

test_auth.py covers the auth rules on a bare app. This module checks that
backend.app actually applies them: every non-public route is closed to
anonymous callers, and /query takes the caller's role from the token instead
of defaulting to 'student'.

Importing backend.app loads the sentence encoder, so this module is slower than
test_auth.py. Startup events are not run, and the router is mocked.
"""

import re
import time
from unittest.mock import AsyncMock

import jwt
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from backend import app as app_module
from backend import auth
from backend.middleware.role_middleware import PUBLIC_PATHS

QUERY_BODY = {"query": "When is the midterm?", "guild_id": 1, "channel_id": 2}


@pytest.fixture
def process_query(monkeypatch):
    mock = AsyncMock(return_value={"answer": "Friday"})
    monkeypatch.setattr(app_module.semantic_router, "process_query", mock)
    return mock


@pytest.fixture
def client():
    # Not used as a context manager, so startup events do not run.
    return TestClient(app_module.app)


def make_token(role="student", **overrides):
    now = int(time.time())
    payload = {"sub": "user-1", "role": role, "aud": auth.AUDIENCE, "iss": auth.ISSUER,
               "iat": now, "exp": now + 3600}
    payload.update(overrides)
    payload = {k: v for k, v in payload.items() if v is not None}
    return jwt.encode(payload, auth._SIGNING_KEY, algorithm=auth.ALGORITHM)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _protected_routes():
    for route in app_module.app.routes:
        if isinstance(route, APIRoute) and route.path not in PUBLIC_PATHS:
            for method in sorted(route.methods):
                yield method, re.sub(r"\{[^}]+\}", "x", route.path)


@pytest.mark.parametrize("method, path", list(_protected_routes()))
def test_every_non_public_route_requires_credentials(client, method, path):
    response = client.request(method, path)
    assert response.status_code == 401, f"{method} {path} is reachable anonymously"


def test_query_route_is_registered():
    assert ("POST", "/query") in set(_protected_routes())


def test_query_without_credentials_returns_401(client, process_query):
    response = client.post("/query", json=QUERY_BODY)
    assert response.status_code == 401
    process_query.assert_not_called()


def test_query_with_malformed_scheme_returns_401(client, process_query):
    response = client.post("/query", json=QUERY_BODY,
                           headers={"Authorization": f"Token {make_token()}"})
    assert response.status_code == 401
    process_query.assert_not_called()


def test_query_with_expired_token_returns_401(client, process_query):
    response = client.post("/query", json=QUERY_BODY,
                           headers=bearer(make_token(exp=int(time.time()) - 10)))
    assert response.status_code == 401
    process_query.assert_not_called()


@pytest.mark.parametrize("role, status_code", [(None, 403), ("superuser", 403)])
def test_query_without_an_allowed_role_returns_403(client, process_query, role, status_code):
    response = client.post("/query", json=QUERY_BODY, headers=bearer(make_token(role=role)))
    assert response.status_code == status_code
    process_query.assert_not_called()


@pytest.mark.parametrize("role", ["student", "professor"])
def test_query_passes_the_token_role_to_the_router(client, process_query, role):
    response = client.post("/query", json=QUERY_BODY, headers=bearer(make_token(role=role)))
    assert response.status_code == 200
    assert response.json() == {"answer": "Friday"}
    assert process_query.await_args.kwargs["role"] == role


def test_bot_service_token_reaches_bot_routes(client, process_query):
    headers = auth.auth_headers()
    assert client.post("/query", json=QUERY_BODY, headers=headers).status_code == 200
    # Reaches the handler (a 403/401 would mean auth rejected it).
    assert client.post("/setup_routes", json=[], headers=headers).status_code not in (401, 403)


def test_student_token_cannot_reach_admin_routes(client):
    response = client.post("/setup_routes", json={}, headers=bearer(make_token(role="student")))
    assert response.status_code == 403
