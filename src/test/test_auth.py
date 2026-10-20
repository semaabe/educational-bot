"""Authentication and role-enforcement tests for the protected API routes.

The routes are rebuilt here on a bare FastAPI app rather than importing
backend.app, so these tests need no ChromaDB, Postgres or OpenAI key.
"""

import time

import jwt
import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from backend import auth
from backend.middleware.role_middleware import RoleMiddleware

# Tokens are signed with the API's own per-process key, as the bot's are.
SECRET = auth._SIGNING_KEY
AUDIENCE = auth.AUDIENCE
ISSUER = auth.ISSUER


@pytest.fixture(autouse=True)
def fresh_service_token():
    auth.reset_service_token_cache()
    yield
    auth.reset_service_token_cache()


@pytest.fixture
def client():
    app = FastAPI()

    @app.get("/query", dependencies=[Depends(auth.require_roles(*auth.QUERY_ROLES))])
    async def query():
        return {"ok": True}

    @app.get("/content", dependencies=[Depends(auth.require_roles(*auth.CONTENT_ROLES))])
    async def content():
        return {"ok": True}

    @app.get("/admin", dependencies=[Depends(auth.require_roles(*auth.ADMIN_ROLES))])
    async def admin():
        return {"ok": True}

    @app.get("/whoami")
    async def whoami(user: auth.AuthenticatedUser = Depends(auth.require_roles(*auth.QUERY_ROLES))):
        return {"subject": user.subject, "role": user.role}

    return TestClient(app)


@pytest.fixture
def gated_client():
    """An app behind RoleMiddleware, with routes that declare no role policy."""
    app = FastAPI()
    app.add_middleware(RoleMiddleware)
    reached = []

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/unguarded")
    async def unguarded(request: Request):
        reached.append(request.state.user_role)
        return {"role": request.state.user_role}

    test_client = TestClient(app)
    test_client.reached = reached
    return test_client


def make_token(
    role="student",
    *,
    secret=SECRET,
    audience=AUDIENCE,
    issuer=ISSUER,
    expires_in=3600,
    subject="user-1",
    omit=(),
):
    now = int(time.time())
    payload = {
        "sub": subject,
        "role": role,
        "aud": audience,
        "iss": issuer,
        "iat": now,
        "exp": now + expires_in,
    }
    for claim in omit:
        payload.pop(claim, None)
    return jwt.encode(payload, secret, algorithm="HS256")


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


# --- missing credentials ----------------------------------------------------

def test_missing_credentials_returns_401(client):
    response = client.get("/query")
    assert response.status_code == 401
    assert response.json()["detail"] == "Missing authentication credentials"


def test_missing_credentials_sets_www_authenticate_header(client):
    response = client.get("/query")
    assert response.headers["www-authenticate"] == "Bearer"


def test_no_student_fallback_for_anonymous_callers(client):
    """The old middleware answered 200 here by assuming the student role."""
    response = client.get("/whoami")
    assert response.status_code == 401
    assert "student" not in response.text


@pytest.mark.parametrize("header", ["", "   "])
def test_blank_authorization_header_returns_401(client, header):
    response = client.get("/query", headers={"Authorization": header})
    assert response.status_code == 401


# --- malformed schemes ------------------------------------------------------

@pytest.mark.parametrize(
    "header",
    [
        "Basic dXNlcjpwYXNz",                      # wrong scheme
        "Token abc.def.ghi",                       # wrong scheme
        "bearer",                                  # scheme with no token
        "Bearer",                                  # scheme with no token
        "abc.def.ghi",                             # bare token, no scheme
        "Bearer abc.def.ghi extra",                # too many segments
        "Bearer  ",                                # scheme with blank token
    ],
)
def test_malformed_authorization_scheme_returns_401(client, header):
    response = client.get("/query", headers={"Authorization": header})
    assert response.status_code == 401


def test_bearer_scheme_is_case_insensitive(client):
    response = client.get("/query", headers={"Authorization": f"bearer {make_token()}"})
    assert response.status_code == 200


# --- token validation -------------------------------------------------------

def test_expired_token_returns_401(client):
    response = client.get("/query", headers=bearer(make_token(expires_in=-10)))
    assert response.status_code == 401
    assert response.json()["detail"] == "Token has expired"


def test_invalid_signature_returns_401(client):
    forged = make_token(secret="not-the-real-secret")
    response = client.get("/query", headers=bearer(forged))
    assert response.status_code == 401
    assert response.json()["detail"] == "Token signature is invalid"


def test_unsigned_token_returns_401(client):
    """An alg=none token must not be accepted."""
    payload = {
        "sub": "user-1",
        "role": "professor",
        "aud": AUDIENCE,
        "iss": ISSUER,
        "exp": int(time.time()) + 3600,
    }
    unsigned = jwt.encode(payload, key=None, algorithm="none")
    response = client.get("/query", headers=bearer(unsigned))
    assert response.status_code == 401


def test_mismatched_audience_returns_401(client):
    response = client.get("/query", headers=bearer(make_token(audience="some-other-api")))
    assert response.status_code == 401
    assert response.json()["detail"] == "Token audience is not accepted"


def test_mismatched_issuer_returns_401(client):
    response = client.get("/query", headers=bearer(make_token(issuer="https://evil.test")))
    assert response.status_code == 401
    assert response.json()["detail"] == "Token issuer is not accepted"


def test_garbage_token_returns_401(client):
    response = client.get("/query", headers=bearer("not-a-jwt"))
    assert response.status_code == 401


@pytest.mark.parametrize("claim", ["exp", "aud", "iss"])
def test_token_missing_required_claim_returns_401(client, claim):
    response = client.get("/query", headers=bearer(make_token(omit=[claim])))
    assert response.status_code == 401


# --- role enforcement -------------------------------------------------------

def test_token_without_role_returns_403(client):
    response = client.get("/query", headers=bearer(make_token(omit=["role"])))
    assert response.status_code == 403
    assert response.json()["detail"] == "Token does not contain a role claim"


def test_null_role_returns_403(client):
    response = client.get("/query", headers=bearer(make_token(role=None)))
    assert response.status_code == 403


def test_non_string_role_returns_403(client):
    response = client.get("/query", headers=bearer(make_token(role=42)))
    assert response.status_code == 403


def test_unsupported_role_returns_403(client):
    response = client.get("/query", headers=bearer(make_token(role="superuser")))
    assert response.status_code == 403
    assert "Unsupported role" in response.json()["detail"]


def test_known_role_without_route_permission_returns_403(client):
    """student is a real role, but not one /admin accepts."""
    response = client.get("/admin", headers=bearer(make_token(role="student")))
    assert response.status_code == 403
    assert "may not access" in response.json()["detail"]


def test_student_cannot_reach_content_routes(client):
    response = client.get("/content", headers=bearer(make_token(role="student")))
    assert response.status_code == 403


# --- valid tokens -----------------------------------------------------------

def test_valid_student_token_is_accepted(client):
    response = client.get("/query", headers=bearer(make_token(role="student")))
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_valid_professor_token_is_accepted(client):
    token = make_token(role="professor")
    assert client.get("/query", headers=bearer(token)).status_code == 200
    assert client.get("/content", headers=bearer(token)).status_code == 200
    assert client.get("/admin", headers=bearer(token)).status_code == 200


@pytest.mark.parametrize("role", ["admin", "professor", "ta", "student"])
def test_every_allowed_role_can_query(client, role):
    response = client.get("/query", headers=bearer(make_token(role=role)))
    assert response.status_code == 200


def test_identity_is_available_to_handlers(client):
    token = make_token(role="professor", subject="discord-42")
    response = client.get("/whoami", headers=bearer(token))
    assert response.status_code == 200
    assert response.json() == {"subject": "discord-42", "role": "professor"}


# --- service token used by the Discord bot ----------------------------------

def test_service_token_is_accepted_by_protected_routes(client):
    headers = auth.auth_headers()
    assert headers["Authorization"].startswith("Bearer ")
    assert client.get("/query", headers=headers).status_code == 200
    assert client.get("/content", headers=headers).status_code == 200


def test_service_token_carries_service_role():
    user = auth.authenticate(auth.auth_headers()["Authorization"])
    assert user.role == "professor"
    assert user.subject == "discord-bot"


def test_service_token_is_cached_between_calls():
    assert auth.service_token() == auth.service_token()


def test_service_token_refreshes_once_expired(monkeypatch):
    first = auth.service_token()
    # Captured before patching, or the lambda would call itself.
    later = time.time() + 3600
    monkeypatch.setattr(auth.time, "time", lambda: later)
    assert auth.service_token() != first


def test_token_from_another_process_is_rejected(client):
    """Each process generates its own key, so a token signed elsewhere fails."""
    other_process_key = b"x" * 32
    response = client.get("/query", headers=bearer(make_token(secret=other_process_key)))
    assert response.status_code == 401


def test_service_role_can_reach_admin_routes(client):
    """The bot registers users and sets up routes."""
    assert client.get("/admin", headers=auth.auth_headers()).status_code == 200


def test_mint_token_rejects_unknown_role():
    with pytest.raises(ValueError):
        auth.mint_token("someone", "superuser")


# --- global middleware gate -------------------------------------------------
# These routes have no require_roles dependency, so anything rejected here was
# rejected by RoleMiddleware before the request was forwarded.

@pytest.mark.parametrize(
    "headers, status_code",
    [
        ({}, 401),                                                   # missing credentials
        ({"Authorization": "Basic dXNlcjpwYXNz"}, 401),              # wrong scheme
        ({"Authorization": "abc.def.ghi"}, 401),                     # no scheme
        (bearer(make_token(expires_in=-10)), 401),                   # expired
        (bearer(make_token(secret="not-the-real-secret")), 401),     # bad signature
        (bearer(make_token(audience="some-other-api")), 401),        # wrong audience
        (bearer(make_token(issuer="https://evil.test")), 401),       # wrong issuer
        (bearer(make_token(omit=["role"])), 403),                    # no role
        (bearer(make_token(role="superuser")), 403),                 # unsupported role
    ],
)
def test_middleware_rejects_before_forwarding(gated_client, headers, status_code):
    response = gated_client.get("/unguarded", headers=headers)
    assert response.status_code == status_code
    assert gated_client.reached == []


def test_middleware_missing_credentials_sets_www_authenticate_header(gated_client):
    response = gated_client.get("/unguarded")
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"] == "Missing authentication credentials"


def test_middleware_has_no_student_fallback(gated_client):
    """PR #2's middleware forwarded anonymous requests as role 'student'."""
    response = gated_client.get("/unguarded")
    assert response.status_code == 401
    assert "student" not in gated_client.reached


@pytest.mark.parametrize("role", ["student", "professor"])
def test_middleware_forwards_valid_token_with_its_role(gated_client, role):
    response = gated_client.get("/unguarded", headers=bearer(make_token(role=role)))
    assert response.status_code == 200
    assert response.json() == {"role": role}
    assert gated_client.reached == [role]


def test_middleware_leaves_public_paths_open(gated_client):
    assert gated_client.get("/health").status_code == 200


def test_middleware_rejects_unknown_paths_without_credentials(gated_client):
    """Unauthenticated callers cannot probe which routes exist."""
    assert gated_client.get("/does-not-exist").status_code == 401


# --- route policy guards ----------------------------------------------------

def test_require_roles_rejects_unknown_role():
    with pytest.raises(ValueError):
        auth.require_roles("superuser")


def test_require_roles_requires_at_least_one_role():
    with pytest.raises(ValueError):
        auth.require_roles()
