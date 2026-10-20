"""Bearer-token authentication and role enforcement for the FastAPI backend.

Two layers use this module:

* ``backend.middleware.role_middleware.RoleMiddleware`` calls ``authenticate``
  on every non-public request, so nothing is forwarded without a valid token
  carrying a known role.
* Each route declares the roles it accepts, e.g.

      @app.post('/resource_query', dependencies=[Depends(require_roles(*QUERY_ROLES))])

There is deliberately no unauthenticated fallback: a request without usable
credentials never reaches a handler.

Tokens are HS256 JWTs signed with a key generated when the process starts.
src/main.py runs the Discord bot and this API in one process, so the bot signs
its own service token with that key and no secret has to be configured or
stored. A token minted by any other process will not verify.

Status codes
------------
401  credentials missing, a scheme other than ``Bearer``, or a token that fails
     signature / expiry / audience / issuer validation.
403  a structurally valid token whose ``role`` claim is missing, unknown, or
     not permitted for the route.

Token contents are never logged.
"""

import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional

import jwt
from fastapi import HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# Per-process signing key; never leaves memory.
_SIGNING_KEY = secrets.token_bytes(32)
ALGORITHM = "HS256"
AUDIENCE = "fams-educational-bot-api"
ISSUER = "fams-educational-bot"

# Roles recognised by the API, mirroring the user_role enum in init-db.sql.
ALLOWED_ROLES = frozenset({"admin", "professor", "ta", "student"})

# Role groups used by the route declarations in backend/app.py.
QUERY_ROLES = ("student", "ta", "professor", "admin")
CONTENT_ROLES = ("ta", "professor", "admin")
ADMIN_ROLES = ("professor", "admin")

# The Discord bot's service token. It registers users and sets up routes, so it
# needs a role in ADMIN_ROLES.
SERVICE_SUBJECT = "discord-bot"
SERVICE_ROLE = "professor"
SERVICE_TOKEN_TTL_SECONDS = 3600

# Declared so Swagger UI renders an "Authorize" button. auto_error is off
# because this module distinguishes a missing header from a malformed one and
# answers both with 401 rather than Starlette's default 403.
bearer_scheme = HTTPBearer(auto_error=False, description="HS256 JWT issued for this API")

_UNAUTHENTICATED_HEADERS = {"WWW-Authenticate": "Bearer"}

# Cached service token: (token, expires_at). Refreshed shortly before expiry.
_service_token_cache: tuple = (None, 0.0)
_SERVICE_TOKEN_REFRESH_MARGIN = 60


@dataclass(frozen=True)
class AuthenticatedUser:
    """The validated identity behind a request."""

    subject: str
    role: str
    claims: dict = field(default_factory=dict, repr=False)


def _unauthenticated(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers=_UNAUTHENTICATED_HEADERS,
    )


def _forbidden(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


def extract_bearer_token(authorization: Optional[str]) -> str:
    """Pull the token out of an Authorization header, or raise 401."""
    if not authorization or not authorization.strip():
        raise _unauthenticated("Missing authentication credentials")

    parts = authorization.split()
    if len(parts) != 2:
        raise _unauthenticated("Malformed Authorization header; expected 'Bearer <token>'")

    scheme, token = parts
    if scheme.lower() != "bearer":
        raise _unauthenticated(f"Unsupported authentication scheme '{scheme}'; expected 'Bearer'")

    if not token.strip():
        raise _unauthenticated("Missing bearer token")

    return token


def decode_token(token: str) -> dict:
    """Validate signature, expiry, audience and issuer; return the claims."""
    try:
        return jwt.decode(
            token,
            _SIGNING_KEY,
            algorithms=[ALGORITHM],
            audience=AUDIENCE,
            issuer=ISSUER,
            options={"require": ["exp", "aud", "iss"]},
        )
    except jwt.ExpiredSignatureError:
        raise _unauthenticated("Token has expired")
    except jwt.InvalidAudienceError:
        raise _unauthenticated("Token audience is not accepted")
    except jwt.InvalidIssuerError:
        raise _unauthenticated("Token issuer is not accepted")
    except jwt.InvalidSignatureError:
        raise _unauthenticated("Token signature is invalid")
    except jwt.MissingRequiredClaimError as exc:
        # exc names the claim only, never a value.
        raise _unauthenticated(f"Token is missing required claim: {exc.claim}")
    except jwt.InvalidTokenError as exc:
        # Log the failure class, never the token itself.
        logger.warning("Rejected bearer token: %s", type(exc).__name__)
        raise _unauthenticated("Invalid token")


def authenticate(authorization: Optional[str]) -> AuthenticatedUser:
    """Authenticate a raw Authorization header value."""
    claims = decode_token(extract_bearer_token(authorization))

    role = claims.get("role")
    if not role or not isinstance(role, str):
        raise _forbidden("Token does not contain a role claim")

    if role not in ALLOWED_ROLES:
        raise _forbidden(f"Unsupported role '{role}'")

    subject = claims.get("sub") or "unknown"
    return AuthenticatedUser(subject=str(subject), role=role, claims=claims)


def require_roles(*allowed_roles: str):
    """Build a dependency that admits only the given roles."""
    if not allowed_roles:
        raise ValueError("require_roles needs at least one role")

    allowed = frozenset(allowed_roles)
    unknown = allowed - ALLOWED_ROLES
    if unknown:
        raise ValueError(f"Unknown roles in route policy: {sorted(unknown)}")

    async def dependency(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    ) -> AuthenticatedUser:
        user = authenticate(request.headers.get("Authorization"))

        if user.role not in allowed:
            logger.info(
                "Denied %s %s for role '%s'", request.method, request.url.path, user.role
            )
            raise _forbidden(f"Role '{user.role}' may not access this resource")

        # Available to handlers that need the caller's identity.
        request.state.user = user
        request.state.user_role = user.role
        return user

    return dependency


def mint_token(subject: str, role: str, ttl_seconds: int = SERVICE_TOKEN_TTL_SECONDS) -> str:
    """Issue a token for `subject` carrying `role`, valid in this process only."""
    if role not in ALLOWED_ROLES:
        raise ValueError(f"Cannot mint a token for unknown role '{role}'")

    now = int(time.time())
    payload = {
        "sub": subject,
        "role": role,
        "aud": AUDIENCE,
        "iss": ISSUER,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    return jwt.encode(payload, _SIGNING_KEY, algorithm=ALGORITHM)


def service_token() -> str:
    """The Discord bot's own token, cached until shortly before it expires."""
    global _service_token_cache

    token, expires_at = _service_token_cache
    if token and time.time() < expires_at - _SERVICE_TOKEN_REFRESH_MARGIN:
        return token

    token = mint_token(SERVICE_SUBJECT, SERVICE_ROLE, ttl_seconds=SERVICE_TOKEN_TTL_SECONDS)
    _service_token_cache = (token, time.time() + SERVICE_TOKEN_TTL_SECONDS)
    return token


def auth_headers() -> dict:
    """Authorization header for internal backend calls."""
    return {"Authorization": f"Bearer {service_token()}"}


def reset_service_token_cache() -> None:
    """Drop the cached service token (used by tests)."""
    global _service_token_cache
    _service_token_cache = (None, 0.0)
