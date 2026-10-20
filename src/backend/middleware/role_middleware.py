"""Global authentication gate for the FastAPI backend.

Every request outside ``PUBLIC_PATHS`` must carry a valid ``Bearer`` JWT whose
``role`` is one of ``auth.ALLOWED_ROLES`` before it is forwarded. There is no
anonymous fallback role.

This is the default-deny layer, so a route added without a ``require_roles``
dependency is still closed to unauthenticated callers. Which roles may use a
particular route is decided by that route's ``require_roles`` dependency.
"""

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.auth import authenticate

# Reachable without credentials. Swagger UI needs its own pages to offer the
# "Authorize" button; every operation it calls is still protected.
PUBLIC_PATHS = frozenset({
    "/health",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
})


class RoleMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, public_paths=PUBLIC_PATHS):
        super().__init__(app)
        self.public_paths = frozenset(public_paths)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.public_paths:
            return await call_next(request)

        # Exceptions raised in BaseHTTPMiddleware bypass FastAPI's handlers,
        # so the auth error is turned into a response here.
        try:
            user = authenticate(request.headers.get("Authorization"))
        except HTTPException as exc:
            return JSONResponse(
                {"detail": exc.detail},
                status_code=exc.status_code,
                headers=exc.headers,
            )

        request.state.user = user
        request.state.user_role = user.role
        return await call_next(request)
