"""
JWT Authentication Middleware.

Validates Bearer tokens on every protected request.
Public routes are whitelisted (health, docs, openapi).

Uses:
    • python-jose for JWT decode
    • FastAPI Depends() for per-route auth
"""

import time
import uuid
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# ── Config ────────────────────────────────────────────────────────────────────

SECRET_KEY = "CHANGE_ME_IN_PRODUCTION_USE_ENV_VAR"   # ⚠️ Use env var!
ALGORITHM  = "HS256"
TOKEN_EXPIRE_MINUTES = 60

# Routes that skip auth entirely
PUBLIC_ROUTES = {
    "/health",
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/api/v1/auth/token",
}

# ── Bearer Scheme ─────────────────────────────────────────────────────────────

bearer_scheme = HTTPBearer(auto_error=False)


# ── Token Utilities ───────────────────────────────────────────────────────────

def create_access_token(
    subject: str,
    roles: list[str] = None,
    expires_in: int = TOKEN_EXPIRE_MINUTES,
) -> str:
    """
    Create a signed JWT access token.

    Args:
        subject   : user/client identifier
        roles     : list of roles (e.g. ["admin", "user"])
        expires_in: token lifetime in minutes

    Returns:
        Signed JWT string
    """
    payload = {
        "sub"  : subject,
        "roles": roles or ["user"],
        "iat"  : int(time.time()),
        "exp"  : int(time.time()) + expires_in * 60,
        "jti"  : str(uuid.uuid4()),       # unique token ID
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT string

    Returns:
        Decoded payload dict

    Raises:
        HTTPException 401 if token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = f"Invalid or expired token: {str(e)}",
            headers     = {"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI Dependency ────────────────────────────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    """
    FastAPI dependency that validates JWT and returns current user.

    Usage:
        @router.post("/endpoint")
        async def endpoint(user = Depends(get_current_user)):
            ...
    """
    if credentials is None:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Authentication required",
            headers     = {"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)
    return {
        "user_id" : payload.get("sub"),
        "roles"   : payload.get("roles", []),
        "jti"     : payload.get("jti"),
    }


def require_role(role: str):
    """
    Dependency factory: require a specific role.

    Usage:
        @router.delete("/identity/{id}")
        async def delete(user = Depends(require_role("admin"))):
            ...
    """
    async def role_checker(
        user: dict = Depends(get_current_user)
    ) -> dict:
        if role not in user.get("roles", []):
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail      = f"Role '{role}' required",
            )
        return user
    return role_checker


# ── Middleware Class ──────────────────────────────────────────────────────────

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Global auth middleware.

    Skips public routes. For all other routes,
    attaches user info to request.state if valid token present.
    Does NOT block — blocking is done via Depends().
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip public routes
        if path in PUBLIC_ROUTES or path.startswith("/docs"):
            return await call_next(request)

        # Try to decode token (non-blocking — just attaches user)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            try:
                payload = decode_token(token)
                request.state.user    = payload.get("sub")
                request.state.roles   = payload.get("roles", [])
                request.state.token   = token
            except HTTPException:
                request.state.user  = None
                request.state.roles = []
        else:
            request.state.user  = None
            request.state.roles = []

        return await call_next(request)
