"""
Rate Limiting Middleware.

Implements a sliding-window rate limiter per IP address.

Limits:
    • Default  : 100 requests / 60 seconds
    • Inference: 20  requests / 60 seconds (heavy endpoints)
    • Register : 10  requests / 60 seconds

Uses in-memory store (swap for Redis in production).
"""

import time
from collections import defaultdict, deque
from typing import Deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# ── Rate Limit Config ─────────────────────────────────────────────────────────

RATE_LIMITS = {
    # endpoint prefix → (max_requests, window_seconds)
    "/api/v1/recognition"   : (20,  60),
    "/api/v1/register"      : (10,  60),
    "/api/v1/logs"          : (100, 60),
    "default"               : (100, 60),
}

# Whitelisted IPs (no rate limiting)
WHITELIST_IPS: set[str] = {"127.0.0.1", "::1"}


# ── Rate Limiter ──────────────────────────────────────────────────────────────

class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    Tracks request timestamps per (ip, endpoint) key.
    Evicts timestamps outside the window on each check.
    """

    def __init__(self):
        # { (ip, endpoint): deque of timestamps }
        self._store: dict[str, Deque[float]] = defaultdict(deque)

    def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit.

        Args:
            key            : unique key (ip + endpoint)
            max_requests   : allowed requests per window
            window_seconds : sliding window size

        Returns:
            (allowed, remaining, retry_after_seconds)
        """
        now = time.time()
        window_start = now - window_seconds
        timestamps = self._store[key]

        # Evict old timestamps outside window
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            # Rate limit exceeded
            retry_after = int(timestamps[0] + window_seconds - now) + 1
            return False, 0, retry_after

        # Allow and record
        timestamps.append(now)
        remaining = max_requests - len(timestamps)
        return True, remaining, 0


# Global limiter instance
_limiter = SlidingWindowRateLimiter()


# ── Middleware ────────────────────────────────────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Applies sliding window rate limiting per IP.

    Returns HTTP 429 with Retry-After header when limit exceeded.
    Adds rate limit headers to every response.
    """

    async def dispatch(self, request: Request, call_next):
        client_ip = self._get_client_ip(request)

        # Whitelist check
        if client_ip in WHITELIST_IPS:
            return await call_next(request)

        # Select rate limit config
        path = request.url.path
        max_req, window = self._get_limit(path)
        key = f"{client_ip}:{path}"

        allowed, remaining, retry_after = _limiter.is_allowed(
            key=key,
            max_requests=max_req,
            window_seconds=window,
        )

        if not allowed:
            return JSONResponse(
                status_code = 429,
                content     = {
                    "error"       : "Rate limit exceeded",
                    "retry_after" : retry_after,
                    "detail"      : (
                        f"Max {max_req} requests per "
                        f"{window}s exceeded"
                    ),
                },
                headers = {
                    "Retry-After"           : str(retry_after),
                    "X-RateLimit-Limit"     : str(max_req),
                    "X-RateLimit-Remaining" : "0",
                    "X-RateLimit-Window"    : str(window),
                },
            )

        # Allow → add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"]     = str(max_req)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"]    = str(window)
        return response

    def _get_limit(self, path: str) -> tuple[int, int]:
        """Select rate limit based on endpoint prefix."""
        for prefix, limits in RATE_LIMITS.items():
            if prefix != "default" and path.startswith(prefix):
                return limits
        return RATE_LIMITS["default"]

    def _get_client_ip(self, request: Request) -> str:
        for header in ["X-Forwarded-For", "X-Real-IP"]:
            val = request.headers.get(header)
            if val:
                return val.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
