"""
Request / Response Logging Middleware.

Logs every request with:
    • Method, path, status code
    • Processing time (ms)
    • Client IP
    • Request ID (UUID injected into response headers)

Output: structured JSON logs → logs/access/
"""

import json
import time
import uuid
import logging
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# ── Logger Setup ──────────────────────────────────────────────────────────────

LOG_DIR = Path("logs/access")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Access logger → JSON file
access_logger = logging.getLogger("access")
access_logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_DIR / "access.log")
file_handler.setFormatter(logging.Formatter("%(message)s"))
access_logger.addHandler(file_handler)

# Console logger
console_logger = logging.getLogger("api")
console_logger.setLevel(logging.INFO)
console_logger.addHandler(logging.StreamHandler())


# ── Middleware ────────────────────────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs all incoming requests and outgoing responses.

    Injects X-Request-ID header into every response.
    Writes structured JSON to logs/access/access.log.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()

        # Attach request_id to state for downstream use
        request.state.request_id = request_id

        # Get client info
        client_ip = self._get_client_ip(request)

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            status_code = 500
            console_logger.error(
                f"Unhandled exception: {exc}", exc_info=True
            )
            raise

        # Calculate processing time
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Inject request ID into response headers
        response.headers["X-Request-ID"]    = request_id
        response.headers["X-Process-Time"]  = f"{elapsed_ms:.2f}ms"

        # Build log record
        log_record = {
            "request_id"  : request_id,
            "method"      : request.method,
            "path"        : request.url.path,
            "query"       : str(request.url.query),
            "status_code" : status_code,
            "elapsed_ms"  : round(elapsed_ms, 2),
            "client_ip"   : client_ip,
            "user_agent"  : request.headers.get("user-agent", ""),
            "user_id"     : getattr(request.state, "user", None),
        }

        # Write JSON log
        access_logger.info(json.dumps(log_record))

        # Console log
        level = "INFO" if status_code < 400 else "WARNING"
        console_logger.log(
            logging.getLevelName(level),
            f"[{request_id[:8]}] "
            f"{request.method} {request.url.path} "
            f"→ {status_code} ({elapsed_ms:.1f}ms)"
        )

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP (handles reverse proxy headers)."""
        # Check forwarded headers (nginx, AWS ALB, Cloudflare)
        for header in ["X-Forwarded-For", "X-Real-IP", "CF-Connecting-IP"]:
            value = request.headers.get(header)
            if value:
                return value.split(",")[0].strip()

        # Fallback to direct connection
        if request.client:
            return request.client.host
        return "unknown"
