"""
Face Recognition API — Main Application Entry Point.

Stack:
    • FastAPI  0.135.x
    • Pydantic v2.13
    • Python   3.10+

Endpoints:
    POST /api/v1/register          → Register a new face identity
    POST /api/v1/recognition/identify  → Identify face from image
    POST /api/v1/recognition/verify    → Verify two faces match
    GET  /api/v1/logs              → Access inference & audit logs
    GET  /health                   → Health check
    GET  /docs                     → Swagger UI (auto-generated)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.middleware.auth       import AuthMiddleware
from api.middleware.logging    import RequestLoggingMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.routes                import recognition, register, logs


# ── App Lifespan (replaces deprecated @app.on_event) ─────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Startup  : load face recognition model into memory
    Shutdown : clean up GPU resources, close DB connections
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    print("🚀 Starting Face Recognition API...")

    # Load model (lazy import to avoid circular deps)
    from api.routes.recognition import load_model
    await load_model()

    print("✅ Model loaded. API ready.")

    yield  # ← App runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    print("🛑 Shutting down API. Releasing resources...")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title          = "Face Recognition API",
    description    = (
        "Production-grade face recognition API supporting:\n"
        "- **Face Registration** (enroll identities)\n"
        "- **Face Identification** (1:N search)\n"
        "- **Face Verification** (1:1 matching)\n"
        "- **Liveness Detection** (anti-spoofing)\n"
        "- **Audit Logging** (GDPR-aware)"
    ),
    version        = "1.0.0",
    docs_url       = "/docs",
    redoc_url      = "/redoc",
    openapi_url    = "/openapi.json",
    lifespan       = lifespan,                 # ✅ FastAPI 0.93+ lifespan
)


# ── Middleware Stack ──────────────────────────────────────────────────────────
# Order matters: added last = executed first on request

# 1. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins      = ["http://localhost:3000", "https://yourapp.com"],
    allow_credentials  = True,
    allow_methods      = ["GET", "POST", "PUT", "DELETE"],
    allow_headers      = ["*"],
)

# 2. GZip compression for large responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 3. Rate limiting
app.add_middleware(RateLimitMiddleware)

# 4. Request / response logging
app.add_middleware(RequestLoggingMiddleware)

# 5. JWT auth (applied per-route via Depends, but also globally here)
app.add_middleware(AuthMiddleware)


# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(
    register.router,
    prefix = "/api/v1",
    tags   = ["Registration"],
)

app.include_router(
    recognition.router,
    prefix = "/api/v1",
    tags   = ["Recognition"],
)

app.include_router(
    logs.router,
    prefix = "/api/v1",
    tags   = ["Logs & Audit"],
)


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get(
    "/health",
    tags    = ["Health"],
    summary = "Health check endpoint",
)
async def health_check():
    """Returns API status and version info."""
    return JSONResponse({
        "status"  : "healthy",
        "version" : "1.0.0",
        "model"   : "ArcFace-ResNet50",
    })


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Face Recognition API. Visit /docs for Swagger UI."}
