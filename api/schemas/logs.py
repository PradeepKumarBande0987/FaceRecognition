"""
Pydantic v2 Schemas for Logs & Audit endpoints.

Covers:
    • Inference logs   — what the model processed
    • Access logs      — who called what endpoint
    • Audit logs       — GDPR-relevant data access events
    • Training logs    — model training history
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class LogLevel(str, Enum):
    DEBUG   = "debug"
    INFO    = "info"
    WARNING = "warning"
    ERROR   = "error"
    CRITICAL = "critical"


class EventType(str, Enum):
    REGISTER      = "register"
    IDENTIFY      = "identify"
    VERIFY        = "verify"
    DELETE        = "delete"
    LOGIN         = "login"
    UNAUTHORIZED  = "unauthorized"
    ERROR         = "error"
    SYSTEM        = "system"


class LogSortBy(str, Enum):
    TIMESTAMP   = "timestamp"
    LEVEL       = "level"
    EVENT_TYPE  = "event_type"
    IDENTITY_ID = "identity_id"


# ── Query Params Schema ───────────────────────────────────────────────────────

class LogQueryParams(BaseModel):
    """
    Query parameters for log retrieval.

    Used with FastAPI Depends() for clean query param validation.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    level        : Optional[LogLevel]   = None
    event_type   : Optional[EventType]  = None
    identity_id  : Optional[str]        = Field(default=None, max_length=64)
    start_time   : Optional[datetime]   = None
    end_time     : Optional[datetime]   = None
    limit        : Annotated[
        int, Field(default=50, ge=1, le=1000)
    ]
    offset       : Annotated[
        int, Field(default=0, ge=0)
    ]
    sort_by      : LogSortBy = LogSortBy.TIMESTAMP
    descending   : bool      = True


# ── Log Entry Schemas ─────────────────────────────────────────────────────────

class InferenceLogEntry(BaseModel):
    """A single inference log entry."""

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "log_id"           : "log_001",
                "event_type"       : "identify",
                "identity_id"      : "user_001",
                "similarity_score" : 0.87,
                "is_match"         : True,
                "liveness_result"  : "real",
                "processing_ms"    : 42.5,
                "model_version"    : "arcface-r50-v1.2",
                "timestamp"        : "2026-06-13T12:00:00Z",
            }
        }
    )

    log_id            : str
    event_type        : EventType
    identity_id       : Optional[str]   = None
    similarity_score  : Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = None
    is_match          : Optional[bool]  = None
    liveness_result   : Optional[str]   = None
    face_quality      : Optional[float] = Field(default=None, ge=0.0, le=1.0)
    processing_ms     : float
    model_version     : str
    api_version       : str             = "1.0.0"
    ip_address        : Optional[str]   = None
    user_agent        : Optional[str]   = None
    timestamp         : datetime


class AccessLogEntry(BaseModel):
    """A single API access log entry."""

    log_id       : str
    method       : str           # GET, POST, etc.
    endpoint     : str
    status_code  : int
    response_ms  : float
    ip_address   : Optional[str] = None
    user_id      : Optional[str] = None
    api_key_id   : Optional[str] = None
    timestamp    : datetime


class AuditLogEntry(BaseModel):
    """
    GDPR-aware audit log entry.

    Records all data access events for compliance.
    """

    log_id       : str
    event_type   : EventType
    actor_id     : str           # who performed the action
    subject_id   : Optional[str] = None  # whose data was accessed
    action       : str
    resource     : str
    outcome      : str           # "success" | "failure" | "denied"
    reason       : Optional[str] = None
    ip_address   : Optional[str] = None
    timestamp    : datetime


# ── Response Schemas ──────────────────────────────────────────────────────────

class InferenceLogsResponse(BaseModel):
    """Paginated inference log response."""

    total        : int
    limit        : int
    offset       : int
    logs         : list[InferenceLogEntry]
    has_more     : bool


class AccessLogsResponse(BaseModel):
    """Paginated access log response."""

    total        : int
    limit        : int
    offset       : int
    logs         : list[AccessLogEntry]
    has_more     : bool


class AuditLogsResponse(BaseModel):
    """Paginated audit log response."""

    total        : int
    limit        : int
    offset       : int
    logs         : list[AuditLogEntry]
    has_more     : bool


class LogStatsResponse(BaseModel):
    """Summary statistics for logs dashboard."""

    total_requests      : int
    total_identifications : int
    total_verifications : int
    total_registrations : int
    avg_processing_ms   : float
    match_rate          : Annotated[float, Field(ge=0.0, le=1.0)]
    liveness_pass_rate  : Annotated[float, Field(ge=0.0, le=1.0)]
    error_rate          : Annotated[float, Field(ge=0.0, le=1.0)]
    period_start        : datetime
    period_end          : datetime
