"""API Schemas — Pydantic v2 models for all endpoints."""

from .register import (
    RegisterRequest,
    RegisterResponse,
    BulkRegisterRequest,
    BulkRegisterResponse,
    DeleteIdentityResponse,
    Gender,
)

from .recognition import (
    IdentifyRequest,
    IdentifyResponse,
    VerifyRequest,
    VerifyResponse,
    BatchIdentifyRequest,
    BatchIdentifyResponse,
    FaceMatch,
    LivenessCheck,
    LivenessResult,
    ConfidenceLevel,
    RecognitionMode,
)

from .logs import (
    LogQueryParams,
    InferenceLogEntry,
    AccessLogEntry,
    AuditLogEntry,
    InferenceLogsResponse,
    AccessLogsResponse,
    AuditLogsResponse,
    LogStatsResponse,
    LogLevel,
    EventType,
)

__all__ = [
    # Register
    "RegisterRequest", "RegisterResponse",
    "BulkRegisterRequest", "BulkRegisterResponse",
    "DeleteIdentityResponse", "Gender",

    # Recognition
    "IdentifyRequest", "IdentifyResponse",
    "VerifyRequest", "VerifyResponse",
    "BatchIdentifyRequest", "BatchIdentifyResponse",
    "FaceMatch", "LivenessCheck",
    "LivenessResult", "ConfidenceLevel", "RecognitionMode",

    # Logs
    "LogQueryParams", "InferenceLogEntry",
    "AccessLogEntry", "AuditLogEntry",
    "InferenceLogsResponse", "AccessLogsResponse",
    "AuditLogsResponse", "LogStatsResponse",
    "LogLevel", "EventType",
]
