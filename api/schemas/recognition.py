"""
Pydantic v2 Schemas for Face Recognition endpoints.

Covers:
    • Face Identification (1:N)  — who is this person?
    • Face Verification   (1:1)  — are these the same person?
    • Liveness Detection         — is this a real face?
    • Batch Recognition          — multiple faces at once
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class RecognitionMode(str, Enum):
    IDENTIFY = "identify"    # 1:N search
    VERIFY   = "verify"      # 1:1 matching


class LivenessResult(str, Enum):
    REAL    = "real"
    SPOOF   = "spoof"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    HIGH    = "high"      # score > 0.85
    MEDIUM  = "medium"    # score 0.65 – 0.85
    LOW     = "low"       # score < 0.65


# ── Request Schemas ───────────────────────────────────────────────────────────

class IdentifyRequest(BaseModel):
    """
    Request body for 1:N face identification.

    Searches the face database and returns the
    top-K most similar identities.
    """
    model_config = ConfigDict(
        str_strip_whitespace = True,
        json_schema_extra    = {
            "example": {
                "image_base64"   : "<base64_image>",
                "top_k"          : 5,
                "threshold"      : 0.6,
                "check_liveness" : True,
            }
        }
    )

    image_base64   : Annotated[
        str,
        Field(
            min_length  = 100,
            description = "Base64-encoded query face image",
        )
    ]

    top_k          : Annotated[
        int,
        Field(
            default     = 5,
            ge          = 1,
            le          = 50,
            description = "Number of top matches to return",
        )
    ]

    threshold      : Annotated[
        float,
        Field(
            default     = 0.60,
            ge          = 0.0,
            le          = 1.0,
            description = "Minimum similarity score to include in results",
        )
    ]

    check_liveness : bool = Field(
        default     = True,
        description = "Run anti-spoofing liveness check before recognition",
    )

    return_embedding : bool = Field(
        default     = False,
        description = "Include face embedding vector in response",
    )

    @field_validator("image_base64")
    @classmethod
    def strip_data_uri(cls, v: str) -> str:
        if v.startswith("data:image"):
            v = v.split(",", 1)[-1]
        return v


class VerifyRequest(BaseModel):
    """
    Request body for 1:1 face verification.

    Compares two face images and returns a
    match/no-match decision with similarity score.
    """
    model_config = ConfigDict(
        str_strip_whitespace = True,
        json_schema_extra    = {
            "example": {
                "image1_base64" : "<base64_image_1>",
                "image2_base64" : "<base64_image_2>",
                "threshold"     : 0.6,
            }
        }
    )

    image1_base64 : Annotated[
        str,
        Field(min_length=100, description="First face image (Base64)")
    ]

    image2_base64 : Annotated[
        str,
        Field(min_length=100, description="Second face image (Base64)")
    ]

    threshold     : float = Field(
        default     = 0.60,
        ge          = 0.0,
        le          = 1.0,
        description = "Similarity threshold for match decision",
    )

    check_liveness : bool = Field(default=False)

    @model_validator(mode="after")
    def strip_data_uris(self) -> "VerifyRequest":
        """Strip data URI prefix from both images."""
        for field in ["image1_base64", "image2_base64"]:
            v = getattr(self, field)
            if v.startswith("data:image"):
                setattr(self, field, v.split(",", 1)[-1])
        return self


class BatchIdentifyRequest(BaseModel):
    """Request body for batch identification (multiple images at once)."""

    images_base64  : Annotated[
        list[str],
        Field(
            min_length  = 1,
            max_length  = 20,
            description = "List of Base64-encoded face images (max 20)",
        )
    ]

    top_k          : int   = Field(default=3, ge=1, le=20)
    threshold      : float = Field(default=0.60, ge=0.0, le=1.0)
    check_liveness : bool  = Field(default=True)


# ── Response Schemas ──────────────────────────────────────────────────────────

class FaceMatch(BaseModel):
    """A single face match result from identification."""

    identity_id      : str
    name             : str
    similarity_score : Annotated[
        float,
        Field(ge=0.0, le=1.0)
    ]
    confidence       : ConfidenceLevel
    department       : Optional[str]  = None
    metadata         : Optional[dict] = None


class LivenessCheck(BaseModel):
    """Liveness detection result."""

    result          : LivenessResult
    confidence      : Annotated[float, Field(ge=0.0, le=1.0)]
    spoof_type      : Optional[str]  = None   # "print", "replay", "mask"
    processing_ms   : Optional[float] = None


class IdentifyResponse(BaseModel):
    """Response from 1:N face identification."""

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "status"            : "success",
                "face_detected"     : True,
                "matches"           : [],
                "liveness"          : None,
                "processing_time_ms": 45.3,
                "request_id"        : "req_abc123",
            }
        }
    )

    status             : str
    face_detected      : bool
    matches            : list[FaceMatch]
    top_match          : Optional[FaceMatch]  = None
    liveness           : Optional[LivenessCheck] = None
    embedding          : Optional[list[float]]   = None
    face_quality_score : Optional[float]         = Field(
        default=None, ge=0.0, le=1.0
    )
    processing_time_ms : float
    request_id         : str
    timestamp          : datetime


class VerifyResponse(BaseModel):
    """Response from 1:1 face verification."""

    status             : str
    is_match           : bool
    similarity_score   : Annotated[float, Field(ge=0.0, le=1.0)]
    threshold_used     : float
    confidence         : ConfidenceLevel
    liveness           : Optional[LivenessCheck] = None
    processing_time_ms : float
    request_id         : str
    timestamp          : datetime


class BatchIdentifyResponse(BaseModel):
    """Response from batch identification."""

    total_images       : int
    successful         : int
    failed             : int
    results            : list[IdentifyResponse]
    total_time_ms      : float
    timestamp          : datetime
