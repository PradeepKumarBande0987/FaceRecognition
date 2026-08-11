"""
Pydantic v2 Schemas for Face Registration endpoints.

All schemas use Pydantic v2.13 syntax:
    • model_config = ConfigDict(...)     replaces class Config
    • Field(...)                         with json_schema_extra
    • Annotated types for validation
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class Gender(str, Enum):
    MALE    = "male"
    FEMALE  = "female"
    OTHER   = "other"
    UNKNOWN = "unknown"


class ImageFormat(str, Enum):
    JPEG = "jpeg"
    PNG  = "png"
    WEBP = "webp"


# ── Request Schemas ───────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    """
    Request body for registering a new face identity.

    The image must be Base64-encoded JPEG/PNG.
    """
    model_config = ConfigDict(
        str_strip_whitespace = True,
        json_schema_extra    = {
            "example": {
                "identity_id"  : "user_001",
                "name"         : "John Doe",
                "image_base64" : "<base64_encoded_image>",
                "metadata"     : {"department": "engineering"},
            }
        }
    )

    identity_id  : Annotated[
        str,
        Field(
            min_length  = 3,
            max_length  = 64,
            pattern     = r"^[a-zA-Z0-9_\-]+$",
            description = "Unique identity identifier",
        )
    ]

    name         : Annotated[
        str,
        Field(
            min_length  = 2,
            max_length  = 128,
            description = "Full name of the person",
        )
    ]

    image_base64 : Annotated[
        str,
        Field(
            min_length  = 100,
            description = "Base64-encoded face image (JPEG or PNG)",
        )
    ]

    gender       : Gender          = Gender.UNKNOWN
    age_estimate : Optional[int]   = Field(default=None, ge=0, le=120)
    department   : Optional[str]   = Field(default=None, max_length=64)
    metadata     : Optional[dict]  = Field(default=None)

    @field_validator("image_base64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Strip data URI prefix if present."""
        if v.startswith("data:image"):
            # Remove: "data:image/jpeg;base64,"
            v = v.split(",", 1)[-1]
        return v

    @field_validator("identity_id")
    @classmethod
    def validate_identity_id(cls, v: str) -> str:
        """Ensure no reserved keywords."""
        reserved = {"admin", "root", "system", "null", "none"}
        if v.lower() in reserved:
            raise ValueError(f"identity_id '{v}' is reserved")
        return v


class BulkRegisterRequest(BaseModel):
    """Request body for bulk face registration (multiple faces)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    identities : Annotated[
        list[RegisterRequest],
        Field(
            min_length  = 1,
            max_length  = 100,
            description = "List of identities to register (max 100)",
        )
    ]


# ── Response Schemas ──────────────────────────────────────────────────────────

class RegisterResponse(BaseModel):
    """Response after successful face registration."""

    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "status"        : "success",
                "identity_id"   : "user_001",
                "face_id"       : "face_abc123",
                "embedding_dim" : 512,
                "registered_at" : "2026-06-13T12:00:00Z",
                "message"       : "Face registered successfully",
            }
        }
    )

    status          : str
    identity_id     : str
    face_id         : str
    embedding_dim   : int       = 512
    registered_at   : datetime
    message         : str
    quality_score   : Optional[float] = Field(
        default = None,
        ge      = 0.0,
        le      = 1.0,
        description = "Face image quality score (0-1)",
    )


class BulkRegisterResponse(BaseModel):
    """Response after bulk registration."""

    total_submitted  : int
    total_success    : int
    total_failed     : int
    results          : list[RegisterResponse]
    errors           : list[dict]    = Field(default_factory=list)


class DeleteIdentityResponse(BaseModel):
    """Response after deleting an identity."""

    status       : str
    identity_id  : str
    message      : str
    deleted_at   : datetime
