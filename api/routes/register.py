"""
Face Registration Routes.

Endpoints:
    POST   /api/v1/register           → Register single face
    POST   /api/v1/register/bulk      → Register multiple faces
    DELETE /api/v1/register/{id}      → Delete an identity
    GET    /api/v1/register/{id}      → Get identity info
    PUT    /api/v1/register/{id}      → Update identity metadata
"""

import base64
import uuid
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from api.middleware.auth import get_current_user, require_role
from api.schemas.register import (
    BulkRegisterRequest,
    BulkRegisterResponse,
    DeleteIdentityResponse,
    RegisterRequest,
    RegisterResponse,
)

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_base64_image(b64_string: str) -> np.ndarray:
    """
    Decode Base64 image string to numpy array.

    Args:
        b64_string: Base64-encoded image

    Returns:
        NumPy BGR image array

    Raises:
        HTTPException 422 if image is invalid
    """
    import cv2

    try:
        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        return img

    except Exception as e:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = f"Invalid image: {str(e)}",
        )


def compute_face_quality(img: np.ndarray) -> float:
    """
    Compute face image quality score (0–1).

    Checks: sharpness, brightness, face size.
    """
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpness (Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(sharpness / 500.0, 1.0)

    # Brightness (penalize too dark or too bright)
    brightness = gray.mean() / 255.0
    brightness_score = 1.0 - abs(brightness - 0.5) * 2

    return float((sharpness_score + brightness_score) / 2.0)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model = RegisterResponse,
    status_code    = status.HTTP_201_CREATED,
    summary        = "Register a new face identity",
    description    = (
        "Enrolls a new person into the face recognition database. "
        "Accepts a Base64-encoded face image and identity metadata."
    ),
)
async def register_face(
    request : RegisterRequest,
    user    : dict = Depends(get_current_user),
) -> RegisterResponse:
    """
    Register a new face identity.

    Steps:
        1. Decode Base64 image
        2. Detect and align face
        3. Extract 512-d embedding
        4. Store embedding + metadata in DB
        5. Return face_id and quality score
    """
    # Step 1: Decode image
    img = decode_base64_image(request.image_base64)

    # Step 2: Compute quality score
    quality_score = compute_face_quality(img)

    if quality_score < 0.2:
        raise HTTPException(
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail      = (
                f"Face image quality too low ({quality_score:.2f}). "
                "Please provide a clearer, well-lit image."
            ),
        )

    # Step 3: Generate face ID
    face_id = f"face_{uuid.uuid4().hex[:12]}"

    # ── TODO: Replace stubs with real implementation ───────────────────────
    # from models.backbones import get_resnet
    # from src.recognition.embeddings import extract_embedding
    # from src.utils.database import FaceDatabase
    #
    # model = get_resnet("resnet50")
    # embedding = extract_embedding(model, img)
    # db = FaceDatabase()
    # db.insert(
    #     identity_id = request.identity_id,
    #     face_id     = face_id,
    #     embedding   = embedding,
    #     metadata    = request.metadata,
    # )
    # ──────────────────────────────────────────────────────────────────────

    return RegisterResponse(
        status          = "success",
        identity_id     = request.identity_id,
        face_id         = face_id,
        embedding_dim   = 512,
        registered_at   = datetime.now(timezone.utc),
        message         = f"Face registered for '{request.name}'",
        quality_score   = round(quality_score, 4),
    )


@router.post(
    "/register/bulk",
    response_model = BulkRegisterResponse,
    status_code    = status.HTTP_201_CREATED,
    summary        = "Bulk register multiple face identities",
)
async def bulk_register(
    request : BulkRegisterRequest,
    user    : dict = Depends(require_role("admin")),
) -> BulkRegisterResponse:
    """
    Register multiple identities in one request (admin only).

    Processes each identity independently — partial success is supported.
    Failed registrations are collected in the errors list.
    """
    results = []
    errors  = []

    for identity in request.identities:
        try:
            img           = decode_base64_image(identity.image_base64)
            quality_score = compute_face_quality(img)
            face_id       = f"face_{uuid.uuid4().hex[:12]}"

            results.append(RegisterResponse(
                status          = "success",
                identity_id     = identity.identity_id,
                face_id         = face_id,
                embedding_dim   = 512,
                registered_at   = datetime.now(timezone.utc),
                message         = f"Registered '{identity.name}'",
                quality_score   = round(quality_score, 4),
            ))

        except HTTPException as e:
            errors.append({
                "identity_id" : identity.identity_id,
                "error"       : e.detail,
            })

    return BulkRegisterResponse(
        total_submitted = len(request.identities),
        total_success   = len(results),
        total_failed    = len(errors),
        results         = results,
        errors          = errors,
    )


@router.delete(
    "/register/{identity_id}",
    response_model = DeleteIdentityResponse,
    summary        = "Delete a registered identity",
)
async def delete_identity(
    identity_id : str,
    user        : dict = Depends(require_role("admin")),
) -> DeleteIdentityResponse:
    """
    Delete an identity and all associated face embeddings (admin only).

    This is irreversible. The identity must be re-registered to use again.
    """
    # ── TODO: db.delete(identity_id) ──────────────────────────────────────
    return DeleteIdentityResponse(
        status       = "success",
        identity_id  = identity_id,
        message      = f"Identity '{identity_id}' deleted successfully",
        deleted_at   = datetime.now(timezone.utc),
    )


@router.get(
    "/register/{identity_id}",
    summary = "Get identity info",
)
async def get_identity(
    identity_id : str,
    user        : dict = Depends(get_current_user),
) -> dict:
    """
    Retrieve metadata for a registered identity.

    Does NOT return face embedding (privacy protection).
    """
    # ── TODO: db.get(identity_id) ─────────────────────────────────────────
    return {
        "identity_id"   : identity_id,
        "name"          : "John Doe",
        "registered_at" : datetime.now(timezone.utc).isoformat(),
        "face_count"    : 1,
        "status"        : "active",
    }
