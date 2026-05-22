from fastapi import APIRouter

router = APIRouter()


@router.post("/")
def recognize():
    return {"status": "ok", "message": "recognition endpoint placeholder"}
