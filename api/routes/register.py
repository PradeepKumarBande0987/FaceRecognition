from fastapi import APIRouter

router = APIRouter()


@router.post("/")
def register():
    return {"status": "ok", "message": "register endpoint placeholder"}
