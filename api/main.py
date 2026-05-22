from fastapi import FastAPI
from api.routes import recognition, register, logs


def create_app() -> FastAPI:
    app = FastAPI(title="Face Recognition API")
    app.include_router(recognition.router, prefix="/recognize", tags=["recognition"])
    app.include_router(register.router, prefix="/register", tags=["register"])
    app.include_router(logs.router, prefix="/logs", tags=["logs"])
    return app
