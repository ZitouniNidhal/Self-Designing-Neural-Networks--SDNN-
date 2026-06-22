from fastapi import FastAPI
from .endpoints import router


def create_app() -> FastAPI:
    app = FastAPI(title="ArchitectAI API")
    app.include_router(router, prefix="/api")
    return app
