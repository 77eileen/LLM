from fastapi import APIRouter
from app.api.v1.endpoint import items, files

api_router = APIRouter()
api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(files.router, prefix="/files", tags=["files"])

