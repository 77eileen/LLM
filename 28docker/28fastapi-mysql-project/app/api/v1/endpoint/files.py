from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from app.db.session import get_db
from app.services.file_service import FileService
from app.schemas.file import FileResponse

router = APIRouter()

@router.post("/upload", response_model=FileResponse)
async def upload_file(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    db_file = await FileService.save_file(db, file)
    return db_file

@router.get("/", response_model=List[FileResponse])
async def read_files(skip: int = 0, limit: int = 100, db: AsyncSession = Depends(get_db)):
    return await FileService.get_files(db, skip=skip, limit=limit)

@router.get("/{file_id}", response_model=FileResponse)
async def read_file(file_id: int, db: AsyncSession = Depends(get_db)):
    file = await FileService.get_file(db, file_id)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    return file

@router.delete("/{file_id}")
async def delete_file(file_id: int, db: AsyncSession = Depends(get_db)):
    deleted_file = await FileService.delete_file(db, file_id)
    if not deleted_file:
        raise HTTPException(status_code=404, detail="File not found")
    return {"message": f"File {deleted_file.filename} deleted successfully"}