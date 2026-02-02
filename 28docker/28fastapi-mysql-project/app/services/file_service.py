import os
import uuid
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.file_model import UploadedFile

UPLOAD_DIR = "uploads"

from sqlalchemy.future import select

class FileService:
    @staticmethod
    async def save_file(db: AsyncSession, file: UploadFile):
        # ... (기존 코드 생략) ...
        # 1. 디렉토리 생성 확인
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        # 2. 파일명 중복 방지를 위한 UUID 생성
        file_extension = os.path.splitext(file.filename)[1]
        saved_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, saved_filename)

        # 3. 로컬 저장소에 실제 파일 저장
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # 4. DB에 메타데이터 저장
        db_file = UploadedFile(
            filename=file.filename,
            saved_filename=saved_filename,
            file_path=f"/{UPLOAD_DIR}/{saved_filename}", # Nginx 접근 경로
            file_size=len(content),
            content_type=file.content_type
        )
        db.add(db_file)
        await db.commit()
        await db.refresh(db_file)
        
        return db_file

    @staticmethod
    async def delete_file(db: AsyncSession, file_id: int):
        # 1. DB에서 파일 정보 조회
        result = await db.execute(select(UploadedFile).filter(UploadedFile.id == file_id))
        db_file = result.scalars().first()
        
        if not db_file:
            return None

        # 2. 실제 파일 삭제 (os.remove)
        # DB에 저장된 경로는 /uploads/... 형식이므로 앞의 /를 제거해야 로컬 경로와 맞음
        local_path = db_file.file_path.lstrip("/")
        
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception as e:
                print(f"Error deleting file {local_path}: {e}")

        # 3. DB 레코드 삭제
        await db.delete(db_file)
        await db.commit()
        return db_file

    @staticmethod
    async def get_files(db: AsyncSession, skip: int = 0, limit: int = 100):
        result = await db.execute(select(UploadedFile).order_by(UploadedFile.created_at.desc()).offset(skip).limit(limit))
        return result.scalars().all()

    @staticmethod
    async def get_file(db: AsyncSession, file_id: int):
        result = await db.execute(select(UploadedFile).filter(UploadedFile.id == file_id))
        return result.scalars().first()