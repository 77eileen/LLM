from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.models.base import Base

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)      # 원본 파일명
    saved_filename = Column(String(255), nullable=False) # 서버에 저장된 유니크한 파일명 (UUID 등)
    file_path = Column(String(500), nullable=False)     # 실제 저장 경로 (ex: /uploads/image.png)
    file_size = Column(Integer, nullable=False)         # 파일 크기 (Bytes)
    content_type = Column(String(100), nullable=False)  # MIME Type (image/jpeg, video/mp4)
    created_at = Column(DateTime(timezone=True), server_default=func.now()) # 업로드 시간