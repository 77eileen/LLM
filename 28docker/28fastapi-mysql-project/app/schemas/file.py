from pydantic import BaseModel
from datetime import datetime

class FileResponse(BaseModel):
    id: int
    filename: str
    file_path: str
    file_size: int
    content_type: str
    created_at: datetime
    
    # 클라이언트에게 보여줄 다운로드/보기 URL (Computed Field처럼 사용 가능)
    @property
    def url(self) -> str:
        return f"http://localhost{self.file_path}"

    class Config:
        from_attributes = True