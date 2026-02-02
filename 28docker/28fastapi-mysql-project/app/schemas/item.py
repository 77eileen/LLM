from pydantic import BaseModel
from typing import Optional

# 공통 속성
class ItemBase(BaseModel):
    title: str
    description: Optional[str] = None
    owner_name: Optional[str] = None

# 생성 시 필요한 속성 (Client -> Server)
class ItemCreate(ItemBase):
    pass

# 수정 시 필요한 속성
class ItemUpdate(ItemBase):
    pass

# 조회 시 반환할 속성 (Server -> Client)
class ItemResponse(ItemBase):
    id: int

    class Config:
        from_attributes = True # ORM 객체를 Pydantic 모델로 변환 허용 (구 orm_mode)