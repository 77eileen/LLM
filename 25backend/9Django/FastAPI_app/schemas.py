from pydantic import BaseModel   # 모든 스키마의 기본 클래스
from typing import Optional     # 선택필드
from datetime import datetime

# 공통필드
class ProductBase(BaseModel):
    name: str
    description : Optional[str] = None
    price: float
    stock: int


# 생성용 스키마 
    # 생성시에만 필요한 스키마
    # 관리자 전용
class ProductCreate(ProductBase):
    pass


# 수정용 스키마
    # 전달될 필드만 업데이트
    # PATCH, PUT 지원
    # 실제 사용할 때는 : product.model_dump(exclude_unset=True) 
class ProductUpdate(BaseModel):
    name: Optional[str] = None
    description : Optional[str] = None
    price: Optional[float] = None
    stock: Optional[int] = None


# 응답용 스키마
class Product(ProductBase):
    id : int
    created_at : datetime
    updated_at : datetime
    class Config:
        from_atributes = True  # ORM 모델을 Pydantic 모델로 변환