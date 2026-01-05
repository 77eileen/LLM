from pydantic import BaseModel   # 모든 스키마의 기본 클래스
from typing import Optional     # 선택필드
from datetime import datetime



############## 사용자 권한 관련 #######################

# 공통필드
class UserBase(BaseModel):
    username : str
    email : str
    full_name : Optional[str] = None
    role : str = 'user'  # user, manager, admin


class UserCreate(UserBase):
    password : str # 평문 패스워드
    

class UserUpdate(BaseModel):
    email : Optional[str] = None
    full_name : Optional[str] = None
    password : Optional[str] = None
    role : Optional[str] = None


class User(UserBase):
    id : int
    is_active : bool
    created_at : datetime
    updated_at : datetime
    class Config:
        from_attributes = True  # ORM 모델을 Pydantic 모델로 변환


class UserInDB(User):
    hashed_password : str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None



############## 제품 관련 #######################


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