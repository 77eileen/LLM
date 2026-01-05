from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()

# RDS 정보 입력
RDS_HOST = os.getenv('RDS_HOST')
RDS_USER = os.getenv('RDS_USER')
RDS_PASSWORD = os.getenv('RDS_PASSWORD')
RDS_DB_NAME = os.getenv('RDS_DB_NAME')
RDS_PORT = int(os.getenv('RDS_PORT'))


# mysql connection string
DATABASE_URL = f"mysql+pymysql://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# DB모델
class User (Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    role = Column(String(20), default='member')
    created_at = Column(DateTime, default=datetime.now)

Base.metadata.create_all(bind=engine)

# pydantic 모델
class UserCreate(BaseModel):
    name: str
    email: str
    role: str ='member'

class UserUpdate(BaseModel):
    name:str | None = None    
    role:str | None = None

class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    created_at: datetime
    class config:
        from_attributes = True

# FastAPI 앱
app = FastAPI()

# DB접속 function - 공통
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", summary='헬스체크')
def read_root():
    return {'status':'ok', 'time':datetime.now()}


# 사용자 생성
@app.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = User(**user.model_dump())
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user
# 사용자 목록조회
# 사용자 검색
# 사용자 정보 수정
# 사용자 정보 삭제




