
# 터미널 실행
# uvicorn fastapi_s3_main:app --host 0.0.0.0 --port 8000 --reload



from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
import boto3
import uuid
from dotenv import load_dotenv
import os
load_dotenv()


# RDS 정보 입력
RDS_HOST = os.getenv('RDS_HOST')
RDS_USER = os.getenv('RDS_USER')
RDS_PASSWORD = os.getenv('RDS_PASSWORD')
RDS_DB_NAME = os.getenv('RDS_DB_NAME')
RDS_PORT = int(os.getenv('RDS_PORT'))


# AWS S3
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')


# RDS - mysql 접속 정보 및 base객체 생성
DATABASE_URL = f"mysql+pymysql://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# FastAPI 앱
app = FastAPI(title="S3 + RDS Integration")


# DB접속 function - 공통
def get_db():
    db = SessionLocal()
    try:f
        yield db
    finally:
        db.close()


# 이미지 정보를 저장할 테이블 정의
class UserImage(Base):
    __tablename__ = 'user_images'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50))
    image_url = Column(String(255)) # s3 url 
    file_name = Column(String(255))


# S3 클라이언트 설정
s3_client = boto3.client(  #boto3 aws s3 통신용
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)


# S3 endpoint
@app.post('/upload/', tags=['S3-RDS'])
async def upload_image_to_s3_and_db(
    username: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    '''
    1. 파일을 S3에 업로드
    2. 생성된 S3 URL을 RDS MySQL에 저장    
    '''
    # S3 파일 업로드 로직
    file_name = file.filename
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    s3_key = f"uploads/{username}/{unique_filename}"
    try: 
        # S3 전송
        s3_client.upload_fileobj(file.file, S3_BUCKET_NAME, s3_key,
                                 ExtraArgs={'ContentType': file.content_type})
        # S3 공개 URL생성
        image_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"

        # RDS 저장 로직
        new_image = UserImage(username=username, file_name=file_name, image_url=image_url)
        db.add(new_image)
        db.commit()
        db.refresh(new_image) 
        return {
            'status':"success",
            'username': username,
            'image_url': image_url,
            'db_id' : new_image.id
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/images/", tags=['S3-RDS'])
def get_all_images(db: Session = Depends(get_db)):
    '''DB에 저장된 모든 이미지 목록 조회'''
    images = db.query(UserImage).all()
    return images






########################## RDS ########################

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




