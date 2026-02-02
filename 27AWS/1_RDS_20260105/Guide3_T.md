# Guide2번까지 실습한후...  ec2 + RDS 

# 가상환경생성
```
   # 기존 venv 삭제 (선택사항)
    rm -rf venv
   
    # Python 3.11로 가상환경 생성
    python3.11 -m venv venv

  2. 가상환경 활성화
    source venv/bin/activate

  3. 버전 확인
  활성화된 상태에서 아래 명령어를 쳤을 때 Python 3.11.x가 나오면 성공입니다.

 4. 라이브러리 설치
 pip install fastapi uvicorn sqlalchemy pymysql pydantic dotenv
   python --version
```


fastapi_main.py
```
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from sqlalchemy import create_engine,Column, Integer, String,DateTime
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
import os
load_dotenv()

# RDS 정보 입력
RDS_HOST=os.getenv('RDS_HOST')
RDS_USER=os.getenv('RDS_USER')
RDS_PASSWORD=os.getenv('RDS_PASSWORD')
RDS_DB_NAME=os.getenv('RDS_DB_NAME')
RDS_PORT=os.getenv('RDS_PORT')

DATABASE_URL = f"mysql+pymysql://{RDS_USER}:{RDS_PASSWORD}@{RDS_HOST}:{RDS_PORT}/{RDS_DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False ,bind=engine)
Base = declarative_base()

# DB 모델
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    role = Column(String(20), default='member')
    created_at = Column(DateTime, default=datetime.now)

Base.metadata.create_all(bind=engine)

# pydantic 모델
class UserCreate(BaseModel):
    name:str
    email:str
    role:str = 'member'
class UserUpdate(BaseModel):
    name:str | None = None    
    role:str | None = None

class UserResponse(BaseModel):
    id:int
    name:str
    email:str
    role:str
    created_at:datetime
    class config:
        from_attributes = True
# FastAPI 앱
app = FastAPI()
# DB 접속 function - 공통
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get('/',summary='헬스체크')
def read_root():
    return {'status':'ok','time':datetime.now()}

#사용자 생성
@app.post('/users', response_model=UserResponse,status_code=status.HTTP_201_CREATED)
def create_user(user:UserCreate, db:Session=Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail='이미 존재하는 사용자입니다.')
    new_user = User(**user.model_dump())
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

#사용자 목록 조회
#사용자 검색
#사용자 정보 수정
#사용자 정보 삭제
       
```

# 실행
uvicorn fastapi_main:app --host 0.0.0.0 --port 8000 --reload



# 고정ip를 사용하고 싶을때
 ---
  EC2 자체의 공인 IP를 고정하고 싶을 때
  만약 EC2를 재부팅할 때마다 퍼블릭 IP가 바뀌는 게 싫다면 탄력적 IP (Elastic IP, EIP)를 사용해야 합니다.

   1. AWS 콘솔 → EC2 → 왼쪽 메뉴 [탄력적 IP].
   2. [탄력적 IP 주소 할당] → 할당 클릭.
   3. 생성된 IP 선택 → [작업] → [탄력적 IP 주소 연결].
   4. 내 EC2 인스턴스를 선택하고 연결.
   5. 주의: 탄력적 IP를 만들고 EC2에 연결하지 않거나, EC2를 중지(Stop) 해두면 요금(시간당 약 6원)이 부과됩니다. (실행 중인 인스턴스에 연결되어 있으면 무료)  
  ---