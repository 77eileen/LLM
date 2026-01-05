from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from argon2 import PasswordHasher
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from database import get_db
import models
import secrets
import os
from dotenv import load_dotenv
load_dotenv()




# 지금 이 코드는 한 줄로 말하면 👇
# 👉 **“로그인한 사람에게만 API 접근 권한을 주기 위한 준비 코드”**예요.
# 흐름은 이렇게 됩니다:
# 사용자가 아이디 / 비밀번호로 로그인
# 서버가 비밀번호를 검증
# 맞으면 JWT 토큰 발급
# 이후 요청마다 토큰을 들고 와야 접근 허용




# 보안 설정
SECRET_KEY = os.getenv('SECRET_KEY')
    # JWT 토큰에 서명할 때 쓰는 key, 토큰이 서버에서 만든게 맞는지 검증 (실제는 .env에 저장해서 사용)
    # 서버를 재시작할때마다 secret_key 발급되면서 기존키는 무효화
    # 서버 실행시 기준키를 재발행 --> 모든 사용자의 토큰이 무효과 --> 강제 로그아웃
ALGORITHM = 'HS256' 
    # JWT를 어떤 암호 알고리즘으로 서명할지/ HS256 대칭키 방식(SECTRET_KEY 하나로 서명+검증)
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 토큰 유효시간


# 해싱:
# 서버도 비밀번호 원본을 몰라야 안전
# DB에는 해싱된 값만 저장
# 로그인시 : 입력한 비밀번호를 같은 방식으로 해싱, DB값과 비교
    # 절대 X : if password == db_password
    # 대신 : pwd_context.verify(password, db_password)
ph = PasswordHasher()
#schemes=["bcrypt"] 비밀번호 해싱하는 방식 
# deprecated="auto" 나중에 더 좋은 해싱 방식이 나오면 기존 비밀번호를 자동으로 업그레이드 가능하게 하겠다
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")
# 이건 “토큰이 어디서 발급되는지 FastAPI에게 알려주는 설정”
# Swagger(UI)에서: 
    # 👤 아이디 / 🔑 비밀번호 입력
    # → api/auth/token 으로 요청
    # → JWT 토큰 발급
# 로그인 api 엔드포인트 지정/ 아이디,패스워드를 보내서 토큰을 받음.


# 비밀번호 검증하는 함수
def verify_password(plain_password:str, hashed_password:str) -> bool:
    '''패스워드 검증'''
    return ph.verify(hashed_password, plain_password)

def get_password_hash(password:str) -> str:
    '''패스워드 해시 생성'''  # 해싱.. 암호화?
    password = password[:72]
    return ph.hash(password)


# 토큰 생성
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    '''엑세스 토큰 생성'''
    to_encode = data.copy()
    if expires_delta:  # 유효기간
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt



def authenticate_user (db: Session, username: str, password: str):
    '''사용자 인증'''
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user :
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    '''현재 사용자 정보 가져오기'''  
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try: 
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception 
    except JWTError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: models.User = Depends(get_current_user)):
    '''활성화된 현재 사용자 정보 가져오기'''
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user



def check_permission(user: models.User, required_role: str) -> bool:
    '''권한 확인'''
    # 숫자가 클수록 권한이 높음
    role_hierarchy={
        "admin": 3,
        "manager": 2,
        "user": 1
    }
    user_level = role_hierarchy.get(user.role, 0)
    required_level = role_hierarchy.get(required_role, 0)
    return user_level >= required_level  



if __name__ == "__main__":
    print(SECRET_KEY)