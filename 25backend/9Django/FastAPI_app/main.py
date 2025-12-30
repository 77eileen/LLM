from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware  # Django 8000 포트와 FastAPI 8001 포트 연동시 필요. CORS 문제 해결
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List
import models
import schemas
from database import engine, get_db
from auth import *
from datetime import timedelta


# 테이블 생성
models.Base.metadata.create_all(bind=engine)


app = FastAPI(
    title= "product API",
    description="제품관리",
    #version="1.0.0" #제품관리
    version="2.0.0"
)


# CROS 설정 : Django 와 FastAPI 연동시 필요
app. add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],   # allow_origins=["*"] 모든 출처 허용 
    allow_credentials=True,  # 쿠키, 인증정보 허용
    allow_methods=["*"], # 모든 메서드 허용 : GET, POST, PUT, DELETE 
    allow_headers=["*"], # 모든 헤더 허용 : Authorization, Content-Type.. 있음
)



# 라우터 설정
@app.get('/')
def root():
    return {
        "message": "welcome to the Product API",
        'docs': '/docs',
        'endpoints': {
            'products': '/api/products',
            'product': '/api/products/{id}',
            'register': '/api/auth/register',  # 사용자관련추가
            'login': '/api/auth/token', # 사용자 관련 추가
            'me': '/api/auth/me' # 사용자 관련 추가
             }
    }


# 인증관련 (사용자관련 추가)
@app.post('/api/auth/register', response_model=schemas.User, status_code=status.HTTP_201_CREATED)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    '''회원가입'''
    # 중복체크 : username
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    # 중복체크 : 이메일 검증
    db_email = db.query(models.User).filter(models.User.email == user.email).first()
    if db_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Email {user.email} already registered")
    # 사용자 생성
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name,
        role=user.role,
        )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.post('/api/auth/token', response_model=schemas.Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
    ):
    '''로그인 및 토큰발급'''
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # 토큰 만료시간 설정
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, 'role': user.role}, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
            # "token_type": "bearer" 이 토큰을 bear 소지하고 있는 주체가 권한을 가진다




# Swagger ui (/docs 화면이 Swagger)에서 테스트할 때는 요청헤더에 로그인해서 발생된 토큰이 있어야함
# http://127.0.0.1:8001/docs/ UI에서 자물쇠 버튼을 클릭하고
# username / password 칸에 로그인 정보 넣고 Authorize 버튼을 눌러야 함
# 👉 그러면 Swagger가 자동으로 토큰을 받아서 Authorization 헤더에 붙여줌
# curl - X 'GET' \
#   'http://127.0.0.1:8001/api/auth/me' \
#   -H 'accept: application/json' \
#   -H 'Authorization: Bearer eyJhbGciOiJIUzI1 ~~~
@app.get('/api/auth/me', response_model=schemas.User)
def read_users_me(current_user: models.User = Depends(get_current_active_user)):
    '''현재 로그인한 사용자 정보 조회'''
    return current_user


@app.get('/api/auth/users', response_model=List[schemas.User])
def get_users(
    current_user: models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
    ):
    '''사용자 목록 조회 (관리자만)'''
    if not check_permission(current_user, 'admin'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied")
    users = db.query(models.User).all()
    return users    






# 제품목록 조회
# response_model
    # 반환 데이터 자동검증
    # ORM 모델 --> JSON 변환
    # Swagger 문서 자동 생성
@app.get("/api/products", response_model=List[schemas.Product])
def get_products(
    skip: int = 0,      # skip 0이면, skip 안함
    limit: int = 100,  # 한번에 100개씩 가져옴
    db: Session = Depends(get_db)   # 함수 실행이 끝나면 DB세션을 자동 종료
    ):
    products = db.query(models.Product).offset(skip).limit(limit).all()
    return products


# 제품 상세 조회
@app.get("/api/products/{id}", response_model=schemas.Product)
def get_product(
    id: int,
    db: Session = Depends(get_db)
    ):
    product = db.query(models.Product).filter(models.Product.id == id).first()  # first() 여러개 나왔을때, 첫번째 나오는 하나만 추출
    if not product :
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Product not found{id}")
    return product


# 제품생성
# 성공하면 HTTP_201_CREATED 상태 코드
@app.post("/api/products", response_model=schemas.Product, status_code=status.HTTP_201_CREATED)
def create_product(
    product: schemas.ProductCreate,
    db: Session = Depends(get_db)
    ):
    db_product = models.Product(**product.model_dump())
    # DB에 저장
    db.add(db_product)  # DB 세션에 저장
    db.commit()   # 실제 DB에 insert
    db.refresh(db_product) # 방금 저장된 데이터를 다시 조회
    return db_product


# 제품 수정(업데이트)
@app.put("/api/products/{id}", response_model=schemas.Product)
def update_product(
    id: int,
    product: schemas.ProductUpdate,
    db: Session = Depends(get_db)
    ): 
    db_product = db.query(models.Product).filter(models.Product.id == id).first()  # first() 여러개 나왔을때, 첫번째 나오는 하나만 추출
    if not db_product :
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Product not found{id}")
    update_product = product.model_dump(exclude_unset=True)  # 전송되지 않는 값들은 업데이트 되지 않도록. 전달된 필드만 업데이트 되도록. 
                                            # schemas.ProductUpdate 여기 들어가보면 name: Optional[str] = None 이기 때문에, 
                                            # 만약 가격만 변경하면 나머지가 None으로 업데이트 되므로, 
                                            # 이것을 방지하기 위해 변경된 가격만 반영하도록함.
    for key, value in update_product.items():
        setattr(db_product, key, value)  # 동적으로 속성 설정 / 변경 감지 기능이 있어서 업데이트된 필드만 반영                    
    db.commit()
    db.refresh(db_product) # 현재 서버에 있는 기준 최신버전을 사용자에게 보여지도록..?
    return db_product


# 제품 삭제
@app.delete("/api/products/{id}", status_code=status.HTTP_204_NO_CONTENT) # HTTP_204_NO_CONTENT 성공했지만 return 값은 없다
def delete_product(
    id: int,
    db: Session = Depends(get_db)
    ):
    product = db.query(models.Product).filter(models.Product.id == id).first()  # first() 여러개 나왔을때, 첫번째 나오는 하나만 추출
    if not product :
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Product not found{id}")
    db.delete(product)
    db.commit()
    return None