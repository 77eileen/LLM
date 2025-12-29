# Django + FastAPI 연동 프로젝트 학습 가이드 🚀

## 📌 프로젝트 개요

이 프로젝트는 **백엔드(FastAPI)**와 **프론트엔드(Django)**를 분리해서 만든 웹 애플리케이션입니다!

### 전체 구조
```
사용자 브라우저
    ↓
Django (메인 서버, 8000 포트)
    ↓ HTTP 요청
FastAPI (API/데이터 서버, 8001 포트)
    ↓
SQLite DB (products.db)
```

---

## 🎯 왜 이렇게 나눠서 만들까?

| 기술 | 역할 | 강점 |
|------|------|------|
| **Django** | 웹 인터페이스 제공 | 관리자 페이지, 템플릿, 사용자 관리 |
| **FastAPI** | RESTful API 제공 | 빠른 속도, 자동 문서화, AI 모델 연동 |

**실무 포인트**: 대규모 서비스에서는 프론트엔드와 백엔드를 분리해서 각자의 강점을 살립니다!

---

## 📁 프로젝트 구조 상세 설명

### 1️⃣ FastAPI_app (백엔드 - 데이터 처리)

```
FastAPI_app/
├── main.py                 # ⭐ FastAPI 서버 메인 파일
├── models.py               # 데이터베이스 테이블 정의 (SQLAlchemy)
├── schemas.py              # 데이터 검증 규칙 (Pydantic)
├── database.py             # DB 연결 설정
├── add_sample_data.py      # 샘플 데이터 추가 스크립트
├── products.db             # SQLite 데이터베이스 파일
└── requirements.txt        # 필요한 패키지 목록
```

#### 주요 파일 역할

**main.py**
- FastAPI 앱 실행
- API 엔드포인트 정의 (상품 조회, 생성, 수정, 삭제)
- CORS 설정 (Django와 통신 허용)
- 실행: `uvicorn main:app --port 8001 --reload`
- 문서: `http://127.0.0.1:8001/docs`

**models.py**
```python
# 예시: 상품 테이블 정의
class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    price = Column(Float)
    description = Column(String)
```

**schemas.py**
```python
# 예시: 데이터 검증
class ProductCreate(BaseModel):
    name: str
    price: float
    description: str
```

**database.py**
- SQLite 연결 설정
- 세션 관리

---

### 2️⃣ django_project (프론트엔드 - 웹 인터페이스)

```
django_project/
├── config/                      # Django 프로젝트 설정
│   ├── settings.py             # ⭐ 전체 설정 (앱 등록, DB, 정적 파일)
│   ├── urls.py                 # 메인 URL 라우팅
│   ├── wsgi.py
│   └── asgi.py
│
├── products/                    # 상품 관리 앱
│   ├── views.py                # ⭐ 화면 로직 (FastAPI 호출)
│   ├── urls.py                 # products 앱 URL 패턴
│   ├── forms.py                # 입력 폼 정의
│   ├── models.py               # (여기선 사용 안 함)
│   ├── admin.py
│   └── apps.py
│
├── templates/                   # HTML 템플릿
│   ├── base.html               # 공통 레이아웃 (Bootstrap)
│   └── products/
│       ├── product_list.html   # 상품 목록 페이지
│       └── product_form.html   # 상품 생성/수정 폼
│
├── static/                      # 정적 파일 (CSS, JS, 이미지)
├── db.sqlite3                   # Django 기본 DB (사용자 관리용)
└── manage.py                    # Django 관리 명령어
```

#### 주요 파일 역할

**config/settings.py**
```python
INSTALLED_APPS = [
    'products',  # 상품 앱 등록
    # ...
]

TEMPLATES = [
    'DIRS': [BASE_DIR / 'templates'],  # 템플릿 경로
]

LANGUAGE_CODE = 'ko-kr'
TIME_ZONE = 'Asia/Seoul'

# FastAPI 서버 주소
FASTAPI_BASE_URL = 'http://127.0.0.1:8001'
```

**products/views.py**
```python
import httpx  # FastAPI와 HTTP 통신

def product_list(request):
    # FastAPI에서 상품 목록 가져오기
    response = httpx.get('http://127.0.0.1:8001/products/')
    products = response.json()
    return render(request, 'products/product_list.html', {'products': products})
```

**products/urls.py**
```python
urlpatterns = [
    path('', views.product_list, name='product_list'),
    path('create/', views.product_create, name='product_create'),
    path('<int:id>/update/', views.product_update, name='product_update'),
    path('<int:id>/delete/', views.product_delete, name='product_delete'),
]
```

---

## 🔄 데이터 흐름 이해하기

### 예시: 상품 목록 보기

```
1. 사용자가 브라우저에서 http://127.0.0.1:8000/products/ 접속

2. Django (urls.py) → products/views.py의 product_list() 실행

3. views.py에서 httpx로 FastAPI 호출
   → GET http://127.0.0.1:8001/products/

4. FastAPI (main.py)에서 DB 조회 후 JSON 반환
   → [{"id": 1, "name": "노트북", "price": 1000000}, ...]

5. Django가 JSON 데이터를 받아서 템플릿에 전달

6. product_list.html이 렌더링되어 사용자에게 보임
```

### 예시: 상품 생성하기

```
1. 사용자가 폼 작성 후 제출

2. Django views.py의 product_create()에서 폼 데이터 받음

3. httpx로 FastAPI에 POST 요청
   → POST http://127.0.0.1:8001/products/
   → Body: {"name": "마우스", "price": 50000, "description": "..."}

4. FastAPI에서 DB에 저장

5. Django에서 성공 메시지와 함께 목록 페이지로 리다이렉트
```

---

## 🛠️ 설치 및 실행 순서

### 1단계: 환경 설정
```bash
# Django 설치
conda install django -y

# 필요한 패키지 설치
pip install httpx fastapi uvicorn sqlalchemy pydantic
```

### 2단계: FastAPI 서버 시작 (터미널 1)
```bash
cd FastAPI_app
uvicorn main:app --port 8001 --reload
```
- 확인: http://127.0.0.1:8001/docs

### 3단계: Django 서버 시작 (터미널 2)
```bash
cd django_project
python manage.py migrate          # DB 초기화
python manage.py createsuperuser  # 관리자 계정 생성
python manage.py runserver
```
- 확인: http://127.0.0.1:8000/products/

---

## 🔑 핵심 개념 정리

### CORS (Cross-Origin Resource Sharing)
- **문제**: 브라우저는 보안상 다른 포트(출처)의 서버와 통신을 기본적으로 차단
- **해결**: FastAPI의 `CORSMiddleware`로 Django(8000)의 요청 허용

```python
# FastAPI main.py
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000"],  # Django 주소
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### httpx 라이브러리
- Django에서 FastAPI로 HTTP 요청을 보내는 도구
- `requests`와 비슷하지만 더 현대적

```python
import httpx

# GET 요청
response = httpx.get('http://127.0.0.1:8001/products/')
data = response.json()

# POST 요청
response = httpx.post('http://127.0.0.1:8001/products/', 
                      json={"name": "상품", "price": 10000})
```

### SQLAlchemy ORM
- Python 객체로 데이터베이스 조작
- SQL 문을 직접 쓰지 않아도 됨

---

## 📚 주요 기능별 코드 흐름

### 1. 상품 목록 조회
```
URL: /products/
Django urls.py → views.product_list()
→ httpx.get('FastAPI/products/')
→ FastAPI: DB 조회
→ Django: 템플릿 렌더링
```

### 2. 상품 생성
```
URL: /products/create/
Django: forms.py로 입력 검증
→ httpx.post('FastAPI/products/')
→ FastAPI: DB 저장
→ Django: 리다이렉트
```

### 3. 상품 수정
```
URL: /products/<id>/update/
Django: 기존 데이터 조회 (GET FastAPI)
→ 폼에 표시
→ 수정 후 제출 (PUT/PATCH FastAPI)
→ DB 업데이트
```

### 4. 상품 삭제
```
URL: /products/<id>/delete/
Django: httpx.delete('FastAPI/products/<id>')
→ FastAPI: DB에서 삭제
→ Django: 목록 페이지로 리다이렉트
```

---

## 💡 학습 포인트

### 초급 단계
1. ✅ 각 파일의 역할 이해하기
2. ✅ Django와 FastAPI 서버를 동시에 실행하는 이유
3. ✅ httpx로 API 호출하는 방법

### 중급 단계
1. ✅ CORS 개념과 설정 방법
2. ✅ Pydantic으로 데이터 검증
3. ✅ SQLAlchemy ORM 사용법
4. ✅ Django Form과 Template 활용

### 고급 단계
1. ✅ 에러 핸들링 (FastAPI 서버가 꺼져있을 때)
2. ✅ 인증/권한 추가
3. ✅ Docker로 배포 환경 구축
4. ✅ 프로덕션 환경 설정 (Gunicorn, Nginx)

---

## 🚀 실습 아이디어

1. **새로운 필드 추가**: 상품에 '재고 수량' 필드 추가해보기
2. **검색 기능**: 상품명으로 검색하는 API와 화면 만들기
3. **카테고리**: 상품 카테고리 기능 추가 (1:N 관계)
4. **파일 업로드**: 상품 이미지 업로드 기능
5. **페이지네이션**: 상품이 많을 때 페이지 나누기

---

## 📖 참고 자료

- **FastAPI 공식 문서**: https://fastapi.tiangolo.com/
- **Django 공식 문서**: https://docs.djangoproject.com/
- **SQLAlchemy 문서**: https://docs.sqlalchemy.org/
- **Bootstrap 문서**: https://getbootstrap.com/

---

## ❓ 자주하는 질문 (FAQ)

**Q: 왜 Django의 ORM을 안 쓰고 FastAPI에서 SQLAlchemy를 쓰나요?**  
A: 각 서버가 독립적으로 동작하도록 하기 위함입니다. FastAPI는 자체 DB를 관리하고, Django는 화면만 담당합니다.

**Q: 실무에서도 이렇게 하나요?**  
A: 네! 마이크로서비스 아키텍처에서 자주 사용하는 패턴입니다.

**Q: FastAPI 대신 Django REST Framework는 안 되나요?**  
A: 됩니다! 하지만 FastAPI가 더 빠르고 자동 문서화가 편리합니다.

---

## ✅ 체크리스트

학습 완료 여부를 확인해보세요!

- [ ] FastAPI와 Django 서버를 동시에 실행할 수 있다
- [ ] 상품 목록을 조회할 수 있다
- [ ] 새 상품을 추가할 수 있다
- [ ] 상품을 수정할 수 있다
- [ ] 상품을 삭제할 수 있다
- [ ] CORS가 무엇인지 설명할 수 있다
- [ ] httpx로 API를 호출하는 방법을 안다
- [ ] FastAPI의 자동 문서를 확인했다 (/docs)
- [ ] 각 파일의 역할을 이해했다

---

**작성일**: 2025.12.29  
**버전**: 1.0