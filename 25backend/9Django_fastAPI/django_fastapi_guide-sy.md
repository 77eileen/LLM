# Django와 FastAPI 프로젝트 설계 완벽 가이드

## 목차
1. [Django 프로젝트 설계](#django-프로젝트-설계)
2. [FastAPI 프로젝트 설계](#fastapi-프로젝트-설계)
3. [CRUD 전체 구현 (수정/삭제 포함)](#crud-전체-구현)
4. [Django와 FastAPI 연동](#django와-fastapi-연동)

---

# 🎨 Django 프로젝트 설계

## 1단계: 프로젝트 초기 설정

```bash
# Django 설치
pip install django

# 프로젝트 생성
django-admin startproject config .

# 앱 생성
python manage.py startapp products
```

**결과 구조:**
```
django_project/
├── config/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── manage.py
```

---

## 2단계: 데이터베이스 모델 설계

**products/models.py** - 데이터 구조 정의
```python
from django.db import models

class Product(models.Model):
    name = models.CharField(max_length=200, verbose_name="상품명")
    description = models.TextField(verbose_name="설명")
    price = models.DecimalField(max_digits=10, decimal_places=2, verbose_name="가격")
    stock = models.IntegerField(verbose_name="재고")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="생성일")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="수정일")
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "상품"
        verbose_name_plural = "상품 목록"
    
    def __str__(self):
        return self.name
```

---

## 3단계: 설정 파일 구성

**config/settings.py** - 앱 등록
```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'products',  # 👈 우리가 만든 앱 추가
]

# 템플릿 경로 설정
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# 정적 파일 설정
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

# 한국어 설정
LANGUAGE_CODE = 'ko-kr'
TIME_ZONE = 'Asia/Seoul'
USE_I18N = True
USE_TZ = True
```

---

## 4단계: 데이터베이스 마이그레이션

```bash
# 마이그레이션 파일 생성
python manage.py makemigrations

# 데이터베이스에 적용
python manage.py migrate
```

---

## 5단계: 관리자 페이지 설정

**products/admin.py**
```python
from django.contrib import admin
from .models import Product

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ['name', 'price', 'stock', 'created_at', 'updated_at']
    list_filter = ['created_at', 'updated_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('기본 정보', {
            'fields': ('name', 'description')
        }),
        ('가격 및 재고', {
            'fields': ('price', 'stock')
        }),
        ('시간 정보', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
```

```bash
# 관리자 계정 생성
python manage.py createsuperuser
```

---

## 6단계: 뷰(View) 설계 - CRUD 전체 구현

**products/views.py**
```python
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from .models import Product
from .forms import ProductForm

# CREATE - 상품 생성
def product_create(request):
    if request.method == 'POST':
        form = ProductForm(request.POST)
        if form.is_valid():
            product = form.save()
            messages.success(request, f'{product.name} 상품이 등록되었습니다.')
            return redirect('product_detail', pk=product.pk)
    else:
        form = ProductForm()
    
    return render(request, 'products/product_form.html', {
        'form': form,
        'title': '상품 등록'
    })

# READ - 상품 목록
def product_list(request):
    products = Product.objects.all()
    return render(request, 'products/product_list.html', {
        'products': products
    })

# READ - 상품 상세
def product_detail(request, pk):
    product = get_object_or_404(Product, pk=pk)
    return render(request, 'products/product_detail.html', {
        'product': product
    })

# UPDATE - 상품 수정
def product_update(request, pk):
    product = get_object_or_404(Product, pk=pk)
    
    if request.method == 'POST':
        form = ProductForm(request.POST, instance=product)
        if form.is_valid():
            product = form.save()
            messages.success(request, f'{product.name} 상품이 수정되었습니다.')
            return redirect('product_detail', pk=product.pk)
    else:
        form = ProductForm(instance=product)
    
    return render(request, 'products/product_form.html', {
        'form': form,
        'product': product,
        'title': '상품 수정'
    })

# DELETE - 상품 삭제
def product_delete(request, pk):
    product = get_object_or_404(Product, pk=pk)
    
    if request.method == 'POST':
        product_name = product.name
        product.delete()
        messages.success(request, f'{product_name} 상품이 삭제되었습니다.')
        return redirect('product_list')
    
    return render(request, 'products/product_confirm_delete.html', {
        'product': product
    })
```

---

## 7단계: 폼(Form) 설계

**products/forms.py**
```python
from django import forms
from .models import Product

class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name', 'description', 'price', 'stock']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '상품명을 입력하세요'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': '상품 설명을 입력하세요'
            }),
            'price': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': '가격을 입력하세요'
            }),
            'stock': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': '재고를 입력하세요'
            }),
        }
    
    def clean_price(self):
        price = self.cleaned_data.get('price')
        if price and price < 0:
            raise forms.ValidationError('가격은 0보다 커야 합니다.')
        return price
    
    def clean_stock(self):
        stock = self.cleaned_data.get('stock')
        if stock and stock < 0:
            raise forms.ValidationError('재고는 0보다 커야 합니다.')
        return stock
```

---

## 8단계: URL 라우팅

**products/urls.py** (새로 생성)
```python
from django.urls import path
from . import views

urlpatterns = [
    path('', views.product_list, name='product_list'),
    path('create/', views.product_create, name='product_create'),
    path('<int:pk>/', views.product_detail, name='product_detail'),
    path('<int:pk>/update/', views.product_update, name='product_update'),
    path('<int:pk>/delete/', views.product_delete, name='product_delete'),
]
```

**config/urls.py**
```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('products/', include('products.urls')),
]
```

---

## 9단계: 템플릿(HTML) 설계

**templates/products/base.html**
```html
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}상품 관리{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding-top: 20px; }
        .messages { margin: 20px 0; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="{% url 'product_list' %}">상품 관리 시스템</a>
            <div class="navbar-nav">
                <a class="nav-link" href="{% url 'product_list' %}">상품 목록</a>
                <a class="nav-link" href="{% url 'product_create' %}">상품 추가</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        {% if messages %}
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show" role="alert">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        {% endif %}

        {% block content %}{% endblock %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

**templates/products/product_list.html**
```html
{% extends 'products/base.html' %}

{% block title %}상품 목록{% endblock %}

{% block content %}
<div class="d-flex justify-content-between align-items-center mb-4">
    <h1>상품 목록</h1>
    <a href="{% url 'product_create' %}" class="btn btn-primary">새 상품 등록</a>
</div>

{% if products %}
<div class="table-responsive">
    <table class="table table-striped table-hover">
        <thead class="table-dark">
            <tr>
                <th>번호</th>
                <th>상품명</th>
                <th>가격</th>
                <th>재고</th>
                <th>등록일</th>
                <th>관리</th>
            </tr>
        </thead>
        <tbody>
            {% for product in products %}
            <tr>
                <td>{{ product.id }}</td>
                <td>
                    <a href="{% url 'product_detail' product.pk %}">{{ product.name }}</a>
                </td>
                <td>{{ product.price|floatformat:0 }}원</td>
                <td>{{ product.stock }}개</td>
                <td>{{ product.created_at|date:"Y-m-d H:i" }}</td>
                <td>
                    <a href="{% url 'product_update' product.pk %}" class="btn btn-sm btn-warning">수정</a>
                    <a href="{% url 'product_delete' product.pk %}" class="btn btn-sm btn-danger">삭제</a>
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% else %}
<div class="alert alert-info">등록된 상품이 없습니다.</div>
{% endif %}
{% endblock %}
```

**templates/products/product_detail.html**
```html
{% extends 'products/base.html' %}

{% block title %}{{ product.name }} - 상세정보{% endblock %}

{% block content %}
<div class="card">
    <div class="card-header d-flex justify-content-between align-items-center">
        <h2>{{ product.name }}</h2>
        <div>
            <a href="{% url 'product_update' product.pk %}" class="btn btn-warning">수정</a>
            <a href="{% url 'product_delete' product.pk %}" class="btn btn-danger">삭제</a>
            <a href="{% url 'product_list' %}" class="btn btn-secondary">목록으로</a>
        </div>
    </div>
    <div class="card-body">
        <div class="row mb-3">
            <div class="col-md-3 fw-bold">상품명:</div>
            <div class="col-md-9">{{ product.name }}</div>
        </div>
        <div class="row mb-3">
            <div class="col-md-3 fw-bold">설명:</div>
            <div class="col-md-9">{{ product.description }}</div>
        </div>
        <div class="row mb-3">
            <div class="col-md-3 fw-bold">가격:</div>
            <div class="col-md-9">{{ product.price|floatformat:0 }}원</div>
        </div>
        <div class="row mb-3">
            <div class="col-md-3 fw-bold">재고:</div>
            <div class="col-md-9">{{ product.stock }}개</div>
        </div>
        <div class="row mb-3">
            <div class="col-md-3 fw-bold">등록일:</div>
            <div class="col-md-9">{{ product.created_at|date:"Y년 m월 d일 H:i" }}</div>
        </div>
        <div class="row mb-3">
            <div class="col-md-3 fw-bold">수정일:</div>
            <div class="col-md-9">{{ product.updated_at|date:"Y년 m월 d일 H:i" }}</div>
        </div>
    </div>
</div>
{% endblock %}
```

**templates/products/product_form.html**
```html
{% extends 'products/base.html' %}

{% block title %}{{ title }}{% endblock %}

{% block content %}
<div class="card">
    <div class="card-header">
        <h2>{{ title }}</h2>
    </div>
    <div class="card-body">
        <form method="post">
            {% csrf_token %}
            
            {% for field in form %}
            <div class="mb-3">
                <label for="{{ field.id_for_label }}" class="form-label">
                    {{ field.label }}
                </label>
                {{ field }}
                {% if field.errors %}
                    <div class="text-danger">
                        {% for error in field.errors %}
                            <small>{{ error }}</small>
                        {% endfor %}
                    </div>
                {% endif %}
            </div>
            {% endfor %}
            
            <div class="d-flex gap-2">
                <button type="submit" class="btn btn-primary">저장</button>
                <a href="{% url 'product_list' %}" class="btn btn-secondary">취소</a>
                {% if product %}
                <a href="{% url 'product_detail' product.pk %}" class="btn btn-info">상세보기</a>
                {% endif %}
            </div>
        </form>
    </div>
</div>
{% endblock %}
```

**templates/products/product_confirm_delete.html**
```html
{% extends 'products/base.html' %}

{% block title %}상품 삭제 확인{% endblock %}

{% block content %}
<div class="card border-danger">
    <div class="card-header bg-danger text-white">
        <h2>상품 삭제 확인</h2>
    </div>
    <div class="card-body">
        <div class="alert alert-warning">
            <strong>경고!</strong> 다음 상품을 정말 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.
        </div>
        
        <div class="mb-3">
            <strong>상품명:</strong> {{ product.name }}
        </div>
        <div class="mb-3">
            <strong>가격:</strong> {{ product.price|floatformat:0 }}원
        </div>
        <div class="mb-3">
            <strong>재고:</strong> {{ product.stock }}개
        </div>
        
        <form method="post" class="d-inline">
            {% csrf_token %}
            <button type="submit" class="btn btn-danger">삭제</button>
            <a href="{% url 'product_detail' product.pk %}" class="btn btn-secondary">취소</a>
        </form>
    </div>
</div>
{% endblock %}
```

---

# ⚡ FastAPI 프로젝트 설계

## 1단계: 프로젝트 초기 설정

```bash
# 필요한 패키지 설치
pip install fastapi uvicorn sqlalchemy pydantic

# 프로젝트 폴더 생성
mkdir fastapi_app
cd fastapi_app
```

---

## 2단계: 데이터베이스 설정

**database.py**
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLite 데이터베이스 파일 경로
SQLALCHEMY_DATABASE_URL = "sqlite:///./products.db"

# 엔진 생성
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}  # SQLite 전용
)

# 세션 생성기
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 베이스 클래스
Base = declarative_base()

# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## 3단계: 모델 설계

**models.py**
```python
from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database import Base

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    description = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    stock = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

---

## 4단계: 스키마 설계 (Pydantic)

**schemas.py**
```python
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional

# 상품 생성/수정 시 입력 데이터
class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="상품명")
    description: str = Field(..., min_length=1, description="상품 설명")
    price: float = Field(..., gt=0, description="가격 (0보다 커야 함)")
    stock: int = Field(..., ge=0, description="재고 (0 이상)")
    
    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('가격은 0보다 커야 합니다')
        return v
    
    @validator('stock')
    def stock_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('재고는 0 이상이어야 합니다')
        return v

# 상품 수정용 (모든 필드 optional)
class ProductUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=1)
    price: Optional[float] = Field(None, gt=0)
    stock: Optional[int] = Field(None, ge=0)

# 상품 응답 데이터
class ProductResponse(BaseModel):
    id: int
    name: str
    description: str
    price: float
    stock: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# 삭제 응답
class DeleteResponse(BaseModel):
    message: str
    deleted_id: int
```

---

## 5단계: 메인 애플리케이션 - CRUD 전체 구현

**main.py**
```python
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

import models, schemas
from database import engine, get_db

# 테이블 생성
models.Base.metadata.create_all(bind=engine)

# FastAPI 앱 생성
app = FastAPI(
    title="상품 관리 API",
    description="상품 CRUD API",
    version="1.0.0"
)

# CREATE - 상품 생성
@app.post(
    "/products/",
    response_model=schemas.ProductResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Products"],
    summary="상품 생성"
)
def create_product(
    product: schemas.ProductCreate,
    db: Session = Depends(get_db)
):
    """
    새로운 상품을 생성합니다.
    
    - **name**: 상품명 (필수)
    - **description**: 상품 설명 (필수)
    - **price**: 가격 (0보다 큰 값, 필수)
    - **stock**: 재고 (0 이상, 필수)
    """
    db_product = models.Product(**product.dict())
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product

# READ - 모든 상품 조회
@app.get(
    "/products/",
    response_model=List[schemas.ProductResponse],
    tags=["Products"],
    summary="모든 상품 조회"
)
def get_products(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    모든 상품 목록을 조회합니다.
    
    - **skip**: 건너뛸 개수 (페이징용)
    - **limit**: 최대 조회 개수
    """
    products = db.query(models.Product).offset(skip).limit(limit).all()
    return products

# READ - 특정 상품 조회
@app.get(
    "/products/{product_id}",
    response_model=schemas.ProductResponse,
    tags=["Products"],
    summary="특정 상품 조회"
)
def get_product(product_id: int, db: Session = Depends(get_db)):
    """
    특정 ID의 상품을 조회합니다.
    """
    product = db.query(models.Product).filter(
        models.Product.id == product_id
    ).first()
    
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID {product_id}인 상품을 찾을 수 없습니다"
        )
    
    return product

# UPDATE - 상품 수정 (전체)
@app.put(
    "/products/{product_id}",
    response_model=schemas.ProductResponse,
    tags=["Products"],
    summary="상품 전체 수정"
)
def update_product(
    product_id: int,
    product: schemas.ProductCreate,
    db: Session = Depends(get_db)
):
    """
    특정 ID의 상품을 전체 수정합니다 (모든 필드 필수).
    """
    db_product = db.query(models.Product).filter(
        models.Product.id == product_id
    ).first()
    
    if not db_product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID {product_id}인 상품을 찾을 수 없습니다"
        )
    
    # 모든 필드 업데이트
    for key, value in product.dict().items():
        setattr(db_product, key, value)
    
    db.commit()
    db.refresh(db_product)
    return db_product

# UPDATE - 상품 부분 수정
@app.patch(
    "/products/{product_id}",
    response_model=schemas.ProductResponse,
    tags=["Products"],
    summary="상품 부분 수정"
)
def partial_update_product(
    product_id: int,
    product: schemas.ProductUpdate,
    db: Session = Depends(get_db)
):
    """
    특정 ID의 상품을 부분 수정합니다 (원하는 필드만 수정 가능).
    """
    db_product = db.query(models.Product).filter(
        models.Product.id == product_id
    ).first()
    
    if not db_product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID {product_id}인 상품을 찾을 수 없습니다"
        )
    
    # 제공된 필드만 업데이트
    update_data = product.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_product, key, value)
    
    db.commit()
    db.refresh(db_product)
    return db_product

# DELETE - 상품 삭제
@app.delete(
    "/products/{product_id}",
    response_model=schemas.DeleteResponse,
    tags=["Products"],
    summary="상품 삭제"
)
def delete_product(product_id: int, db: Session = Depends(get_db)):
    """
    특정 ID의 상품을 삭제합니다.
    """
    db_product = db.query(models.Product).filter(
        models.Product.id == product_id
    ).first()
    
    if not db_product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ID {product_id}인 상품을 찾을 수 없습니다"
        )
    
    product_name = db_product.name
    db.delete(db_product)
    db.commit()
    
    return {
        "message": f"'{product_name}' 상품이 삭제되었습니다",
        "deleted_id": product_id
    }

# 서버 상태 확인
@app.get("/", tags=["Root"])
def read_root():
    """
    API 서버 상태를 확인합니다.
    """
    return {
        "message": "상품 관리 API 서버가 정상 작동 중입니다",
        "docs": "/docs",
        "redoc": "/redoc"
    }
```

---

## 6단계: 샘플 데이터 추가

**add_sample_data.py**
```python
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models

models.Base.metadata.create_all(bind=engine)

def add_sample_products():
    db = SessionLocal()
    
    # 기존 데이터 확인
    existing_count = db.query(models.Product).count()
    if existing_count > 0:
        print(f"이미 {existing_count}개의 상품이 존재합니다.")
        db.close()
        return
    
    products = [
        models.Product(
            name="MacBook Pro 16\"",
            description="Apple M3 Max 칩, 36GB 통합 메모리, 1TB SSD",
            price=4390000,
            stock=15
        ),
        models.Product(
            name="iPhone 15 Pro",
            description="티타늄 디자인, A17 Pro 칩, 256GB",
            price=1550000,
            stock=30
        ),
        models.Product(
            name="AirPods Pro (2세대)",
            description="적응형 오디오, 투명 모드, USB-C",
            price=359000,
            stock=50
        ),
        models.Product(
            name="Magic Mouse",
            description="무선 마우스, 멀티터치 표면, 충전식 배터리",
            price=99000,
            stock=100
        ),
        models.Product(
            name="LG 그램 17",
            description="17인치, Intel i7, 16GB RAM, 512GB SSD",
            price=2190000,
            stock=20
        ),
    ]
    
    db.add_all(products)
    db.commit()
    db.close()
    print(f"✅ {len(products)}개의 샘플 상품이 추가되었습니다!")

if __name__ == "__main__":
    add_sample_products()
```

**실행:**
```bash
python add_sample_data.py
```

---

## 7단계: 서버 실행

```bash
# FastAPI 서버 실행
uvicorn main:app --reload --port 8001

# 자동 문서 확인
# Swagger UI: http://localhost:8001/docs
# ReDoc: http://localhost:8001/redoc
```

---

# 🔄 Django와 FastAPI 연동하기

## 방법 1: Django에서 FastAPI 호출

**Django에 requests 설치:**
```bash
pip install requests
```

**products/views.py에 추가:**
```python
import requests
from django.shortcuts import render
from django.contrib import messages

FASTAPI_URL = "http://localhost:8001"

def api_products(request):
    """FastAPI에서 상품 목록 가져오기"""
    try:
        response = requests.get(f'{FASTAPI_URL}/products/')
        response.raise_for_status()
        products = response.json()
        
        return render(request, 'products/api_list.html', {
            'products': products,
            'api_url': FASTAPI_URL
        })
    except requests.exceptions.RequestException as e:
        messages.error(request, f'API 서버 연결 실패: {str(e)}')
        return render(request, 'products/api_list.html', {
            'products': [],
            'api_url': FASTAPI_URL
        })

def api_product_detail(request, product_id):
    """FastAPI에서 특정 상품 조회"""
    try:
        response = requests.get(f'{FASTAPI_URL}/products/{product_id}')
        response.raise_for_status()
        product = response.json()
        
        return render(request, 'products/api_detail.html', {
            'product': product
        })
    except requests.exceptions.RequestException as e:
        messages.error(request, f'상품을 찾을 수 없습니다: {str(e)}')
        return redirect('api_products')
```

**products/urls.py에 추가:**
```python
urlpatterns = [
    # 기존 URL들...
    path('api/', views.api_products, name='api_products'),
    path('api/<int:product_id>/', views.api_product_detail, name='api_product_detail'),
]
```

**templates/products/api_list.html:**
```html
{% extends 'products/base.html' %}

{% block title %}API 상품 목록{% endblock %}

{% block content %}
<div class="d-flex justify-content-between align-items-center mb-4">
    <h1>API 상품 목록</h1>
    <div>
        <span class="badge bg-info">FastAPI 연동</span>
        <a href="{{ api_url }}/docs" target="_blank" class="btn btn-sm btn-outline-primary">API 문서</a>
    </div>
</div>

{% if products %}
<div class="table-responsive">
    <table class="table table-striped">
        <thead class="table-dark">
            <tr>
                <th>ID</th>
                <th>상품명</th>
                <th>가격</th>
                <th>재고</th>
                <th>등록일</th>
            </tr>
        </thead>
        <tbody>
            {% for product in products %}
            <tr>
                <td>{{ product.id }}</td>
                <td>
                    <a href="{% url 'api_product_detail' product.id %}">{{ product.name }}</a>
                </td>
                <td>{{ product.price|floatformat:0 }}원</td>
                <td>{{ product.stock }}개</td>
                <td>{{ product.created_at|slice:":10" }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% else %}
<div class="alert alert-warning">
    API 서버에서 상품을 가져올 수 없습니다. FastAPI 서버가 실행 중인지 확인하세요.
</div>
{% endif %}
{% endblock %}
```

---

## 방법 2: 같은 데이터베이스 공유

**Django settings.py:**
```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR.parent / 'fastapi_app' / 'products.db',  # FastAPI DB 경로
    }
}
```

**FastAPI database.py:**
```python
# Django 프로젝트의 DB를 사용
SQLALCHEMY_DATABASE_URL = "sqlite:///../django_project/db.sqlite3"
```

---

# 📊 CRUD 작업 비교표

| 작업 | Django | FastAPI |
|------|--------|---------|
| **생성 (CREATE)** | `form.save()` | `db.add()` + `db.commit()` |
| **조회 (READ)** | `Product.objects.all()` | `db.query(Product).all()` |
| **수정 (UPDATE)** | `form.save()` | `setattr()` + `db.commit()` |
| **삭제 (DELETE)** | `product.delete()` | `db.delete()` + `db.commit()` |
| **유효성 검증** | Django Forms | Pydantic Schema |
| **URL 라우팅** | `urls.py` | 데코레이터 (`@app.get`) |

---

# 🎯 API 테스트 방법

## 1. Swagger UI 사용 (권장)
```
http://localhost:8001/docs
```
- 모든 엔드포인트를 웹 브라우저에서 테스트 가능
- 자동 생성된 문서와 함께 제공

## 2. curl 명령어

**상품 목록 조회:**
```bash
curl http://localhost:8001/products/
```

**상품 생성:**
```bash
curl -X POST http://localhost:8001/products/ \
  -H "Content-Type: application/json" \
  -d '{
    "name": "테스트 상품",
    "description": "테스트 설명",
    "price": 10000,
    "stock": 100
  }'
```

**상품 수정:**
```bash
curl -X PUT http://localhost:8001/products/1 \
  -H "Content-Type: application/json" \
  -d '{
    "name": "수정된 상품",
    "description": "수정된 설명",
    "price": 20000,
    "stock": 50
  }'
```

**상품 부분 수정:**
```bash
curl -X PATCH http://localhost:8001/products/1 \
  -H "Content-Type: application/json" \
  -d '{
    "price": 15000
  }'
```

**상품 삭제:**
```bash
curl -X DELETE http://localhost:8001/products/1
```

## 3. Python requests 라이브러리

**test_api.py:**
```python
import requests
import json

BASE_URL = "http://localhost:8001"

# 상품 생성
def test_create():
    data = {
        "name": "Python 테스트 상품",
        "description": "requests로 생성",
        "price": 50000,
        "stock": 10
    }
    response = requests.post(f"{BASE_URL}/products/", json=data)
    print(f"생성: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return response.json()['id']

# 상품 조회
def test_get(product_id):
    response = requests.get(f"{BASE_URL}/products/{product_id}")
    print(f"\n조회: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

# 상품 수정
def test_update(product_id):
    data = {"price": 45000, "stock": 20}
    response = requests.patch(f"{BASE_URL}/products/{product_id}", json=data)
    print(f"\n수정: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

# 상품 삭제
def test_delete(product_id):
    response = requests.delete(f"{BASE_URL}/products/{product_id}")
    print(f"\n삭제: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    product_id = test_create()
    test_get(product_id)
    test_update(product_id)
    test_delete(product_id)
```

---

# 🚀 프로젝트 실행 체크리스트

## Django 프로젝트

```bash
# 1. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 패키지 설치
pip install django

# 3. 마이그레이션
python manage.py makemigrations
python manage.py migrate

# 4. 관리자 계정 생성
python manage.py createsuperuser

# 5. 서버 실행
python manage.py runserver

# 접속:
# 웹사이트: http://localhost:8000/products/
# 관리자: http://localhost:8000/admin/
```

## FastAPI 프로젝트

```bash
# 1. 패키지 설치
pip install fastapi uvicorn sqlalchemy

# 2. 샘플 데이터 추가
python add_sample_data.py

# 3. 서버 실행
uvicorn main:app --reload --port 8001

# 접속:
# API 문서: http://localhost:8001/docs
# ReDoc: http://localhost:8001/redoc
# API 엔드포인트: http://localhost:8001/products/
```

---

# 💡 주요 개념 정리

## Django의 핵심 구조

```
요청 → URL → View → Model/Form → Template → 응답
```

1. **Model**: 데이터베이스 구조 정의
2. **View**: 비즈니스 로직 처리
3. **Template**: HTML 렌더링
4. **Form**: 입력 데이터 검증
5. **URL**: 라우팅

## FastAPI의 핵심 구조

```
요청 → 라우터(데코레이터) → Schema 검증 → Model → DB → Schema → 응답
```

1. **Model (SQLAlchemy)**: 데이터베이스 테이블
2. **Schema (Pydantic)**: 입출력 데이터 검증
3. **Router**: 엔드포인트 정의 (데코레이터)
4. **Dependency**: DB 세션 관리

---

# 🔧 트러블슈팅

## 자주 발생하는 오류

### Django

**1. "No such table" 오류**
```bash
python manage.py makemigrations
python manage.py migrate
```

**2. 정적 파일이 로드되지 않음**
```python
# settings.py 확인
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
```

**3. CSRF 토큰 오류**
```html
<!-- 폼에 {% csrf_token %} 추가 -->
<form method="post">
    {% csrf_token %}
    ...
</form>
```

### FastAPI

**1. "Table already exists" 오류**
```python
# models.py에서 테이블 삭제 후 재생성
models.Base.metadata.drop_all(bind=engine)
models.Base.metadata.create_all(bind=engine)
```

**2. Pydantic 검증 오류**
```python
# 스키마의 필드 제약 조건 확인
price: float = Field(..., gt=0)  # 0보다 큰 값만 허용
```

**3. CORS 오류 (프론트엔드 연동 시)**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

# 📝 요약

## Django는 언제 사용?
- 완전한 웹 애플리케이션
- 관리자 페이지가 필요할 때
- 템플릿 렌더링이 필요할 때
- 빠른 프로토타이핑

## FastAPI는 언제 사용?
- REST API만 필요할 때
- 고성능이 중요할 때
- 자동 API 문서가 필요할 때
- 마이크로서비스 아키텍처

## 함께 사용하면?
- Django: 웹 UI + 관리자
- FastAPI: 모바일 앱 API + 외부 연동
- 최고의 조합! 🎉

---