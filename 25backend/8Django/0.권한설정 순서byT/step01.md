# Step 1: 프로젝트 초기 설정

## 목표
- Django 프로젝트 생성
- Django REST Framework 설치 및 설정
- 기본 앱 구조 설계

### 1. 프로젝트 생성

```bash
# 1-1. 프로젝트 디렉토리 생성
mkdir django-api
cd django-api

# 1-2. 가상환경 생성 및 활성화
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 1-3. 필수 패키지 설치
pip install -r requirements.txt

# 1-4. Django 프로젝트 생성
django-admin startproject config .

# 1-5. blog 앱 생성
python manage.py startapp blog
```

### 2. settings.py 설정

`config/settings.py` 파일을 열어 다음과 같이 수정합니다:

```python
# INSTALLED_APPS에 추가
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third-party apps
    'rest_framework',
    'rest_framework.authtoken',
    'drf_spectacular',
    'django_filters',
    
    # Local apps
    'blog',
]

# REST Framework 설정
REST_FRAMEWORK = {
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 10,
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],
}

# drf-spectacular 설정
SPECTACULAR_SETTINGS = {
    'TITLE': 'Blog API',
    'DESCRIPTION': 'Django REST Framework를 이용한 Blog API',
    'VERSION': '1.0.0',
    'SERVE_INCLUDE_SCHEMA': False,
}
```

### 3. URL 설정

`config/urls.py`:

```python
from django.contrib import admin
from django.urls import path, include
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularSwaggerView,
    SpectacularRedocView,
)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('blog.urls')),
    
    # API 스키마 및 문서
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/schema/swagger-ui/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/schema/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
]
```

### 4. 데이터베이스 마이그레이션

```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. 슈퍼유저 생성

```bash
python manage.py createsuperuser
```

### 6. 개발 서버 실행

```bash
python manage.py runserver
```

### 7. 확인

브라우저에서 다음 URL을 방문하여 설정이 정상적으로 되었는지 확인:

- Admin: http://127.0.0.1:8000/admin/
- Swagger UI: http://127.0.0.1:8000/api/schema/swagger-ui/
- Redoc: http://127.0.0.1:8000/api/schema/redoc/

## 다음 단계

Step 2에서 모델을 설계하고 데이터베이스 스키마를 정의하겠습니다.
