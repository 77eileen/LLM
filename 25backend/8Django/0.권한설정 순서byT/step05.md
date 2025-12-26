# Step 5: Router 및 URL 설정

##  학습 목표

- DefaultRouter를 사용한 자동 URL 생성
- ViewSet과 Router 연결
- 커스텀 액션의 URL 패턴 이해
- URL 네임스페이스 활용

---

##  Router란?

Router는 ViewSet의 액션들을 자동으로 URL 패턴으로 변환해주는 도구입니다.

---

##  실습 1: DefaultRouter 설정

### blog/urls.py

```python
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

app_name = 'blog'

# Router 생성
router = DefaultRouter()

# ViewSet 등록
router.register(r'posts', views.PostViewSet, basename='post')
router.register(r'comments', views.CommentViewSet, basename='comment')
router.register(r'categories', views.CategoryViewSet, basename='category')
router.register(r'tags', views.TagViewSet, basename='tag')

urlpatterns = [
    path('', include(router.urls)),
]
```

---

##  자동 생성되는 URL 패턴

### PostViewSet의 경우:

| HTTP Method | URL | Action | Name |
|-------------|-----|--------|------|
| GET | /posts/ | list | post-list |
| POST | /posts/ | create | post-list |
| GET | /posts/{pk}/ | retrieve | post-detail |
| PUT | /posts/{pk}/ | update | post-detail |
| PATCH | /posts/{pk}/ | partial_update | post-detail |
| DELETE | /posts/{pk}/ | destroy | post-detail |

### 커스텀 액션의 URL:

| HTTP Method | URL | Action |
|-------------|-----|--------|
| POST | /posts/{pk}/publish/ | publish |
| POST | /posts/{pk}/unpublish/ | unpublish |
| GET | /posts/my_posts/ | my_posts |
| GET | /posts/drafts/ | drafts |
| GET | /posts/{pk}/comments/ | comments |

---

##  실습 2: 프로젝트 URL 설정

### config/urls.py

```python
from django.contrib import admin
from django.urls import path, include
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularSwaggerView,
    SpectacularRedocView,
)

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # API Schema
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/schema/swagger-ui/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/schema/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
    
    # API v1
    path('api/v1/', include('blog.urls')),
]
```

---

##  실습 3: 전체 API 엔드포인트 확인

서버를 실행하고 Swagger UI에서 확인:

```bash
python manage.py runserver
```

브라우저에서 접속:
- http://127.0.0.1:8000/api/schema/swagger-ui/

### 생성된 엔드포인트 목록:

```
# 카테고리
GET    /api/v1/categories/
GET    /api/v1/categories/{id}/
GET    /api/v1/categories/{id}/posts/

# 태그
GET    /api/v1/tags/
GET    /api/v1/tags/{id}/
GET    /api/v1/tags/{id}/posts/

# 게시글
GET    /api/v1/posts/                    # 목록
POST   /api/v1/posts/                    # 생성
GET    /api/v1/posts/{id}/               # 상세
PUT    /api/v1/posts/{id}/               # 전체 수정
PATCH  /api/v1/posts/{id}/               # 부분 수정
DELETE /api/v1/posts/{id}/               # 삭제
POST   /api/v1/posts/{id}/publish/       # 발행
POST   /api/v1/posts/{id}/unpublish/     # 비공개
GET    /api/v1/posts/{id}/comments/      # 게시글의 댓글
GET    /api/v1/posts/my_posts/           # 내 게시글
GET    /api/v1/posts/drafts/             # 임시저장

# 댓글
GET    /api/v1/comments/                 # 목록
POST   /api/v1/comments/                 # 생성
GET    /api/v1/comments/{id}/            # 상세
PUT    /api/v1/comments/{id}/            # 전체 수정
PATCH  /api/v1/comments/{id}/            # 부분 수정
DELETE /api/v1/comments/{id}/            # 삭제
```

---

##  실습 4: @action 데코레이터 상세

### detail=True (객체별 액션)

```python
@action(detail=True, methods=['post'])
def publish(self, request, pk=None):
    """
    URL: /api/v1/posts/{pk}/publish/
    특정 게시글을 발행
    """
    post = self.get_object()
    # ...
```

### detail=False (리스트 액션)

```python
@action(detail=False, methods=['get'])
def my_posts(self, request):
    """
    URL: /api/v1/posts/my_posts/
    내가 작성한 모든 게시글
    """
    # ...
```

### url_path 커스터마이징

```python
@action(detail=False, methods=['get'], url_path='my-drafts')
def my_drafts(self, request):
    """
    URL: /api/v1/posts/my-drafts/
    함수명과 다른 URL 사용
    """
    # ...
```

---

##  실습 5: SimpleRouter vs DefaultRouter

### SimpleRouter
기본적인 URL만 생성

### DefaultRouter
API Root View 추가 제공

```python
# DefaultRouter를 사용하면 이런 화면이 추가됨
GET /api/v1/

{
    "posts": "http://127.0.0.1:8000/api/v1/posts/",
    "comments": "http://127.0.0.1:8000/api/v1/comments/",
    "categories": "http://127.0.0.1:8000/api/v1/categories/",
    "tags": "http://127.0.0.1:8000/api/v1/tags/"
}
```

---

##  실습 6: 여러 Router 결합

### blog/urls.py

```python
from rest_framework.routers import DefaultRouter
from . import views

# 메인 Router
router = DefaultRouter()
router.register(r'posts', views.PostViewSet, basename='post')
router.register(r'comments', views.CommentViewSet, basename='comment')

# 읽기 전용 Router
readonly_router = DefaultRouter()
readonly_router.register(r'categories', views.CategoryViewSet, basename='category')
readonly_router.register(r'tags', views.TagViewSet, basename='tag')

urlpatterns = [
    path('', include(router.urls)),
    path('readonly/', include(readonly_router.urls)),
]
```

---

##  실습 7: URL Reverse (URL 역참조)

### View에서 URL 생성

```python
from django.urls import reverse
from rest_framework.reverse import reverse as api_reverse

# 템플릿에서
url = reverse('blog:post-detail', kwargs={'pk': 1})
# /api/v1/posts/1/

# DRF에서
url = api_reverse('blog:post-detail', kwargs={'pk': 1}, request=request)
# http://127.0.0.1:8000/api/v1/posts/1/
```


##  다음 단계

**Step 6: 인증 및 권한**에서 토큰 기반 인증과 권한 관리를 구현합니다.
