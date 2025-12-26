# Step 4: API View 구현

##  학습 목표

- Function-based Views와 Class-based Views 이해
- Generic Views와 ViewSet의 차이점 학습
- ModelViewSet으로 전체 CRUD 구현
- 커스텀 액션 추가하기

---

##  View의 종류

### 1. Function-based Views (@api_view)
가장 간단한 형태의 View입니다.

### 2. APIView
클래스 기반의 기본 View입니다.

### 3. Generic Views
반복되는 패턴을 자동화한 View입니다.

### 4. ViewSet
관련된 여러 View를 하나로 묶은 View입니다.

---

##  실습 1: Function-based Views

### blog/views.py

```python
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Post
from .serializers import PostSerializer

@api_view(['GET', 'POST'])
def post_list(request):
    """
    GET: 게시글 목록 조회
    POST: 게시글 생성
    """
    if request.method == 'GET':
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        serializer = PostSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(author=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'PUT', 'DELETE'])
def post_detail(request, pk):
    """
    GET: 게시글 상세 조회
    PUT: 게시글 수정
    DELETE: 게시글 삭제
    """
    try:
        post = Post.objects.get(pk=pk)
    except Post.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    if request.method == 'GET':
        serializer = PostSerializer(post)
        return Response(serializer.data)
    
    elif request.method == 'PUT':
        serializer = PostSerializer(post, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        post.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
```

---

##  실습 2: APIView (Class-based)

### blog/views.py

```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import Http404
from .models import Post
from .serializers import PostSerializer

class PostListAPIView(APIView):
    """
    게시글 목록 조회 및 생성
    """
    def get(self, request):
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        serializer = PostSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(author=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PostDetailAPIView(APIView):
    """
    게시글 상세 조회, 수정, 삭제
    """
    def get_object(self, pk):
        try:
            return Post.objects.get(pk=pk)
        except Post.DoesNotExist:
            raise Http404
    
    def get(self, request, pk):
        post = self.get_object(pk)
        serializer = PostSerializer(post)
        return Response(serializer.data)
    
    def put(self, request, pk):
        post = self.get_object(pk)
        serializer = PostSerializer(post, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def delete(self, request, pk):
        post = self.get_object(pk)
        post.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
```

---

##  실습 3: Generic Views

### blog/views.py

```python
from rest_framework import generics
from .models import Post, Comment
from .serializers import PostSerializer, CommentSerializer

class PostListCreateView(generics.ListCreateAPIView):
    """
    게시글 목록 조회 및 생성
    """
    queryset = Post.objects.all()
    serializer_class = PostSerializer
    
    def perform_create(self, serializer):
        serializer.save(author=self.request.user)


class PostRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    """
    게시글 상세 조회, 수정, 삭제
    """
    queryset = Post.objects.all()
    serializer_class = PostSerializer


class CommentListCreateView(generics.ListCreateAPIView):
    """
    댓글 목록 조회 및 생성
    """
    serializer_class = CommentSerializer
    
    def get_queryset(self):
        post_id = self.kwargs['post_pk']
        return Comment.objects.filter(post_id=post_id)
    
    def perform_create(self, serializer):
        post_id = self.kwargs['post_pk']
        serializer.save(
            author=self.request.user,
            post_id=post_id
        )
```

---

##  실습 4: ViewSet과 ModelViewSet

### blog/views.py (최종 권장 방식)

```python
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.db.models import Q
from .models import Post, Comment, Category, Tag
from .serializers import (
    PostListSerializer,
    PostDetailSerializer,
    CommentSerializer,
    CategorySerializer,
    TagSerializer
)

class CategoryViewSet(viewsets.ReadOnlyModelViewSet):
    """
    카테고리 ViewSet (읽기 전용)
    """
    queryset = Category.objects.all()
    serializer_class = CategorySerializer
    
    @action(detail=True, methods=['get'])
    def posts(self, request, pk=None):
        """특정 카테고리의 게시글 목록"""
        category = self.get_object()
        posts = Post.objects.filter(category=category, status='published')
        serializer = PostListSerializer(posts, many=True)
        return Response(serializer.data)


class TagViewSet(viewsets.ReadOnlyModelViewSet):
    """
    태그 ViewSet (읽기 전용)
    """
    queryset = Tag.objects.all()
    serializer_class = TagSerializer
    
    @action(detail=True, methods=['get'])
    def posts(self, request, pk=None):
        """특정 태그의 게시글 목록"""
        tag = self.get_object()
        posts = tag.posts.filter(status='published')
        serializer = PostListSerializer(posts, many=True)
        return Response(serializer.data)


class PostViewSet(viewsets.ModelViewSet):
    """
    게시글 ViewSet (전체 CRUD)
    """
    queryset = Post.objects.select_related('author', 'category').prefetch_related('tags')
    
    def get_serializer_class(self):
        if self.action == 'list':
            return PostListSerializer
        return PostDetailSerializer
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # 발행된 게시글만 표시 (작성자는 본인 글 모두 볼 수 있음)
        if self.request.user.is_authenticated:
            queryset = queryset.filter(
                Q(status='published') | Q(author=self.request.user)
            )
        else:
            queryset = queryset.filter(status='published')
        
        return queryset.order_by('-created_at')
    
    def perform_create(self, serializer):
        """게시글 생성 시 작성자 자동 설정"""
        serializer.save(author=self.request.user)
    
    @action(detail=True, methods=['post'])
    def publish(self, request, pk=None):
        """게시글 발행"""
        post = self.get_object()
        if post.author != request.user:
            return Response(
                {'error': '권한이 없습니다.'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        post.status = 'published'
        post.save()
        return Response({'status': '게시글이 발행되었습니다.'})
    
    @action(detail=True, methods=['post'])
    def unpublish(self, request, pk=None):
        """게시글 비공개"""
        post = self.get_object()
        if post.author != request.user:
            return Response(
                {'error': '권한이 없습니다.'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        post.status = 'draft'
        post.save()
        return Response({'status': '게시글이 비공개되었습니다.'})
    
    @action(detail=False, methods=['get'])
    def my_posts(self, request):
        """내가 작성한 게시글 목록"""
        posts = self.queryset.filter(author=request.user)
        serializer = self.get_serializer(posts, many=True)
        return Response(serializer.data)
    
    @action(detail=False, methods=['get'])
    def drafts(self, request):
        """임시저장 게시글 목록"""
        posts = self.queryset.filter(author=request.user, status='draft')
        serializer = self.get_serializer(posts, many=True)
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def comments(self, request, pk=None):
        """특정 게시글의 댓글 목록"""
        post = self.get_object()
        comments = post.comments.all()
        serializer = CommentSerializer(comments, many=True)
        return Response(serializer.data)


class CommentViewSet(viewsets.ModelViewSet):
    """
    댓글 ViewSet (전체 CRUD)
    """
    queryset = Comment.objects.select_related('author', 'post')
    serializer_class = CommentSerializer
    
    def get_queryset(self):
        queryset = super().get_queryset()
        
        # 쿼리 파라미터로 필터링
        post_id = self.request.query_params.get('post', None)
        if post_id:
            queryset = queryset.filter(post_id=post_id)
        
        return queryset.order_by('created_at')
    
    def perform_create(self, serializer):
        """댓글 생성 시 작성자 자동 설정"""
        serializer.save(author=self.request.user)
```

---

##  URL 설정 (미리보기)

### blog/urls.py

```python
from django.urls import path
from . import views

app_name = 'blog'

# Function-based Views 방식
urlpatterns_fbv = [
    path('posts/', views.post_list, name='post-list'),
    path('posts/<int:pk>/', views.post_detail, name='post-detail'),
]

# Class-based Views 방식
urlpatterns_cbv = [
    path('posts/', views.PostListAPIView.as_view(), name='post-list'),
    path('posts/<int:pk>/', views.PostDetailAPIView.as_view(), name='post-detail'),
]

# Generic Views 방식
urlpatterns_generic = [
    path('posts/', views.PostListCreateView.as_view(), name='post-list'),
    path('posts/<int:pk>/', views.PostRetrieveUpdateDestroyView.as_view(), name='post-detail'),
    path('posts/<int:post_pk>/comments/', views.CommentListCreateView.as_view(), name='comment-list'),
]

# ViewSet은 Router를 사용하므로 다음 단계에서 설정
```

---

##  각 방식의 장단점

### Function-based Views
**장점:**
- 간단하고 직관적
- 빠르게 작성 가능

**단점:**
- 코드 재사용이 어려움
- 복잡한 로직에서 코드가 길어짐

### APIView
**장점:**
- HTTP 메서드별로 명확하게 분리
- 유연한 커스터마이징

**단점:**
- 반복되는 코드 작성 필요

### Generic Views
**장점:**
- 공통 패턴 자동화
- 코드량 감소

**단점:**
- 제한적인 커스터마이징

### ViewSet
**장점:**
- 관련 로직을 하나로 묶음
- Router와 함께 사용하면 URL 자동 생성
- 가장 강력하고 유연함

**단점:**
- 초기 학습 곡선



##  다음 단계

**Step 5: Router 및 URL 설정**에서 ViewSet과 Router를 연결하여 자동으로 URL을 생성하는 방법을 배웁니다.

