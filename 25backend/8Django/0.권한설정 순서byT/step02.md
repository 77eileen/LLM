# Step 2: 모델 설계

## 목표
- Blog 애플리케이션의 데이터 모델 설계
- 모델 간 관계 설정 (ForeignKey, ManyToMany)
- 모델 메서드 및 속성 정의
- Admin 페이지 커스터마이징

### 1. 모델 구조 설계

우리는 다음과 같은 모델을 설계합니다:

- **Category**: 게시글 카테고리
- **Tag**: 게시글 태그
- **Post**: 게시글
- **Comment**: 댓글

### 2. models.py 작성

`blog/models.py`:

```python
from django.db import models
from django.contrib.auth.models import User
from django.utils.text import slugify


class Category(models.Model):
    """게시글 카테고리 모델"""
    name = models.CharField(max_length=100, unique=True, verbose_name='카테고리명')
    slug = models.SlugField(max_length=100, unique=True, verbose_name='슬러그')
    description = models.TextField(blank=True, verbose_name='설명')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='생성일')
    
    class Meta:
        verbose_name = '카테고리'
        verbose_name_plural = '카테고리'
        ordering = ['name']
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name, allow_unicode=True)
        super().save(*args, **kwargs)


class Tag(models.Model):
    """게시글 태그 모델"""
    name = models.CharField(max_length=50, unique=True, verbose_name='태그명')
    slug = models.SlugField(max_length=50, unique=True, verbose_name='슬러그')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='생성일')
    
    class Meta:
        verbose_name = '태그'
        verbose_name_plural = '태그'
        ordering = ['name']
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name, allow_unicode=True)
        super().save(*args, **kwargs)


class Post(models.Model):
    """게시글 모델"""
    STATUS_CHOICES = [
        ('draft', '임시저장'),
        ('published', '발행됨'),
    ]
    
    title = models.CharField(max_length=200, verbose_name='제목')
    slug = models.SlugField(max_length=200, unique=True, verbose_name='슬러그')
    content = models.TextField(verbose_name='내용')
    excerpt = models.TextField(max_length=500, blank=True, verbose_name='요약')
    
    author = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='posts',
        verbose_name='작성자'
    )
    category = models.ForeignKey(
        Category, 
        on_delete=models.SET_NULL, 
        null=True,
        related_name='posts',
        verbose_name='카테고리'
    )
    tags = models.ManyToManyField(
        Tag, 
        blank=True,
        related_name='posts',
        verbose_name='태그'
    )
    
    status = models.CharField(
        max_length=10, 
        choices=STATUS_CHOICES, 
        default='draft',
        verbose_name='상태'
    )
    
    views = models.PositiveIntegerField(default=0, verbose_name='조회수')
    featured = models.BooleanField(default=False, verbose_name='주요 게시글')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='생성일')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='수정일')
    published_at = models.DateTimeField(null=True, blank=True, verbose_name='발행일')
    
    class Meta:
        verbose_name = '게시글'
        verbose_name_plural = '게시글'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['status']),
            models.Index(fields=['author']),
        ]
    
    def __str__(self):
        return self.title
    
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title, allow_unicode=True)
        super().save(*args, **kwargs)
    
    @property
    def comment_count(self):
        """댓글 수 반환"""
        return self.comments.count()
    
    def increment_views(self):
        """조회수 증가"""
        self.views += 1
        self.save(update_fields=['views'])


class Comment(models.Model):
    """댓글 모델"""
    post = models.ForeignKey(
        Post, 
        on_delete=models.CASCADE, 
        related_name='comments',
        verbose_name='게시글'
    )
    author = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        related_name='comments',
        verbose_name='작성자'
    )
    content = models.TextField(verbose_name='내용')
    
    parent = models.ForeignKey(
        'self', 
        on_delete=models.CASCADE, 
        null=True, 
        blank=True,
        related_name='replies',
        verbose_name='부모 댓글'
    )
    
    is_approved = models.BooleanField(default=True, verbose_name='승인 여부')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='생성일')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='수정일')
    
    class Meta:
        verbose_name = '댓글'
        verbose_name_plural = '댓글'
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['post', 'created_at']),
        ]
    
    def __str__(self):
        return f'{self.author.username}의 댓글: {self.content[:30]}'
```

### 3. admin.py 작성

`blog/admin.py`:

```python
from django.contrib import admin
from .models import Category, Tag, Post, Comment


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug', 'created_at']
    prepopulated_fields = {'slug': ('name',)}
    search_fields = ['name', 'description']


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug', 'created_at']
    prepopulated_fields = {'slug': ('name',)}
    search_fields = ['name']


class CommentInline(admin.TabularInline):
    model = Comment
    extra = 0
    fields = ['author', 'content', 'is_approved', 'created_at']
    readonly_fields = ['created_at']


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ['title', 'author', 'category', 'status', 'views', 'created_at']
    list_filter = ['status', 'category', 'created_at', 'featured']
    search_fields = ['title', 'content']
    prepopulated_fields = {'slug': ('title',)}
    filter_horizontal = ['tags']
    date_hierarchy = 'created_at'
    inlines = [CommentInline]
    
    fieldsets = (
        ('기본 정보', {
            'fields': ('title', 'slug', 'author', 'category', 'tags')
        }),
        ('내용', {
            'fields': ('excerpt', 'content')
        }),
        ('설정', {
            'fields': ('status', 'featured', 'views')
        }),
        ('날짜', {
            'fields': ('published_at',),
            'classes': ('collapse',)
        }),
    )


@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ['post', 'author', 'content_preview', 'is_approved', 'created_at']
    list_filter = ['is_approved', 'created_at']
    search_fields = ['content', 'author__username', 'post__title']
    
    def content_preview(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content
    content_preview.short_description = '내용'
```

### 4. 마이그레이션 생성 및 적용

```bash
# 마이그레이션 파일 생성
python manage.py makemigrations blog

# 마이그레이션 적용
python manage.py migrate blog
```

### 5. 테스트 데이터 생성

Django shell을 사용하여 테스트 데이터를 생성합니다:

```bash
python manage.py shell
```

```python
from django.contrib.auth.models import User
from blog.models import Category, Tag, Post, Comment
from django.utils import timezone

# 사용자 생성
user = User.objects.create_user(username='testuser', password='testpass123')

# 카테고리 생성
django_cat = Category.objects.create(name='Django', description='Django 관련 글')
python_cat = Category.objects.create(name='Python', description='Python 관련 글')

# 태그 생성
drf_tag = Tag.objects.create(name='DRF')
api_tag = Tag.objects.create(name='API')
rest_tag = Tag.objects.create(name='REST')

# 게시글 생성
post = Post.objects.create(
    title='Django REST Framework 시작하기',
    content='DRF를 사용하여 REST API를 만드는 방법',
    excerpt='DRF 입문 가이드',
    author=user,
    category=django_cat,
    status='published',
    published_at=timezone.now()
)
post.tags.add(drf_tag, api_tag, rest_tag)

# 댓글 생성
Comment.objects.create(
    post=post,
    author=user,
    content='좋은 글 감사합니다!'
)
```

### 6. 확인

Admin 페이지에서 생성된 데이터를 확인:
- http://127.0.0.1:8000/admin/

## 다음 단계

Step 3에서 Serializer를 구현하여 모델 데이터를 JSON으로 변환하는 방법을 학습하겠습니다.
