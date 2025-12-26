from django.db import models
from django.utils.text import slugify  
# 문자열을 url-식별자 형태의 안전한 slug 형식으로 변환 (소문자로)
    # slugify("hello word") 문자열은 "hello-word"로 변환
    # slugify("django&DRF guide") --> "django-drf-guide"로 변환

from django.contrib.auth.models import User
# Create your models here.


class Category(models.Model):
    """게시글 카테고리 모델"""
    name = models.CharField(max_length=100, unique=True, verbose_name='카테고리명')
    slug = models.SlugField(max_length=100, unique=True, verbose_name='슬러그')  # url에서 사용하는 식별자
    description = models.TextField(blank=True, verbose_name='설명')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='생성일')
    
    class Meta:
        verbose_name = '카테고리'           # admin 에 표시되는 단수 명칭
        verbose_name_plural = '카테고리'    # admin 에 표시되는 복수 명칭
        ordering = ['name']                 # 기본 정렬 기준
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):   # 핵심! 오버라이딩. 
        if not self.slug:
            self.slug = slugify(self.name, allow_unicode=True) # name을 기반으로 slug를 자동으로 생성하는데 한글 허용
            # '장고 튜토리얼' --> '장고-튜토리얼'로 변환
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
    
    views = models.PositiveIntegerField(default=0, verbose_name='조회수')  # PositiveIntegerField: 음수를 허용하지 않음
    featured = models.BooleanField(default=False, verbose_name='주요 게시글')
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='생성일')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='수정일')
    published_at = models.DateTimeField(null=True, blank=True, verbose_name='발행일')
    
    class Meta:
        verbose_name = '게시글'
        verbose_name_plural = '게시글'
        ordering = ['-created_at']
        indexes = [                                 # 인덱스 속성을 주면, 검색속도를 좀더 빠르게 가능
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
    
    @property      # 메서드를 속성처럼 접근하게 만드는 데코레이터./ ex. post.comment_count() 해야 반환되는데, 변수처럼 사용 / 대표적으로 get과 set이 형식이 있음 
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
