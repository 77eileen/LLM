# Step 3: Serializer 구현

## 목표
- ModelSerializer 사용법 이해
- 중첩된(Nested) Serializer 구현
- 커스텀 필드 및 메서드 추가
- Validation 구현

### 1. Serializer 개념

Serializer는 복잡한 데이터 타입(모델 인스턴스, QuerySet)을 Python 네이티브 데이터 타입으로 변환하여 JSON, XML 등으로 쉽게 렌더링할 수 있게 합니다.

**주요 기능:**
- 데이터 직렬화 (Serialization)
- 데이터 역직렬화 (Deserialization)
- 데이터 검증 (Validation)

### 2. serializers.py 작성

`blog/serializers.py`:

```python
from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Category, Tag, Post, Comment


class UserSerializer(serializers.ModelSerializer):
    """사용자 Serializer"""
    post_count = serializers.SerializerMethodField()
    
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'post_count']
        read_only_fields = ['id']
    
    def get_post_count(self, obj):
        """작성한 게시글 수"""
        return obj.posts.count()


class CategorySerializer(serializers.ModelSerializer):
    """카테고리 Serializer"""
    post_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Category
        fields = ['id', 'name', 'slug', 'description', 'post_count', 'created_at']
        read_only_fields = ['id', 'slug', 'created_at']
    
    def get_post_count(self, obj):
        """카테고리에 속한 게시글 수"""
        return obj.posts.count()


class TagSerializer(serializers.ModelSerializer):
    """태그 Serializer"""
    post_count = serializers.SerializerMethodField()
    
    class Meta:
        model = Tag
        fields = ['id', 'name', 'slug', 'post_count', 'created_at']
        read_only_fields = ['id', 'slug', 'created_at']
    
    def get_post_count(self, obj):
        """태그가 달린 게시글 수"""
        return obj.posts.count()


class CommentSerializer(serializers.ModelSerializer):
    """댓글 Serializer"""
    author = UserSerializer(read_only=True)
    author_id = serializers.IntegerField(write_only=True, required=False)
    replies = serializers.SerializerMethodField()
    
    class Meta:
        model = Comment
        fields = [
            'id', 'post', 'author', 'author_id', 'content', 
            'parent', 'replies', 'is_approved', 
            'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
    
    def get_replies(self, obj):
        """대댓글 목록"""
        if obj.replies.exists():
            return CommentSerializer(obj.replies.all(), many=True).data
        return []
    
    def validate_content(self, value):
        """댓글 내용 검증"""
        if len(value) < 2:
            raise serializers.ValidationError("댓글은 최소 2자 이상이어야 합니다.")
        if len(value) > 500:
            raise serializers.ValidationError("댓글은 최대 500자까지 작성 가능합니다.")
        return value


class PostListSerializer(serializers.ModelSerializer):
    """게시글 목록용 Serializer (간단한 정보)"""
    author = UserSerializer(read_only=True)
    category = CategorySerializer(read_only=True)
    tags = TagSerializer(many=True, read_only=True)
    comment_count = serializers.IntegerField(read_only=True)
    
    class Meta:
        model = Post
        fields = [
            'id', 'title', 'slug', 'excerpt', 'author', 'category', 
            'tags', 'status', 'views', 'comment_count', 'featured',
            'created_at', 'published_at'
        ]
        read_only_fields = ['id', 'slug', 'views', 'created_at']


class PostDetailSerializer(serializers.ModelSerializer):
    """게시글 상세용 Serializer (전체 정보)"""
    author = UserSerializer(read_only=True)
    author_id = serializers.IntegerField(write_only=True, required=False)
    
    category = CategorySerializer(read_only=True)
    category_id = serializers.IntegerField(write_only=True, required=False)
    
    tags = TagSerializer(many=True, read_only=True)
    tag_ids = serializers.ListField(
        child=serializers.IntegerField(),
        write_only=True,
        required=False
    )
    
    comments = CommentSerializer(many=True, read_only=True)
    comment_count = serializers.IntegerField(read_only=True)
    
    class Meta:
        model = Post
        fields = [
            'id', 'title', 'slug', 'content', 'excerpt',
            'author', 'author_id',
            'category', 'category_id',
            'tags', 'tag_ids',
            'comments', 'comment_count',
            'status', 'views', 'featured',
            'created_at', 'updated_at', 'published_at'
        ]
        read_only_fields = ['id', 'slug', 'views', 'created_at', 'updated_at']
    
    def validate_title(self, value):
        """제목 검증"""
        if len(value) < 5:
            raise serializers.ValidationError("제목은 최소 5자 이상이어야 합니다.")
        if len(value) > 200:
            raise serializers.ValidationError("제목은 최대 200자까지 작성 가능합니다.")
        return value
    
    def validate_content(self, value):
        """내용 검증"""
        if len(value) < 10:
            raise serializers.ValidationError("내용은 최소 10자 이상이어야 합니다.")
        return value
    
    def create(self, validated_data):
        """게시글 생성"""
        tag_ids = validated_data.pop('tag_ids', [])
        
        # 작성자가 제공되지 않은 경우 요청 사용자로 설정
        if 'author_id' not in validated_data:
            validated_data['author'] = self.context['request'].user
        else:
            author_id = validated_data.pop('author_id')
            validated_data['author_id'] = author_id
        
        # 카테고리 ID 처리
        if 'category_id' in validated_data:
            category_id = validated_data.pop('category_id')
            validated_data['category_id'] = category_id
        
        post = Post.objects.create(**validated_data)
        
        # 태그 설정
        if tag_ids:
            post.tags.set(tag_ids)
        
        return post
    
    def update(self, instance, validated_data):
        """게시글 수정"""
        tag_ids = validated_data.pop('tag_ids', None)
        
        # 작성자 ID 처리
        if 'author_id' in validated_data:
            author_id = validated_data.pop('author_id')
            validated_data['author_id'] = author_id
        
        # 카테고리 ID 처리
        if 'category_id' in validated_data:
            category_id = validated_data.pop('category_id')
            validated_data['category_id'] = category_id
        
        # 필드 업데이트
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        
        # 태그 업데이트
        if tag_ids is not None:
            instance.tags.set(tag_ids)
        
        return instance


class PostCreateSerializer(serializers.ModelSerializer):
    """게시글 생성용 Serializer"""
    tag_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False,
        allow_empty=True
    )
    
    class Meta:
        model = Post
        fields = [
            'title', 'content', 'excerpt', 'category', 
            'tag_ids', 'status', 'featured', 'published_at'
        ]
    
    def create(self, validated_data):
        tag_ids = validated_data.pop('tag_ids', [])
        validated_data['author'] = self.context['request'].user
        
        post = Post.objects.create(**validated_data)
        
        if tag_ids:
            post.tags.set(tag_ids)
        
        return post


class PostUpdateSerializer(serializers.ModelSerializer):
    """게시글 수정용 Serializer"""
    tag_ids = serializers.ListField(
        child=serializers.IntegerField(),
        required=False,
        allow_empty=True
    )
    
    class Meta:
        model = Post
        fields = [
            'title', 'content', 'excerpt', 'category', 
            'tag_ids', 'status', 'featured', 'published_at'
        ]
    
    def update(self, instance, validated_data):
        tag_ids = validated_data.pop('tag_ids', None)
        
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        
        if tag_ids is not None:
            instance.tags.set(tag_ids)
        
        return instance
```

### 3. Serializer 테스트

Django shell에서 Serializer를 테스트합니다:

```bash
python manage.py shell
```

```python
from blog.models import Post, Category, Tag
from blog.serializers import PostDetailSerializer, CategorySerializer
from django.contrib.auth.models import User

# 데이터 가져오기
post = Post.objects.first()
user = User.objects.first()

# Serializer 사용
serializer = PostDetailSerializer(post, context={'request': None})
print(serializer.data)

# 카테고리 Serializer 테스트
category = Category.objects.first()
cat_serializer = CategorySerializer(category)
print(cat_serializer.data)

# 여러 객체 직렬화
posts = Post.objects.all()[:5]
serializer = PostDetailSerializer(posts, many=True, context={'request': None})
print(serializer.data)
```

## 핵심 개념

### 1. SerializerMethodField
커스텀 필드를 정의할 때 사용:
```python
post_count = serializers.SerializerMethodField()

def get_post_count(self, obj):
    return obj.posts.count()
```

### 2. read_only vs write_only
- `read_only=True`: 읽기 전용 (조회 시에만 포함)
- `write_only=True`: 쓰기 전용 (생성/수정 시에만 사용)

### 3. Nested Serializer
다른 모델의 Serializer를 포함:
```python
author = UserSerializer(read_only=True)
```

### 4. Validation
데이터 검증 메서드:
```python
def validate_title(self, value):
    if len(value) < 5:
        raise serializers.ValidationError("제목은 최소 5자 이상")
    return value
```

## 다음 단계

Step 4에서 다양한 View를 구현하여 실제 API 엔드포인트를 만들어보겠습니다.
