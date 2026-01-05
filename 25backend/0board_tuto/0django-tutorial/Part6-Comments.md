# Part 6: 댓글 기능

## 📌 이번 Part에서 배울 내용

- 댓글 모델 생성
- 댓글 작성 기능
- 댓글 수정 및 삭제
- 댓글 개수 표시
- 대댓글 (선택사항)

## 1. Comment 모델 생성

**board/models.py**에 Comment 모델을 추가하세요:

```python
# board/models.py

from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse


class Post(models.Model):
    """게시글 모델"""
    title = models.CharField(max_length=200, verbose_name='제목')
    content = models.TextField(verbose_name='내용')
    author = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='작성자')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='작성일시')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='수정일시')
    views = models.PositiveIntegerField(default=0, verbose_name='조회수')

    class Meta:
        ordering = ['-created_at']
        verbose_name = '게시글'
        verbose_name_plural = '게시글'

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        """게시글의 상세 페이지 URL을 반환"""
        return reverse('board:detail', kwargs={'pk': self.pk})

    def increase_views(self):
        """조회수 1 증가"""
        self.views += 1
        self.save(update_fields=['views'])

    def is_updated(self):
        """수정 여부 확인"""
        return (self.updated_at - self.created_at).total_seconds() > 60


class Comment(models.Model):
    """댓글 모델"""
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments', verbose_name='게시글')
    author = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='작성자')
    content = models.TextField(verbose_name='내용')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='작성일시')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='수정일시')

    class Meta:
        ordering = ['created_at']  # 댓글은 오래된 순서로
        verbose_name = '댓글'
        verbose_name_plural = '댓글'

    def __str__(self):
        return f'{self.post.title}의 댓글'

    def is_updated(self):
        """수정 여부 확인"""
        return (self.updated_at - self.created_at).total_seconds() > 60
```

### 코드 설명

- **ForeignKey(Post)**: 어떤 게시글의 댓글인지 연결
- **related_name='comments'**: `post.comments.all()`로 게시글의 모든 댓글 조회 가능
- **ordering = ['created_at']**: 댓글은 오래된 것부터 표시 (게시글과 반대)

## 2. 마이그레이션

```bash
python manage.py makemigrations
python manage.py migrate
```

## 3. Admin 페이지에 Comment 등록

**board/admin.py**를 수정하세요:

```python
# board/admin.py

from django.contrib import admin
from .models import Post, Comment


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    """게시글 관리자 페이지 설정"""
    list_display = ['title', 'author', 'created_at', 'views']
    list_filter = ['created_at', 'author']
    search_fields = ['title', 'content']
    readonly_fields = ['created_at', 'updated_at', 'views']
    date_hierarchy = 'created_at'


@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    """댓글 관리자 페이지 설정"""
    list_display = ['post', 'author', 'content_preview', 'created_at']
    list_filter = ['created_at', 'author']
    search_fields = ['content']
    readonly_fields = ['created_at', 'updated_at']

    def content_preview(self, obj):
        """댓글 내용 미리보기 (처음 50자만)"""
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content

    content_preview.short_description = '내용 미리보기'
```

## 4. 댓글 폼 만들기

**board/forms.py**에 CommentForm을 추가하세요:

```python
# board/forms.py

from django import forms
from .models import Post, Comment


class PostForm(forms.ModelForm):
    """게시글 작성/수정 폼"""
    class Meta:
        model = Post
        fields = ['title', 'content']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '제목을 입력하세요',
                'maxlength': '200',
            }),
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': '내용을 입력하세요',
                'rows': 10,
            }),
        }
        labels = {
            'title': '제목',
            'content': '내용',
        }

    def clean_title(self):
        title = self.cleaned_data.get('title')
        if len(title) < 2:
            raise forms.ValidationError('제목은 2글자 이상 입력해주세요.')
        return title

    def clean_content(self):
        content = self.cleaned_data.get('content')
        if len(content) < 10:
            raise forms.ValidationError('내용은 10글자 이상 입력해주세요.')
        return content


class CommentForm(forms.ModelForm):
    """댓글 작성/수정 폼"""
    class Meta:
        model = Comment
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': '댓글을 입력하세요',
                'rows': 3,
            }),
        }
        labels = {
            'content': '',
        }

    def clean_content(self):
        """댓글 내용 유효성 검사"""
        content = self.cleaned_data.get('content')
        if len(content) < 2:
            raise forms.ValidationError('댓글은 2글자 이상 입력해주세요.')
        return content
```

## 5. 댓글 뷰 만들기

**board/views.py**에 댓글 관련 뷰를 추가하세요:

```python
# board/views.py

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Post, Comment
from .forms import PostForm, CommentForm


def post_list(request):
    """게시글 목록 페이지"""
    posts = Post.objects.all()
    context = {
        'posts': posts,
    }
    return render(request, 'board/post_list.html', context)


def post_detail(request, pk):
    """게시글 상세 페이지"""
    post = get_object_or_404(Post, pk=pk)
    post.increase_views()

    # 댓글 폼
    comment_form = CommentForm() # 추가

    context = {
        'post': post,
        'comment_form': comment_form, # 추가
    }
    return render(request, 'board/post_detail.html', context)


@login_required
def post_create(request):
    """게시글 작성 페이지"""
    if request.method == 'POST':
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.save()
            messages.success(request, '게시글이 성공적으로 작성되었습니다.')
            return redirect('board:detail', pk=post.pk)
    else:
        form = PostForm()

    context = {
        'form': form,
    }
    return render(request, 'board/post_form.html', context)


@login_required
def post_update(request, pk):
    """게시글 수정 페이지"""
    post = get_object_or_404(Post, pk=pk)

    if post.author != request.user:
        messages.error(request, '수정 권한이 없습니다.')
        return redirect('board:detail', pk=post.pk)

    if request.method == 'POST':
        form = PostForm(request.POST, instance=post)
        if form.is_valid():
            form.save()
            messages.success(request, '게시글이 수정되었습니다.')
            return redirect('board:detail', pk=post.pk)
    else:
        form = PostForm(instance=post)

    context = {
        'form': form,
        'post': post,
    }
    return render(request, 'board/post_form.html', context)


@login_required
def post_delete(request, pk):
    """게시글 삭제"""
    post = get_object_or_404(Post, pk=pk)

    if post.author != request.user:
        messages.error(request, '삭제 권한이 없습니다.')
        return redirect('board:detail', pk=post.pk)

    if request.method == 'POST':
        post.delete()
        messages.success(request, '게시글이 삭제되었습니다.')
        return redirect('board:list')

    context = {
        'post': post,
    }
    return render(request, 'board/post_confirm_delete.html', context)


@login_required
def comment_create(request, post_pk):
    """댓글 작성"""
    post = get_object_or_404(Post, pk=post_pk)

    if request.method == 'POST':
        form = CommentForm(request.POST)
        if form.is_valid():
            comment = form.save(commit=False)
            comment.post = post
            comment.author = request.user
            comment.save()
            messages.success(request, '댓글이 작성되었습니다.')
        else:
            messages.error(request, '댓글 작성에 실패했습니다.')

    return redirect('board:detail', pk=post_pk)


@login_required
def comment_update(request, pk):
    """댓글 수정"""
    comment = get_object_or_404(Comment, pk=pk)

    if comment.author != request.user:
        messages.error(request, '수정 권한이 없습니다.')
        return redirect('board:detail', pk=comment.post.pk)

    if request.method == 'POST':
        form = CommentForm(request.POST, instance=comment)
        if form.is_valid():
            form.save()
            messages.success(request, '댓글이 수정되었습니다.')
        else:
            messages.error(request, '댓글 수정에 실패했습니다.')

    return redirect('board:detail', pk=comment.post.pk)


@login_required
def comment_delete(request, pk):
    """댓글 삭제"""
    comment = get_object_or_404(Comment, pk=pk)
    post_pk = comment.post.pk

    if comment.author != request.user:
        messages.error(request, '삭제 권한이 없습니다.')
        return redirect('board:detail', pk=post_pk)

    if request.method == 'POST':
        comment.delete()
        messages.success(request, '댓글이 삭제되었습니다.')

    return redirect('board:detail', pk=post_pk)
```

## 6. URL 패턴 추가

**board/urls.py**를 수정하세요:

```python
# board/urls.py

from django.urls import path
from . import views

app_name = 'board'

urlpatterns = [
    # 게시글
    path('', views.post_list, name='list'),
    path('post/<int:pk>/', views.post_detail, name='detail'),
    path('post/create/', views.post_create, name='create'),
    path('post/<int:pk>/update/', views.post_update, name='update'),
    path('post/<int:pk>/delete/', views.post_delete, name='delete'),

    # 댓글
    path('post/<int:post_pk>/comment/create/', views.comment_create, name='comment_create'),
    path('comment/<int:pk>/update/', views.comment_update, name='comment_update'),
    path('comment/<int:pk>/delete/', views.comment_delete, name='comment_delete'),
]
```

## 7. 게시글 상세 템플릿에 댓글 추가

**board/templates/board/post_detail.html**을 수정하세요:

```html
<!-- board/templates/board/post_detail.html -->

{% extends 'board/base.html' %}

{% block title %}{{ post.title }} - Django 게시판{% endblock %}

{% block content %}
<article>
    <header style="border-bottom: 2px solid #ecf0f1; padding-bottom: 20px; margin-bottom: 30px;">
        <h2 style="margin-bottom: 15px;">{{ post.title }}</h2>

        <div style="display: flex; justify-content: space-between; color: #7f8c8d; font-size: 0.9em;">
            <div>
                <span>작성자: <strong>{{ post.author.username }}</strong></span>
                <span style="margin-left: 20px;">조회수: {{ post.views }}</span>
                <span style="margin-left: 20px;">댓글: {{ post.comments.count }}</span>
            </div>
            <div>
                <span>작성일: {{ post.created_at|date:"Y년 m월 d일 H:i" }}</span>
                {% if post.is_updated %}
                    <span style="margin-left: 10px; color: #e74c3c;">(수정됨: {{ post.updated_at|date:"Y-m-d H:i" }})</span>
                {% endif %}
            </div>
        </div>
    </header>

    <div style="min-height: 300px; padding: 20px 0; line-height: 1.8;">
        {{ post.content|linebreaks }}
    </div>

    <footer style="border-top: 1px solid #ecf0f1; padding-top: 20px; margin-top: 30px;">
        <div style="display: flex; gap: 10px;">
            <a href="{% url 'board:list' %}" style="background-color: #95a5a6; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block;">
                목록으로
            </a>
            {% if user == post.author %}
                <a href="{% url 'board:update' post.pk %}" style="background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block;">
                    수정
                </a>
                <a href="{% url 'board:delete' post.pk %}" style="background-color: #e74c3c; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block;" onclick="return confirm('정말 삭제하시겠습니까?');">
                    삭제
                </a>
            {% endif %}
        </div>
    </footer>
</article>

<!-- 댓글 섹션 -->
<section style="margin-top: 50px;">
    <h3 style="margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #ecf0f1;">
        댓글 <span style="color: #3498db;">({{ post.comments.count }})</span>
    </h3>

    <!-- 댓글 작성 폼 -->
    {% if user.is_authenticated %}
        <form method="post" action="{% url 'board:comment_create' post.pk %}" style="margin-bottom: 30px;">
            {% csrf_token %}
            <div style="margin-bottom: 10px;">
                {{ comment_form.content }}
                {% if comment_form.content.errors %}
                    <div style="color: #e74c3c; font-size: 0.9em; margin-top: 5px;">
                        {{ comment_form.content.errors }}
                    </div>
                {% endif %}
            </div>
            <button type="submit" style="background-color: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer;">
                댓글 작성
            </button>
        </form>
    {% else %}
        <p style="text-align: center; padding: 20px; background-color: #ecf0f1; border-radius: 4px; color: #7f8c8d;">
            댓글을 작성하려면 <a href="{% url 'accounts:login' %}?next={{ request.path }}" style="color: #3498db;">로그인</a>해주세요.
        </p>
    {% endif %}

    <!-- 댓글 목록 -->
    {% if post.comments.exists %}
        <div style="margin-top: 30px;">
            {% for comment in post.comments.all %}
                <div style="border-bottom: 1px solid #ecf0f1; padding: 20px 0;" id="comment-{{ comment.pk }}">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <div>
                            <strong>{{ comment.author.username }}</strong>
                            <span style="color: #7f8c8d; font-size: 0.9em; margin-left: 10px;">
                                {{ comment.created_at|date:"Y-m-d H:i" }}
                                {% if comment.is_updated %}
                                    <span style="color: #e74c3c;">(수정됨)</span>
                                {% endif %}
                            </span>
                        </div>
                        {% if user == comment.author %}
                            <div style="display: flex; gap: 5px;">
                                <button onclick="editComment({{ comment.pk }}, '{{ comment.content|escapejs }}')" style="background-color: #3498db; color: white; padding: 5px 10px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9em;">
                                    수정
                                </button>
                                <form method="post" action="{% url 'board:comment_delete' comment.pk %}" style="display: inline;" onsubmit="return confirm('댓글을 삭제하시겠습니까?');">
                                    {% csrf_token %}
                                    <button type="submit" style="background-color: #e74c3c; color: white; padding: 5px 10px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.9em;">
                                        삭제
                                    </button>
                                </form>
                            </div>
                        {% endif %}
                    </div>

                    <!-- 댓글 내용 표시 -->
                    <div class="comment-content" id="content-{{ comment.pk }}">
                        {{ comment.content|linebreaks }}
                    </div>

                    <!-- 댓글 수정 폼 (숨겨진 상태) -->
                    <form method="post" action="{% url 'board:comment_update' comment.pk %}" id="edit-form-{{ comment.pk }}" style="display: none;">
                        {% csrf_token %}
                        <textarea name="content" rows="3" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px;">{{ comment.content }}</textarea>
                        <div style="display: flex; gap: 5px;">
                            <button type="submit" style="background-color: #3498db; color: white; padding: 8px 15px; border: none; border-radius: 4px; cursor: pointer;">
                                저장
                            </button>
                            <button type="button" onclick="cancelEdit({{ comment.pk }})" style="background-color: #95a5a6; color: white; padding: 8px 15px; border: none; border-radius: 4px; cursor: pointer;">
                                취소
                            </button>
                        </div>
                    </form>
                </div>
            {% endfor %}
        </div>
    {% else %}
        <p style="text-align: center; padding: 50px; color: #7f8c8d;">
            아직 댓글이 없습니다. 첫 번째 댓글을 작성해보세요!
        </p>
    {% endif %}
</section>

<script>
    function editComment(commentId, content) {
        // 내용 숨기기
        document.getElementById('content-' + commentId).style.display = 'none';

        // 수정 폼 표시
        document.getElementById('edit-form-' + commentId).style.display = 'block';
    }

    function cancelEdit(commentId) {
        // 내용 다시 표시
        document.getElementById('content-' + commentId).style.display = 'block';

        // 수정 폼 숨기기
        document.getElementById('edit-form-' + commentId).style.display = 'none';
    }
</script>

<style>
    .form-control {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 1em;
        font-family: inherit;
        resize: vertical;
    }

    .form-control:focus {
        outline: none;
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
    }
</style>
{% endblock %}
```

### 템플릿 설명

#### 댓글 개수 표시

```django
댓글 <span>({{ post.comments.count }})</span>
```

`post.comments`는 Comment 모델에서 `related_name='comments'`로 설정했기 때문에 사용 가능합니다.

#### 댓글 작성 폼

- 로그인한 사용자만 댓글 작성 가능
- `action="{% url 'board:comment_create' post.pk %}"`: 댓글 작성 URL

#### 댓글 수정

- JavaScript를 사용하여 인라인 수정 기능 구현
- `editComment()`: 수정 폼 표시
- `cancelEdit()`: 수정 취소

#### escapejs 필터

```django
'{{ comment.content|escapejs }}'
```

JavaScript에서 안전하게 사용할 수 있도록 특수 문자를 이스케이프합니다.

## 8. 게시글 목록에 댓글 개수 표시

**board/templates/board/post_list.html**을 수정하세요:

```html
<!-- 제목 부분만 수정 -->
<td style="padding: 12px;">
    <a href="{% url 'board:detail' post.pk %}" style="color: #2c3e50; text-decoration: none;">
        {{ post.title }}
        {% if post.comments.count > 0 %}
            <span style="color: #3498db; font-size: 0.9em;">[{{ post.comments.count }}]</span>
        {% endif %}
        {% if post.is_updated %}
            <span style="color: #e74c3c; font-size: 0.8em;">[수정됨]</span>
        {% endif %}
    </a>
</td>
```

## 9. 기능 테스트하기

### 1) 댓글 작성

1. 게시글 상세 페이지로 이동
2. 댓글 작성란에 내용 입력
3. "댓글 작성" 버튼 클릭
4. 댓글이 표시되는지 확인

### 2) 댓글 수정

1. 본인이 작성한 댓글의 "수정" 버튼 클릭
2. 내용 수정 후 "저장" 클릭
3. 수정된 내용 확인

### 3) 댓글 삭제

1. 본인이 작성한 댓글의 "삭제" 버튼 클릭
2. 확인 대화상자에서 확인
3. 댓글 삭제 확인

### 4) 댓글 개수 확인

1. 목록 페이지에서 댓글 개수 표시 확인
2. 게시글 상세 페이지에서 댓글 개수 확인

## 📝 Part 6 정리

축하합니다! Part 6을 완료했습니다. 다음을 배웠습니다:

✅ 댓글 모델 생성 및 관계 설정<br>
✅ 댓글 작성 기능<br>
✅ 댓글 수정 및 삭제<br>
✅ 댓글 개수 표시<br>
✅ JavaScript를 활용한 인라인 수정<br>
✅ related_name 활용

## 🔍 자주 발생하는 오류

### 1. RelatedObjectDoesNotExist

**원인**: related_name이 잘못 설정되었습니다.

**해결**:
```python
# models.py
post = models.ForeignKey(Post, related_name='comments', ...)
```

### 2. 댓글이 표시되지 않음

**원인**: 댓글 ordering이 잘못되었거나 템플릿 오류

**해결**:
```python
# models.py
class Meta:
    ordering = ['created_at']
```

### 3. 댓글 수정 시 내용이 사라짐

**원인**: form에 instance를 전달하지 않았습니다.

**해결**:
```python
form = CommentForm(request.POST, instance=comment)
```

## 💡 추가 기능 아이디어

### 1. 대댓글 기능

**board/models.py**에 parent 필드 추가:

```python
class Comment(models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='replies')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

### 2. 댓글 좋아요 기능

```python
class Comment(models.Model):
    # 기존 필드들...
    likes = models.ManyToManyField(User, related_name='liked_comments', blank=True)

    def total_likes(self):
        return self.likes.count()
```

### 3. 댓글 페이지네이션

```python
# views.py
from django.core.paginator import Paginator

def post_detail(request, pk):
    post = get_object_or_404(Post, pk=pk)
    post.increase_views()

    comments = post.comments.all()
    paginator = Paginator(comments, 10)  # 페이지당 10개
    page = request.GET.get('page')
    comments = paginator.get_page(page)

    context = {
        'post': post,
        'comments': comments,
        'comment_form': CommentForm(),
    }
    return render(request, 'board/post_detail.html', context)
```

### 4. 실시간 댓글 개수 업데이트 (AJAX)

```html
<script>
function submitComment(event) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);

    fetch(form.action, {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        // 댓글 목록 갱신
        location.reload();
    });
}
</script>
```

## 🚀 다음 단계

[Part 7: 검색 및 페이지네이션](./Part7-SearchPagination.md)에서 게시글 검색과 페이지 나누기 기능을 만들어봅시다!
