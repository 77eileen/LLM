# Part 7: 검색 및 페이지네이션

## 📌 이번 Part에서 배울 내용

- 게시글 검색 기능
- 페이지네이션 (페이지 나누기)
- 검색어 하이라이트
- 정렬 기능
- 검색 조건 유지

## 1. 검색 기능 구현

### views.py 수정

**board/views.py**의 `post_list` 함수를 수정하세요:

```python
# board/views.py

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.db.models import Q
from .models import Post, Comment
from .forms import PostForm, CommentForm


def post_list(request):
    """게시글 목록 페이지 (검색 및 페이지네이션 포함)"""
    # 검색어 가져오기
    search_query = request.GET.get('q', '')
    search_type = request.GET.get('type', 'all')

    # 전체 게시글
    posts = Post.objects.all()

    # 검색 처리
    if search_query:
        if search_type == 'title':
            posts = posts.filter(title__icontains=search_query)
        elif search_type == 'content':
            posts = posts.filter(content__icontains=search_query)
        elif search_type == 'author':
            posts = posts.filter(author__username__icontains=search_query)
        else:  # all
            posts = posts.filter(
                Q(title__icontains=search_query) |
                Q(content__icontains=search_query) |
                Q(author__username__icontains=search_query)
            )

    # 페이지네이션
    paginator = Paginator(posts, 10)  # 페이지당 10개씩
    page = request.GET.get('page')

    try:
        posts = paginator.page(page)
    except PageNotAnInteger:
        # 페이지가 정수가 아닌 경우 첫 페이지
        posts = paginator.page(1)
    except EmptyPage:
        # 페이지가 범위를 벗어난 경우 마지막 페이지
        posts = paginator.page(paginator.num_pages)

    context = {
        'posts': posts,
        'search_query': search_query,
        'search_type': search_type,
    }
    return render(request, 'board/post_list.html', context)


def post_detail(request, pk):
    """게시글 상세 페이지"""
    post = get_object_or_404(Post, pk=pk)
    post.increase_views()

    comment_form = CommentForm()

    context = {
        'post': post,
        'comment_form': comment_form,
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

### 코드 설명

#### Q 객체

```python
from django.db.models import Q

posts = posts.filter(
    Q(title__icontains=search_query) |
    Q(content__icontains=search_query) |
    Q(author__username__icontains=search_query)
)
```

- `Q` 객체로 복잡한 쿼리 조건 생성
- `|`: OR 연산자 (제목, 내용, 작성자 중 하나라도 포함)
- `&`: AND 연산자
- `~`: NOT 연산자

#### icontains

```python
title__icontains=search_query
```

- 대소문자 구분 없이 포함 여부 검사
- `contains`: 대소문자 구분

#### Paginator

```python
paginator = Paginator(posts, 10)
page = request.GET.get('page')
posts = paginator.page(page)
```

- `Paginator(posts, 10)`: 10개씩 페이지 나누기
- `page`: URL의 `?page=2` 파라미터
- `paginator.page(page)`: 해당 페이지의 게시글 가져오기

## 2. 검색 폼이 포함된 목록 템플릿

**board/templates/board/post_list.html**을 수정하세요:

```html
<!-- board/templates/board/post_list.html -->

{% extends 'board/base.html' %}

{% block title %}게시글 목록 - Django 게시판{% endblock %}

{% block content %}
<h2>게시글 목록</h2>

<!-- 검색 폼 -->
<form method="get" style="margin: 30px 0; padding: 20px; background-color: #ecf0f1; border-radius: 8px;">
    <div style="display: flex; gap: 10px; align-items: center;">
        <select name="type" style="padding: 10px; border: 1px solid #ddd; border-radius: 4px;">
            <option value="all" {% if search_type == 'all' %}selected{% endif %}>전체</option>
            <option value="title" {% if search_type == 'title' %}selected{% endif %}>제목</option>
            <option value="content" {% if search_type == 'content' %}selected{% endif %}>내용</option>
            <option value="author" {% if search_type == 'author' %}selected{% endif %}>작성자</option>
        </select>

        <input type="text" name="q" value="{{ search_query }}" placeholder="검색어를 입력하세요" style="flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px;">

        <button type="submit" style="background-color: #3498db; color: white; padding: 10px 30px; border: none; border-radius: 4px; cursor: pointer;">
            검색
        </button>

        {% if search_query %}
            <a href="{% url 'board:list' %}" style="background-color: #95a5a6; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block;">
                전체 목록
            </a>
        {% endif %}
    </div>
</form>

<!-- 검색 결과 정보 -->
{% if search_query %}
    <p style="color: #7f8c8d; margin-bottom: 20px;">
        '<strong>{{ search_query }}</strong>' 검색 결과: <strong>{{ posts.paginator.count }}</strong>개의 게시글
    </p>
{% endif %}

<div style="text-align: right; margin-bottom: 20px;">
    <a href="{% url 'board:create' %}" style="background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block;">
        글쓰기
    </a>
</div>

{% if posts %}
    <table style="width: 100%; border-collapse: collapse;">
        <thead>
            <tr style="background-color: #ecf0f1; border-bottom: 2px solid #bdc3c7;">
                <th style="padding: 12px; text-align: center; width: 80px;">번호</th>
                <th style="padding: 12px; text-align: left;">제목</th>
                <th style="padding: 12px; text-align: center; width: 120px;">작성자</th>
                <th style="padding: 12px; text-align: center; width: 100px;">조회수</th>
                <th style="padding: 12px; text-align: center; width: 150px;">작성일</th>
            </tr>
        </thead>
        <tbody>
            {% for post in posts %}
            <tr style="border-bottom: 1px solid #ecf0f1;">
                <td style="padding: 12px; text-align: center;">{{ post.pk }}</td>
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
                <td style="padding: 12px; text-align: center;">{{ post.author.username }}</td>
                <td style="padding: 12px; text-align: center;">{{ post.views }}</td>
                <td style="padding: 12px; text-align: center;">{{ post.created_at|date:"Y-m-d H:i" }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <!-- 페이지네이션 -->
    {% if posts.paginator.num_pages > 1 %}
        <div style="display: flex; justify-content: center; align-items: center; gap: 5px; margin-top: 30px;">
            {% if posts.has_previous %}
                <a href="?page=1{% if search_query %}&q={{ search_query }}&type={{ search_type }}{% endif %}" style="padding: 8px 12px; background-color: #ecf0f1; text-decoration: none; color: #2c3e50; border-radius: 4px;">
                    &laquo; 처음
                </a>
                <a href="?page={{ posts.previous_page_number }}{% if search_query %}&q={{ search_query }}&type={{ search_type }}{% endif %}" style="padding: 8px 12px; background-color: #ecf0f1; text-decoration: none; color: #2c3e50; border-radius: 4px;">
                    &lsaquo; 이전
                </a>
            {% endif %}

            {% for num in posts.paginator.page_range %}
                {% if posts.number == num %}
                    <span style="padding: 8px 12px; background-color: #3498db; color: white; border-radius: 4px; font-weight: bold;">
                        {{ num }}
                    </span>
                {% elif num > posts.number|add:'-3' and num < posts.number|add:'3' %}
                    <a href="?page={{ num }}{% if search_query %}&q={{ search_query }}&type={{ search_type }}{% endif %}" style="padding: 8px 12px; background-color: #ecf0f1; text-decoration: none; color: #2c3e50; border-radius: 4px;">
                        {{ num }}
                    </a>
                {% endif %}
            {% endfor %}

            {% if posts.has_next %}
                <a href="?page={{ posts.next_page_number }}{% if search_query %}&q={{ search_query }}&type={{ search_type }}{% endif %}" style="padding: 8px 12px; background-color: #ecf0f1; text-decoration: none; color: #2c3e50; border-radius: 4px;">
                    다음 &rsaquo;
                </a>
                <a href="?page={{ posts.paginator.num_pages }}{% if search_query %}&q={{ search_query }}&type={{ search_type }}{% endif %}" style="padding: 8px 12px; background-color: #ecf0f1; text-decoration: none; color: #2c3e50; border-radius: 4px;">
                    마지막 &raquo;
                </a>
            {% endif %}
        </div>

        <p style="text-align: center; margin-top: 15px; color: #7f8c8d;">
            {{ posts.number }} / {{ posts.paginator.num_pages }} 페이지
        </p>
    {% endif %}

    <p style="margin-top: 20px; color: #7f8c8d;">
        총 <strong>{{ posts.paginator.count }}</strong>개의 게시글이 있습니다.
    </p>
{% else %}
    <p style="text-align: center; padding: 50px; color: #7f8c8d;">
        {% if search_query %}
            검색 결과가 없습니다.
        {% else %}
            아직 게시글이 없습니다. 첫 번째 글을 작성해보세요!
        {% endif %}
    </p>
{% endif %}
{% endblock %}
```

### 템플릿 설명

#### 검색 폼

```html
<form method="get">
    <select name="type">...</select>
    <input type="text" name="q" value="{{ search_query }}">
    <button type="submit">검색</button>
</form>
```

- `method="get"`: 검색은 GET 방식 사용
- `name="type"`: 검색 타입 (전체, 제목, 내용, 작성자)
- `name="q"`: 검색어
- `value="{{ search_query }}"`: 검색 후에도 검색어 유지

#### 페이지네이션 객체 속성

- `posts.has_previous`: 이전 페이지가 있는지
- `posts.has_next`: 다음 페이지가 있는지
- `posts.previous_page_number`: 이전 페이지 번호
- `posts.next_page_number`: 다음 페이지 번호
- `posts.number`: 현재 페이지 번호
- `posts.paginator.num_pages`: 총 페이지 수
- `posts.paginator.count`: 총 게시글 수
- `posts.paginator.page_range`: 페이지 번호 범위

#### 검색 조건 유지

```html
?page={{ num }}{% if search_query %}&q={{ search_query }}&type={{ search_type }}{% endif %}
```

페이지를 이동해도 검색 조건을 유지합니다.

## 3. 템플릿 태그로 페이지네이션 개선 (선택사항)

더 깔끔한 코드를 위해 커스텀 템플릿 태그를 만들 수 있습니다.

### templatetags 폴더 생성

```bash
# Windows
mkdir board\templatetags
type nul > board\templatetags\__init__.py

# Mac/Linux
mkdir board/templatetags
touch board/templatetags/__init__.py
```

### board_extras.py 생성

**board/templatetags/board_extras.py** 파일을 만들고:

```python
# board/templatetags/board_extras.py

from django import template

register = template.Library()


@register.simple_tag
def url_replace(request, field, value):
    """현재 URL의 파라미터를 유지하면서 특정 파라미터만 변경"""
    dict_ = request.GET.copy()
    dict_[field] = value
    return dict_.urlencode()
```

### 템플릿에서 사용

```html
{% load board_extras %}

<a href="?{% url_replace request 'page' num %}">{{ num }}</a>
```

이렇게 하면 검색 조건을 자동으로 유지합니다.

## 4. 검색어 하이라이트 (선택사항)

검색어를 강조 표시하는 커스텀 필터를 만들어봅시다.

**board/templatetags/board_extras.py**에 추가:

```python
# board/templatetags/board_extras.py

from django import template
from django.utils.safestring import mark_safe
import re

register = template.Library()


@register.simple_tag
def url_replace(request, field, value):
    """현재 URL의 파라미터를 유지하면서 특정 파라미터만 변경"""
    dict_ = request.GET.copy()
    dict_[field] = value
    return dict_.urlencode()


@register.filter
def highlight(text, search):
    """검색어를 하이라이트 처리"""
    if not search:
        return text

    pattern = re.compile(re.escape(search), re.IGNORECASE)
    highlighted = pattern.sub(
        f'<span style="background-color: yellow; font-weight: bold;">{search}</span>',
        str(text)
    )
    return mark_safe(highlighted)
```

### 템플릿에서 사용

```html
{% load board_extras %}

{{ post.title|highlight:search_query }}
```

## 5. 정렬 기능 추가

### views.py 수정

**board/views.py**의 `post_list` 함수를 수정하세요:

```python
def post_list(request):
    """게시글 목록 페이지"""
    search_query = request.GET.get('q', '')
    search_type = request.GET.get('type', 'all')
    order_by = request.GET.get('order', '-created_at')

    posts = Post.objects.all()

    # 검색 처리
    if search_query:
        if search_type == 'title':
            posts = posts.filter(title__icontains=search_query)
        elif search_type == 'content':
            posts = posts.filter(content__icontains=search_query)
        elif search_type == 'author':
            posts = posts.filter(author__username__icontains=search_query)
        else:
            posts = posts.filter(
                Q(title__icontains=search_query) |
                Q(content__icontains=search_query) |
                Q(author__username__icontains=search_query)
            )

    # 정렬 처리
    valid_orders = ['-created_at', 'created_at', '-views', 'views', 'title', '-title']
    if order_by in valid_orders:
        posts = posts.order_by(order_by)

    # 페이지네이션
    paginator = Paginator(posts, 10)
    page = request.GET.get('page')

    try:
        posts = paginator.page(page)
    except PageNotAnInteger:
        posts = paginator.page(1)
    except EmptyPage:
        posts = paginator.page(paginator.num_pages)

    context = {
        'posts': posts,
        'search_query': search_query,
        'search_type': search_type,
        'order_by': order_by,
    }
    return render(request, 'board/post_list.html', context)
```

### 템플릿에 정렬 드롭다운 추가

**board/templates/board/post_list.html**의 `검색폼 아래`에 추가하세요:

```html
<!-- 검색 폼 아래에 추가 -->
<div style="text-align: right; margin-bottom: 10px;">
    <form method="get" style="display: inline;">
        {% if search_query %}
            <input type="hidden" name="q" value="{{ search_query }}">
            <input type="hidden" name="type" value="{{ search_type }}">
        {% endif %}

        <select name="order" onchange="this.form.submit()" style="padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
            <option value="-created_at" {% if order_by == '-created_at' %}selected{% endif %}>최신순</option>
            <option value="created_at" {% if order_by == 'created_at' %}selected{% endif %}>오래된순</option>
            <option value="-views" {% if order_by == '-views' %}selected{% endif %}>조회수 높은순</option>
            <option value="views" {% if order_by == 'views' %}selected{% endif %}>조회수 낮은순</option>
            <option value="title" {% if order_by == 'title' %}selected{% endif %}>제목 가나다순</option>
        </select>
    </form>
</div>
```

## 📝 Part 7 정리

축하합니다! Part 7을 완료했습니다. 다음을 배웠습니다:

✅ 게시글 검색 기능 (제목, 내용, 작성자)<br>
✅ Q 객체를 사용한 복잡한 쿼리<br>
✅ 페이지네이션 구현<br>
✅ 검색 조건 유지<br>
✅ 정렬 기능<br>
✅ 커스텀 템플릿 태그 (선택사항)

## 🔍 자주 발생하는 오류

### 1. EmptyPage 에러

**원인**: 존재하지 않는 페이지 번호

**해결**:
```python
try:
    posts = paginator.page(page)
except EmptyPage:
    posts = paginator.page(paginator.num_pages)
```

### 2. 검색어가 유지되지 않음

**원인**: 페이지 링크에 검색 파라미터 누락

**해결**:
```html
?page={{ num }}&q={{ search_query }}&type={{ search_type }}
```

### 3. TemplateDoesNotExist: board/templatetags

**원인**: `__init__.py` 파일이 없습니다.

**해결**: templatetags 폴더에 `__init__.py` 생성

## 💡 추가 기능 아이디어

### 1. 날짜 범위 검색

```python
start_date = request.GET.get('start_date')
end_date = request.GET.get('end_date')

if start_date and end_date:
    posts = posts.filter(created_at__range=[start_date, end_date])
```

### 2. 인기 게시글 표시

```python
# 조회수 100 이상인 게시글
popular_posts = Post.objects.filter(views__gte=100)
```

### 3. 검색 기록 저장

```python
class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    query = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
```

### 4. 자동완성 검색

```javascript
<input type="text" id="search-input" autocomplete="off">
<div id="autocomplete-results"></div>

<script>
document.getElementById('search-input').addEventListener('input', function(e) {
    const query = e.target.value;
    if (query.length < 2) return;

    fetch(`/api/autocomplete/?q=${query}`)
        .then(response => response.json())
        .then(data => {
            // 자동완성 결과 표시
        });
});
</script>
```

## 🚀 다음 단계

[Part 8: 스타일링 및 최종 마무리](./Part8-Styling.md)에서 CSS를 적용하고 프로젝트를 완성해봅시다!
