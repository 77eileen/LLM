# Part 4: 게시글 작성, 수정, 삭제

## 📌 이번 Part에서 배울 내용

- Django Form 사용하기
- 게시글 작성 기능 (Create)
- 게시글 수정 기능 (Update)
- 게시글 삭제 기능 (Delete)
- 폼 유효성 검사
- 리다이렉트 처리

## 1. Django Form 만들기

Django Form은 HTML 폼을 Python 클래스로 정의하여 유효성 검사와 데이터 처리를 쉽게 해줍니다.

### forms.py 생성

**board** 폴더에 **forms.py** 파일을 새로 만들고 다음 내용을 작성하세요:

```python
# board/forms.py

from django import forms
from .models import Post


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
        """제목 유효성 검사"""
        title = self.cleaned_data.get('title')
        if len(title) < 2:
            raise forms.ValidationError('제목은 2글자 이상 입력해주세요.')
        return title

    def clean_content(self):
        """내용 유효성 검사"""
        content = self.cleaned_data.get('content')
        if len(content) < 10:
            raise forms.ValidationError('내용은 10글자 이상 입력해주세요.')
        return content
```

### 코드 설명

#### ModelForm

- `forms.ModelForm`: 모델을 기반으로 폼을 자동 생성
- `model = Post`: Post 모델 사용
- `fields = ['title', 'content']`: 폼에 포함할 필드 (author는 자동 설정)

#### widgets

HTML 입력 요소를 커스터마이즈합니다:

- `TextInput`: 한 줄 텍스트 입력
- `Textarea`: 여러 줄 텍스트 입력
- `attrs`: HTML 속성 설정
  - `class`: CSS 클래스
  - `placeholder`: 안내 문구
  - `maxlength`: 최대 글자 수
  - `rows`: 텍스트 영역 높이

#### clean 메서드

각 필드의 유효성을 검사합니다:

- `clean_title()`: 제목 검사
- `clean_content()`: 내용 검사
- `ValidationError`: 유효하지 않으면 에러 발생

## 2. 게시글 작성 뷰 만들기

**board/views.py**에 다음 코드를 추가하세요:

```python
# board/views.py

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Post
from .forms import PostForm


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
    context = {
        'post': post,
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

    # 작성자만 수정 가능
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

    # 작성자만 삭제 가능
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
```

### 코드 설명

#### @login_required 데코레이터

```python
@login_required
def post_create(request):
    ...
```

- 로그인한 사용자만 접근 가능
- 로그인하지 않으면 로그인 페이지로 이동 (Part 5에서 설정)

#### request.method

- `GET`: 페이지를 보여줄 때 (폼을 화면에 표시)
- `POST`: 폼을 제출할 때 (데이터를 처리)

#### post_create 함수 흐름

1. **POST 요청 (폼 제출)**:
   - `form = PostForm(request.POST)`: 제출된 데이터로 폼 생성
   - `form.is_valid()`: 유효성 검사
   - `post = form.save(commit=False)`: 임시 저장 (DB에 저장하지 않음)
   - `post.author = request.user`: 작성자 설정 (현재 로그인한 사용자)
   - `post.save()`: DB에 저장
   - `messages.success()`: 성공 메시지 추가
   - `redirect()`: 상세 페이지로 이동

2. **GET 요청 (페이지 열기)**:
   - `form = PostForm()`: 빈 폼 생성
   - 템플릿 렌더링

#### post_update 함수 특징

- `instance=post`: 기존 게시글 데이터를 폼에 채움
- 작성자 확인: `post.author != request.user`

#### post_delete 함수 특징

- POST 요청일 때만 실제 삭제
- GET 요청이면 삭제 확인 페이지 표시

#### messages 프레임워크

```python
messages.success(request, '성공 메시지')
messages.error(request, '에러 메시지')
messages.warning(request, '경고 메시지')
messages.info(request, '정보 메시지')
```

사용자에게 알림 메시지를 표시합니다.

## 3. URL 패턴 추가

**board/urls.py**를 다음과 같이 수정하세요:

```python
# board/urls.py

from django.urls import path
from . import views

app_name = 'board'

urlpatterns = [
    path('', views.post_list, name='list'),
    path('post/<int:pk>/', views.post_detail, name='detail'),
    path('post/create/', views.post_create, name='create'),
    path('post/<int:pk>/update/', views.post_update, name='update'),
    path('post/<int:pk>/delete/', views.post_delete, name='delete'),
]
```

### URL 패턴 순서 주의

```python
path('post/create/', ...)  # 이것이 먼저 와야 함
path('post/<int:pk>/', ...)  # 이것이 나중에
```

만약 순서를 바꾸면 `create`가 `pk`로 인식되어 오류가 발생합니다!

## 4. 게시글 작성/수정 템플릿

**board/templates/board/post_form.html** 파일을 만들고 다음 내용을 작성하세요:

```html
<!-- board/templates/board/post_form.html -->

{% extends 'board/base.html' %}

{% block title %}
    {% if post %}게시글 수정{% else %}게시글 작성{% endif %} - Django 게시판
{% endblock %}

{% block content %}
<h2>{% if post %}게시글 수정{% else %}게시글 작성{% endif %}</h2>

<form method="post" style="margin-top: 30px;">
    {% csrf_token %}

    {% if form.non_field_errors %}
        <div style="background-color: #ffe6e6; border-left: 4px solid #e74c3c; padding: 15px; margin-bottom: 20px;">
            {{ form.non_field_errors }}
        </div>
    {% endif %}

    <div style="margin-bottom: 20px;">
        <label for="{{ form.title.id_for_label }}" style="display: block; margin-bottom: 5px; font-weight: bold;">
            {{ form.title.label }}
            <span style="color: #e74c3c;">*</span>
        </label>
        {{ form.title }}
        {% if form.title.errors %}
            <div style="color: #e74c3c; font-size: 0.9em; margin-top: 5px;">
                {{ form.title.errors }}
            </div>
        {% endif %}
    </div>

    <div style="margin-bottom: 20px;">
        <label for="{{ form.content.id_for_label }}" style="display: block; margin-bottom: 5px; font-weight: bold;">
            {{ form.content.label }}
            <span style="color: #e74c3c;">*</span>
        </label>
        {{ form.content }}
        {% if form.content.errors %}
            <div style="color: #e74c3c; font-size: 0.9em; margin-top: 5px;">
                {{ form.content.errors }}
            </div>
        {% endif %}
    </div>

    <div style="display: flex; gap: 10px; margin-top: 30px;">
        <button type="submit" style="background-color: #3498db; color: white; padding: 12px 30px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em;">
            {% if post %}수정하기{% else %}작성하기{% endif %}
        </button>
        <a href="{% if post %}{% url 'board:detail' post.pk %}{% else %}{% url 'board:list' %}{% endif %}"
           style="background-color: #95a5a6; color: white; padding: 12px 30px; text-decoration: none; border-radius: 4px; display: inline-block;">
            취소
        </a>
    </div>
</form>

<style>
    .form-control {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 1em;
        font-family: inherit;
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

#### {% csrf_token %}

```django
{% csrf_token %}
```

- CSRF(Cross-Site Request Forgery) 공격 방지
- POST 요청 시 **반드시** 포함해야 함
- 빠뜨리면 403 Forbidden 에러 발생

#### 폼 필드 렌더링

```django
{{ form.title }}
```

forms.py에서 정의한 widget 설정대로 HTML input 태그가 생성됩니다.

#### 에러 메시지 표시

```django
{% if form.title.errors %}
    <div style="color: #e74c3c;">
        {{ form.title.errors }}
    </div>
{% endif %}
```

유효성 검사 실패 시 에러 메시지를 표시합니다.

#### 조건부 텍스트

```django
{% if post %}게시글 수정{% else %}게시글 작성{% endif %}
```

post 변수가 있으면 (수정 모드) "게시글 수정", 없으면 (작성 모드) "게시글 작성"을 표시합니다.

## 5. 게시글 삭제 확인 템플릿

**board/templates/board/post_confirm_delete.html** 파일을 만들고 다음 내용을 작성하세요:

```html
<!-- board/templates/board/post_confirm_delete.html -->

{% extends 'board/base.html' %}

{% block title %}게시글 삭제 - Django 게시판{% endblock %}

{% block content %}
<h2>게시글 삭제</h2>

<div style="background-color: #ffe6e6; border-left: 4px solid #e74c3c; padding: 20px; margin: 30px 0;">
    <h3 style="margin-bottom: 15px;">정말 이 게시글을 삭제하시겠습니까?</h3>
    <p style="font-size: 1.1em; margin-bottom: 10px;">
        <strong>제목:</strong> {{ post.title }}
    </p>
    <p style="color: #7f8c8d;">
        이 작업은 되돌릴 수 없습니다.
    </p>
</div>

<form method="post">
    {% csrf_token %}
    <div style="display: flex; gap: 10px;">
        <button type="submit" style="background-color: #e74c3c; color: white; padding: 12px 30px; border: none; border-radius: 4px; cursor: pointer; font-size: 1em;">
            삭제하기
        </button>
        <a href="{% url 'board:detail' post.pk %}" style="background-color: #95a5a6; color: white; padding: 12px 30px; text-decoration: none; border-radius: 4px; display: inline-block;">
            취소
        </a>
    </div>
</form>
{% endblock %}
```

## 6. 메시지 표시 기능 추가

**board/templates/board/base.html**을 수정하여 메시지를 표시하도록 합니다:

```html
<!-- board/templates/board/base.html -->

<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Django 게시판{% endblock %}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px 0;
            margin-bottom: 30px;
        }

        header h1 {
            text-align: center;
        }

        nav {
            background-color: #34495e;
            padding: 10px 0;
            margin-bottom: 30px;
        }

        nav ul {
            list-style: none;
            display: flex;
            justify-content: center;
            gap: 20px;
        }

        nav a {
            color: white;
            text-decoration: none;
            padding: 5px 15px;
            border-radius: 4px;
            transition: background-color 0.3s;
        }

        nav a:hover {
            background-color: #2c3e50;
        }

        .content {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-height: 500px;
        }

        footer {
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            margin-top: 30px;
        }

        /* 메시지 스타일 */
        .messages {
            list-style: none;
            margin-bottom: 20px;
        }

        .message {
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 4px;
            border-left: 4px solid;
        }

        .message.success {
            background-color: #d4edda;
            border-color: #28a745;
            color: #155724;
        }

        .message.error {
            background-color: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }

        .message.warning {
            background-color: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }

        .message.info {
            background-color: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }
    </style>
</head>
<body>
    <header>
        <h1>Django 게시판</h1>
    </header>

    <nav>
        <ul>
            <li><a href="{% url 'board:list' %}">게시글 목록</a></li>
            <li><a href="{% url 'board:create' %}">글쓰기</a></li>
        </ul>
    </nav>

    <div class="container">
        <!-- 메시지 표시 -->
        {% if messages %}
            <ul class="messages">
                {% for message in messages %}
                    <li class="message {{ message.tags }}">
                        {{ message }}
                    </li>
                {% endfor %}
            </ul>
        {% endif %}

        <div class="content">
            {% block content %}
            {% endblock %}
        </div>
    </div>

    <footer>
        <p>&copy; 2025 Django 게시판. All rights reserved.</p>
    </footer>
</body>
</html>
```

## 7. settings.py 설정

**myboard_project/settings.py**에 다음 내용이 있는지 확인하세요 (보통 기본으로 있음):

```python
# myboard_project/settings.py

# 메시지 프레임워크 (기본으로 포함되어 있음)
INSTALLED_APPS = [
    ...
    'django.contrib.messages',
    ...
]

MIDDLEWARE = [
    ...
    'django.contrib.messages.middleware.MessageMiddleware',
    ...
]

# 로그인 후 리다이렉트 URL
LOGIN_REDIRECT_URL = '/'
LOGIN_URL = '/accounts/login/'  # Part 5에서 설정할 로그인 페이지
```

## 8. 기능 테스트하기

### 1) 게시글 작성

1. 서버 실행: `python manage.py runserver`
2. `http://127.0.0.1:8000/`로 접속
3. 상단 메뉴에서 **"글쓰기"** 클릭
4. 제목과 내용 입력 후 **"작성하기"** 클릭
5. 성공 메시지 확인 및 작성한 게시글 확인

### 2) 게시글 수정

1. 작성한 게시글 상세 페이지로 이동
2. **"수정"** 버튼 클릭
3. 내용 수정 후 **"수정하기"** 클릭
4. 수정 완료 메시지 확인

### 3) 게시글 삭제

1. 게시글 상세 페이지에서 **"삭제"** 버튼 클릭
2. 삭제 확인 페이지에서 **"삭제하기"** 클릭
3. 목록 페이지로 이동하고 삭제 완료 메시지 확인

## 📝 Part 4 정리

축하합니다! Part 4를 완료했습니다. 다음을 배웠습니다:

✅ Django Form 및 ModelForm 생성<br>
✅ 게시글 작성 기능 (Create)<br>
✅ 게시글 수정 기능 (Update)<br>
✅ 게시글 삭제 기능 (Delete)<br>
✅ 폼 유효성 검사 및 에러 처리<br>
✅ Messages 프레임워크 사용<br>
✅ 리다이렉트 처리

## 🔍 자주 발생하는 오류

### 1. Forbidden (403) CSRF verification failed

**원인**: 폼에 `{% csrf_token %}`이 없습니다.

**해결**:
```django
<form method="post">
    {% csrf_token %}
    ...
</form>
```

### 2. IntegrityError: NOT NULL constraint failed: board_post.author_id

**원인**: 작성자를 설정하지 않았습니다.

**해결**:
```python
post = form.save(commit=False)
post.author = request.user  # 이 줄 추가!
post.save()
```

### 3. 로그인하지 않았는데 접근 가능

**원인**: `@login_required` 데코레이터를 빼먹었습니다.

**해결**:
```python
@login_required
def post_create(request):
    ...
```

### 4. ValidationError가 표시되지 않음

**원인**: 템플릿에서 에러를 표시하지 않았습니다.

**해결**:
```django
{% if form.title.errors %}
    {{ form.title.errors }}
{% endif %}
```

## 💡 추가 기능 아이디어

### 1. 작성 취소 시 확인 메시지

```html
<a href="..." onclick="return confirm('작성을 취소하시겠습니까?');">
    취소
</a>
```

### 2. 필수 필드 표시

```django
<label>
    {{ form.title.label }}
    {% if form.title.field.required %}
        <span style="color: red;">*</span>
    {% endif %}
</label>
```

### 3. 글자 수 카운터

```html
<textarea id="id_content" ...></textarea>
<div style="text-align: right; color: #7f8c8d;">
    <span id="char-count">0</span> / 1000자
</div>

<script>
    const textarea = document.getElementById('id_content');
    const charCount = document.getElementById('char-count');

    textarea.addEventListener('input', function() {
        charCount.textContent = this.value.length;
    });

    // 초기값 설정
    charCount.textContent = textarea.value.length;
</script>
```

### 4. 자동 저장 (localStorage)

```html
<script>
    const titleInput = document.querySelector('input[name="title"]');
    const contentInput = document.querySelector('textarea[name="content"]');

    // 불러오기
    titleInput.value = localStorage.getItem('draft_title') || '';
    contentInput.value = localStorage.getItem('draft_content') || '';

    // 저장
    titleInput.addEventListener('input', function() {
        localStorage.setItem('draft_title', this.value);
    });

    contentInput.addEventListener('input', function() {
        localStorage.setItem('draft_content', this.value);
    });

    // 폼 제출 시 삭제
    document.querySelector('form').addEventListener('submit', function() {
        localStorage.removeItem('draft_title');
        localStorage.removeItem('draft_content');
    });
</script>
```

## 🚀 다음 단계

[Part 5: 사용자 인증 기능](./Part5-Authentication.md)에서 회원가입과 로그인 기능을 만들어봅시다!
