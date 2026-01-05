# Part 3: 게시글 목록 및 상세 보기

## 📌 이번 Part에서 배울 내용

- 게시글 목록 뷰 만들기
- 게시글 상세 뷰 만들기
- URL 패턴 설정
- 템플릿 상속 및 재사용
- 템플릿에서 데이터 표시하기

## 1. 베이스 템플릿 만들기

여러 페이지에서 공통으로 사용할 기본 레이아웃을 만들어봅시다.

### base.html 생성

**board/templates/board/base.html** 파일을 만들고 다음 내용을 작성하세요:

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

### 코드 설명

- `{% block title %}...{% endblock %}`: 페이지마다 다른 제목 설정 가능
- `{% block content %}...{% endblock %}`: 페이지마다 다른 내용 표시
- `{% url 'board:list' %}`: URL 패턴의 이름으로 URL 생성 (나중에 설정)
- CSS는 inline으로 작성 (Part 8에서 파일로 분리)

**⚠️ 중요**: base.html의 108번 줄에 있는 `{% url 'board:create' %}`는 Part 4에서 구현될 예정입니다. 지금 이 상태로 페이지를 열면 `NoReverseMatch` 에러가 발생합니다. 임시로 테스트하려면 다음 두 가지 방법 중 하나를 선택하세요:

**방법 1**: 글쓰기 링크를 주석 처리
```html
<nav>
    <ul>
        <li><a href="{% url 'board:list' %}">게시글 목록</a></li>
        <!-- <li><a href="{% url 'board:create' %}">글쓰기</a></li> -->
    </ul>
</nav>
```

**방법 2**: Part 3 완료 후 바로 Part 4를 진행하여 create 뷰를 구현

## 2. 게시글 목록 뷰 만들기

### views.py 수정

**board/views.py**를 다음과 같이 수정하세요:

```python
# board/views.py

from django.shortcuts import render, get_object_or_404
from .models import Post


def post_list(request):
    """
    게시글 목록 페이지
    """
    posts = Post.objects.all()
    context = {
        'posts': posts,
    }
    return render(request, 'board/post_list.html', context)


def post_detail(request, pk):
    """
    게시글 상세 페이지
    pk: Primary Key (게시글 ID)
    """
    post = get_object_or_404(Post, pk=pk)

    # 조회수 증가
    post.increase_views()

    context = {
        'post': post,
    }
    return render(request, 'board/post_detail.html', context)
```

### 코드 설명

#### post_list 함수

- `Post.objects.all()`: 모든 게시글을 가져옵니다
- `context`: 템플릿에 전달할 데이터를 담는 딕셔너리
- `render()`: 템플릿을 렌더링하여 HTML 응답 반환

#### post_detail 함수

- `pk`: URL에서 전달받은 게시글 ID
- `get_object_or_404()`: 객체를 가져오고, 없으면 404 에러 표시
- `post.increase_views()`: Part 2에서 만든 조회수 증가 메서드 호출

## 3. URL 패턴 설정

**board/urls.py**를 다음과 같이 수정하세요:

```python
# board/urls.py

from django.urls import path
from . import views

app_name = 'board'

urlpatterns = [
    path('', views.post_list, name='list'),
    path('post/<int:pk>/', views.post_detail, name='detail'),
]
```

### 코드 설명

- `path('', views.post_list, name='list')`:
  - 빈 경로 → 게시글 목록
  - `name='list'`: 이 URL을 'board:list'로 참조 가능

- `path('post/<int:pk>/', views.post_detail, name='detail')`:
  - `<int:pk>`: 정수형 파라미터 (게시글 ID)
  - 예: `/post/1/`, `/post/2/`
  - `name='detail'`: 이 URL을 'board:detail'로 참조 가능

### index 뷰 제거 (옵션)

Part 1에서 만든 `index` 뷰는 더 이상 필요 없으므로 삭제해도 됩니다.

## 4. 게시글 목록 템플릿 만들기

**board/templates/board/post_list.html** 파일을 만들고 다음 내용을 작성하세요:

```html
<!-- board/templates/board/post_list.html -->

{% extends 'board/base.html' %}

{% block title %}게시글 목록 - Django 게시판{% endblock %}

{% block content %}
<h2>게시글 목록</h2>

<!-- ⚠️ Part 4에서 구현될 예정입니다. 지금은 주석 처리하거나 Part 4를 먼저 완료하세요 -->
<!-- <div style="text-align: right; margin-bottom: 20px;">
    <a href="{% url 'board:create' %}" style="background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block;">
        글쓰기
    </a>
</div> -->

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

    <p style="margin-top: 20px; color: #7f8c8d;">
        총 <strong>{{ posts.count }}</strong>개의 게시글이 있습니다.
    </p>
{% else %}
    <p style="text-align: center; padding: 50px; color: #7f8c8d;">
        아직 게시글이 없습니다. 첫 번째 글을 작성해보세요!
    </p>
{% endif %}
{% endblock %}
```

### 템플릿 문법 설명

#### 템플릿 상속

- `{% extends 'board/base.html' %}`: base.html을 상속받습니다
- `{% block content %}...{% endblock %}`: base.html의 content 블록을 채웁니다

#### 조건문

```django
{% if posts %}
    게시글이 있을 때 표시할 내용
{% else %}
    게시글이 없을 때 표시할 내용
{% endif %}
```

#### 반복문

```django
{% for post in posts %}
    {{ post.title }}
{% endfor %}
```

#### 변수 출력

- `{{ post.pk }}`: 게시글 ID
- `{{ post.title }}`: 게시글 제목
- `{{ post.author.username }}`: 작성자 이름
- `{{ post.views }}`: 조회수
- `{{ posts.count }}`: 게시글 개수

#### 필터

- `{{ post.created_at|date:"Y-m-d H:i" }}`: 날짜 형식 지정
  - `Y`: 4자리 연도 (2025)
  - `m`: 2자리 월 (01-12)
  - `d`: 2자리 일 (01-31)
  - `H`: 24시간 형식 시간 (00-23)
  - `i`: 분 (00-59)

#### URL 생성

- `{% url 'board:list' %}`: 목록 페이지 URL
- `{% url 'board:detail' post.pk %}`: 상세 페이지 URL (게시글 ID 전달)

## 5. 게시글 상세 템플릿 만들기

**board/templates/board/post_detail.html** 파일을 만들고 다음 내용을 작성하세요:

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
            <!-- ⚠️ Part 4에서 구현될 예정입니다. 지금은 주석 처리하거나 Part 4를 먼저 완료하세요 -->
            <!-- <a href="{% url 'board:update' post.pk %}" style="background-color: #3498db; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block;">
                수정
            </a>
            <a href="{% url 'board:delete' post.pk %}" style="background-color: #e74c3c; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; display: inline-block;" onclick="return confirm('정말 삭제하시겠습니까?');">
                삭제
            </a> -->
        </div>
    </footer>
</article>
{% endblock %}
```

### 템플릿 문법 설명

#### linebreaks 필터

```django
{{ post.content|linebreaks }}
```

- 텍스트의 줄바꿈을 HTML `<br>` 태그로 변환
- 빈 줄은 `<p>` 태그로 변환

예:
```
안녕하세요

반갑습니다
```

→

```html
<p>안녕하세요</p>

<p>반갑습니다</p>
```

#### JavaScript 확인 대화상자

```html
onclick="return confirm('정말 삭제하시겠습니까?');"
```

삭제 버튼 클릭 시 확인 메시지를 띄웁니다.

## 6. 결과 확인하기

**⚠️ 중요**: 위 템플릿 코드에는 Part 4에서 구현될 URL들(`board:create`, `board:update`, `board:delete`)이 주석 처리되어 있습니다. 주석을 해제하지 말고 그대로 진행하거나, Part 4를 먼저 완료한 후 주석을 해제하세요.

### 서버 실행

```bash
python manage.py runserver
```

### 페이지 확인

1. **게시글 목록**: `http://127.0.0.1:8000/`
   - Admin에서 작성한 게시글들이 테이블 형식으로 표시됩니다
   - 가장 최근 글이 위에 나타납니다
   - 글쓰기 버튼은 Part 4 구현 전까지 표시되지 않습니다

2. **게시글 상세**: 목록에서 제목 클릭
   - 게시글의 전체 내용이 표시됩니다
   - 조회수가 자동으로 증가합니다
   - 수정/삭제 버튼은 Part 4 구현 전까지 표시되지 않습니다

## 7. 더 많은 템플릿 필터 활용

Django는 다양한 내장 필터를 제공합니다.

### 자주 사용하는 필터

```django
{# 문자열 필터 #}
{{ post.title|upper }}           <!-- 대문자로 변환 -->
{{ post.title|lower }}           <!-- 소문자로 변환 -->
{{ post.title|title }}           <!-- 각 단어의 첫 글자 대문자 -->
{{ post.content|truncatewords:10 }}  <!-- 처음 10단어만 표시 -->
{{ post.content|truncatechars:50 }}  <!-- 처음 50자만 표시 -->

{# 숫자 필터 #}
{{ post.views|add:1 }}           <!-- 1 더하기 -->
{{ 1234|floatformat:2 }}         <!-- 1234.00 -->

{# 날짜 필터 #}
{{ post.created_at|date:"Y-m-d" }}
{{ post.created_at|date:"Y년 m월 d일" }}
{{ post.created_at|time:"H:i" }}

{# 기본값 설정 #}
{{ post.author.email|default:"이메일 없음" }}

{# 안전한 HTML 출력 #}
{{ post.content|safe }}

{# 리스트 필터 #}
{{ posts|length }}               <!-- 리스트 길이 -->
{{ posts|first }}                <!-- 첫 번째 항목 -->
{{ posts|last }}                 <!-- 마지막 항목 -->
```

## 8. 템플릿 태그 활용

### if 태그

```django
{% if user.is_authenticated %}
    <p>환영합니다, {{ user.username }}님!</p>
{% else %}
    <p>로그인해주세요.</p>
{% endif %}

{% if post.views > 100 %}
    <span>인기 게시글</span>
{% elif post.views > 50 %}
    <span>주목받는 게시글</span>
{% else %}
    <span>일반 게시글</span>
{% endif %}
```

### for 태그

```django
{% for post in posts %}
    <p>{{ forloop.counter }}. {{ post.title }}</p>
{% empty %}
    <p>게시글이 없습니다.</p>
{% endfor %}
```

**forloop 변수:**
- `forloop.counter`: 1부터 시작하는 인덱스
- `forloop.counter0`: 0부터 시작하는 인덱스
- `forloop.first`: 첫 번째 항목이면 True
- `forloop.last`: 마지막 항목이면 True

### with 태그

```django
{% with total_posts=posts.count %}
    <p>총 {{ total_posts }}개의 게시글</p>
    {% if total_posts > 10 %}
        <p>많은 게시글이 있습니다!</p>
    {% endif %}
{% endwith %}
```

## 9. 템플릿 주석

```django
{# 한 줄 주석 #}

{% comment %}
여러 줄
주석
{% endcomment %}
```

## 📝 Part 3 정리

축하합니다! Part 3을 완료했습니다. 다음을 배웠습니다:

✅ 베이스 템플릿 생성 및 템플릿 상속<br>
✅ 게시글 목록 뷰 및 템플릿<br>
✅ 게시글 상세 뷰 및 템플릿<br>
✅ URL 패턴 설정 (정적 경로 및 동적 파라미터)<br>
✅ 템플릿 문법 (변수, 필터, 태그)<br>
✅ 조회수 자동 증가 기능

## 🔍 자주 발생하는 오류

### 1. TemplateDoesNotExist

**원인**: 템플릿 파일을 찾을 수 없습니다.

**해결**:
- 파일 경로 확인: `board/templates/board/post_list.html`
- 파일 이름 오타 확인
- settings.py의 INSTALLED_APPS에 'board' 등록 확인

### 2. NoReverseMatch at / 또는 NoReverseMatch at /post/1/

**에러 메시지 예시**:
```
NoReverseMatch at /
Reverse for 'create' not found. 'create' is not a valid view function or pattern name.
```

**원인**: 템플릿에서 사용한 URL 이름(예: `'board:create'`, `'board:update'`, `'board:delete'`)에 해당하는 URL 패턴이 urls.py에 정의되어 있지 않습니다.

**해결 방법**:

**해결 1 (권장)**: Part 4를 진행하여 해당 URL 패턴들을 구현
- Part 4에서 create, update, delete 뷰와 URL 패턴을 모두 만들게 됩니다

**해결 2 (임시)**: 해당 링크들을 주석 처리
- **base.html**에서 108번 줄의 글쓰기 링크 주석 처리:
  ```html
  <!-- <li><a href="{% url 'board:create' %}">글쓰기</a></li> -->
  ```

- **post_list.html**에서 글쓰기 버튼 주석 처리 (이미 주석 처리되어 있음)

- **post_detail.html**에서 수정/삭제 버튼 주석 처리 (이미 주석 처리되어 있음)

**해결 3**: urls.py에 `name='list'` 등이 정확히 설정되어 있는지 확인
- `{% url 'board:list' %}` 형식이 올바른지 확인

### 3. VariableDoesNotExist

**원인**: 템플릿에서 존재하지 않는 변수를 사용했습니다.

**해결**:
- context에 해당 변수가 포함되어 있는지 확인
- 변수 이름 오타 확인

### 4. AttributeError: 'Post' object has no attribute 'increase_views'

**원인**: Part 2에서 메서드를 추가하지 않았습니다.

**해결**:
- board/models.py에 `increase_views()` 메서드 추가
- 서버 재시작

## 💡 추가 개선 아이디어

### 1. 게시글이 오늘 작성되었는지 표시

```python
# views.py
from django.utils import timezone

def post_list(request):
    posts = Post.objects.all()
    today = timezone.now().date()

    for post in posts:
        post.is_new = post.created_at.date() == today

    context = {'posts': posts}
    return render(request, 'board/post_list.html', context)
```

```django
<!-- 템플릿 -->
{{ post.title }}
{% if post.is_new %}
    <span style="color: #e74c3c;">NEW</span>
{% endif %}
```

### 2. 게시글이 없을 때 더 나은 UI

```html
{% if not posts %}
    <div style="text-align: center; padding: 100px;">
        <p style="font-size: 1.5em; color: #7f8c8d; margin-bottom: 20px;">
            아직 작성된 게시글이 없습니다
        </p>
        <a href="{% url 'board:create' %}" style="...">
            첫 번째 글 작성하기
        </a>
    </div>
{% endif %}
```

## 🚀 다음 단계

[Part 4: 게시글 작성, 수정, 삭제](./Part4-CRUD.md)에서 게시글을 작성하고 관리하는 기능을 만들어봅시다!
