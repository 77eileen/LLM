# Part 8: 스타일링 및 최종 마무리

## 📌 이번 Part에서 배울 내용

- 정적 파일(CSS) 설정
- 전문적인 스타일 적용
- 반응형 디자인
- 배포 준비
- 보안 설정
- 성능 최적화

## 1. 정적 파일 설정

### settings.py 설정 확인

**myboard_project/settings.py**에서 다음 설정을 확인하세요:

```python
# myboard_project/settings.py

import os

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'

# 개발 환경에서 정적 파일 경로
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

# 배포 시 정적 파일이 모일 경로
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
```

### static 폴더 생성

프로젝트 루트에 static 폴더를 만듭니다:

```bash
# Windows
mkdir static\css
mkdir static\js

# Mac/Linux
mkdir -p static/css
mkdir -p static/js
```

폴더 구조:
```
myboard/
├── board/
├── accounts/
├── myboard_project/
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
└── manage.py
```

## 2. CSS 파일 생성

**static/css/style.css** 파일을 만들고 다음 내용을 작성하세요:

```css
/* static/css/style.css */

/* ========== 전역 스타일 ========== */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    --primary-color: #3498db;
    --secondary-color: #2c3e50;
    --success-color: #27ae60;
    --danger-color: #e74c3c;
    --warning-color: #f39c12;
    --info-color: #16a085;
    --light-gray: #ecf0f1;
    --dark-gray: #95a5a6;
    --text-color: #2c3e50;
    --border-color: #bdc3c7;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.6;
    color: var(--text-color);
    background-color: #f5f5f5;
}

/* ========== 헤더 ========== */
header {
    background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
    color: white;
    padding: 30px 0;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

header h1 {
    text-align: center;
    font-size: 2.5em;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
}

/* ========== 네비게이션 ========== */
nav {
    background-color: var(--secondary-color);
    padding: 15px 0;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

nav ul {
    list-style: none;
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}

nav a {
    color: white;
    text-decoration: none;
    padding: 10px 20px;
    border-radius: 5px;
    transition: all 0.3s ease;
    font-weight: 500;
}

nav a:hover {
    background-color: var(--primary-color);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

/* ========== 컨테이너 ========== */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.content {
    background-color: white;
    padding: 40px;
    border-radius: 10px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    min-height: 500px;
    margin-bottom: 30px;
}

/* ========== 메시지 ========== */
.messages {
    list-style: none;
    margin-bottom: 20px;
}

.message {
    padding: 15px 20px;
    margin-bottom: 15px;
    border-radius: 8px;
    border-left: 5px solid;
    animation: slideIn 0.3s ease;
}

@keyframes slideIn {
    from {
        transform: translateX(-100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

.message.success {
    background-color: #d4edda;
    border-color: var(--success-color);
    color: #155724;
}

.message.error {
    background-color: #f8d7da;
    border-color: var(--danger-color);
    color: #721c24;
}

.message.warning {
    background-color: #fff3cd;
    border-color: var(--warning-color);
    color: #856404;
}

.message.info {
    background-color: #d1ecf1;
    border-color: var(--info-color);
    color: #0c5460;
}

/* ========== 버튼 ========== */
.btn {
    display: inline-block;
    padding: 12px 24px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 1em;
    text-decoration: none;
    transition: all 0.3s ease;
    font-weight: 500;
}

.btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.btn-primary {
    background-color: var(--primary-color);
    color: white;
}

.btn-primary:hover {
    background-color: #2980b9;
}

.btn-success {
    background-color: var(--success-color);
    color: white;
}

.btn-danger {
    background-color: var(--danger-color);
    color: white;
}

.btn-secondary {
    background-color: var(--dark-gray);
    color: white;
}

/* ========== 테이블 ========== */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

thead {
    background-color: var(--light-gray);
    border-bottom: 2px solid var(--border-color);
}

th, td {
    padding: 15px;
    text-align: left;
}

th {
    font-weight: 600;
    color: var(--secondary-color);
}

tbody tr {
    border-bottom: 1px solid var(--light-gray);
    transition: background-color 0.2s ease;
}

tbody tr:hover {
    background-color: #f8f9fa;
}

/* ========== 폼 ========== */
.form-control {
    width: 100%;
    padding: 12px;
    border: 2px solid #ddd;
    border-radius: 5px;
    font-size: 1em;
    font-family: inherit;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.form-control:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
}

.form-group {
    margin-bottom: 20px;
}

.form-label {
    display: block;
    margin-bottom: 8px;
    font-weight: 600;
    color: var(--secondary-color);
}

.form-help {
    font-size: 0.9em;
    color: var(--dark-gray);
    margin-top: 5px;
}

.form-error {
    color: var(--danger-color);
    font-size: 0.9em;
    margin-top: 5px;
}

/* ========== 검색 폼 ========== */
.search-form {
    background-color: var(--light-gray);
    padding: 25px;
    border-radius: 10px;
    margin: 30px 0;
}

.search-form-group {
    display: flex;
    gap: 10px;
    align-items: center;
}

.search-form select {
    padding: 12px;
    border: 2px solid #ddd;
    border-radius: 5px;
    font-size: 1em;
}

.search-form input[type="text"] {
    flex: 1;
    padding: 12px;
    border: 2px solid #ddd;
    border-radius: 5px;
    font-size: 1em;
}

/* ========== 페이지네이션 ========== */
.pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
    margin: 30px 0;
}

.pagination a,
.pagination span {
    padding: 10px 15px;
    background-color: var(--light-gray);
    text-decoration: none;
    color: var(--text-color);
    border-radius: 5px;
    transition: all 0.3s ease;
}

.pagination a:hover {
    background-color: var(--primary-color);
    color: white;
    transform: translateY(-2px);
}

.pagination .current {
    background-color: var(--primary-color);
    color: white;
    font-weight: bold;
}

/* ========== 댓글 ========== */
.comment-section {
    margin-top: 50px;
}

.comment-section h3 {
    margin-bottom: 25px;
    padding-bottom: 15px;
    border-bottom: 3px solid var(--light-gray);
}

.comment-item {
    border-bottom: 1px solid var(--light-gray);
    padding: 25px 0;
    transition: background-color 0.2s ease;
}

.comment-item:hover {
    background-color: #fafafa;
    padding-left: 10px;
}

.comment-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 15px;
}

.comment-author {
    font-weight: 600;
    color: var(--secondary-color);
}

.comment-date {
    color: var(--dark-gray);
    font-size: 0.9em;
}

.comment-content {
    line-height: 1.8;
    color: #555;
}

.comment-actions {
    display: flex;
    gap: 8px;
}

.comment-actions button,
.comment-actions .btn {
    padding: 6px 12px;
    font-size: 0.9em;
}

/* ========== 게시글 상세 ========== */
.post-header {
    border-bottom: 3px solid var(--light-gray);
    padding-bottom: 25px;
    margin-bottom: 30px;
}

.post-header h2 {
    margin-bottom: 20px;
    color: var(--secondary-color);
    font-size: 2em;
}

.post-meta {
    display: flex;
    justify-content: space-between;
    color: var(--dark-gray);
    font-size: 0.95em;
    flex-wrap: wrap;
    gap: 10px;
}

.post-content {
    min-height: 300px;
    padding: 30px 0;
    line-height: 2;
    font-size: 1.1em;
}

.post-footer {
    border-top: 2px solid var(--light-gray);
    padding-top: 25px;
    margin-top: 30px;
}

/* ========== 푸터 ========== */
footer {
    text-align: center;
    padding: 30px;
    color: var(--dark-gray);
    background-color: white;
    margin-top: 50px;
    border-top: 1px solid var(--light-gray);
}

/* ========== 반응형 디자인 ========== */
@media (max-width: 768px) {
    .content {
        padding: 20px;
    }

    header h1 {
        font-size: 1.8em;
    }

    nav ul {
        flex-direction: column;
        align-items: center;
    }

    table {
        font-size: 0.9em;
    }

    th, td {
        padding: 10px;
    }

    .search-form-group {
        flex-direction: column;
    }

    .search-form input[type="text"] {
        width: 100%;
    }

    .post-meta {
        flex-direction: column;
        gap: 5px;
    }
}

@media (max-width: 480px) {
    .container {
        padding: 10px;
    }

    .content {
        padding: 15px;
    }

    header h1 {
        font-size: 1.5em;
    }

    .btn {
        padding: 10px 15px;
        font-size: 0.9em;
    }
}

/* ========== 유틸리티 클래스 ========== */
.text-center {
    text-align: center;
}

.text-right {
    text-align: right;
}

.mt-1 { margin-top: 10px; }
.mt-2 { margin-top: 20px; }
.mt-3 { margin-top: 30px; }

.mb-1 { margin-bottom: 10px; }
.mb-2 { margin-bottom: 20px; }
.mb-3 { margin-bottom: 30px; }

.p-1 { padding: 10px; }
.p-2 { padding: 20px; }
.p-3 { padding: 30px; }

.flex {
    display: flex;
}

.flex-between {
    display: flex;
    justify-content: space-between;
}

.flex-center {
    display: flex;
    justify-content: center;
    align-items: center;
}

.gap-1 { gap: 10px; }
.gap-2 { gap: 20px; }

/* ========== 애니메이션 ========== */
.fade-in {
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ========== 로딩 스피너 ========== */
.spinner {
    border: 4px solid var(--light-gray);
    border-top: 4px solid var(--primary-color);
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
```

## 3. base.html에 CSS 적용

**board/templates/board/base.html**을 수정하세요:

```html
<!-- board/templates/board/base.html -->

{% load static %}
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Django 게시판{% endblock %}</title>
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
</head>
<body>
    <header>
        <h1>Django 게시판</h1>
    </header>

    <nav>
        <ul>
            <li><a href="{% url 'board:list' %}">게시글 목록</a></li>
            {% if user.is_authenticated %}
                <li><a href="{% url 'board:create' %}">글쓰기</a></li>
                <li><a href="{% url 'accounts:profile' %}">{{ user.username }}님</a></li>
                <li><a href="{% url 'accounts:logout' %}">로그아웃</a></li>
            {% else %}
                <li><a href="{% url 'accounts:login' %}">로그인</a></li>
                <li><a href="{% url 'accounts:signup' %}">회원가입</a></li>
            {% endif %}
        </ul>
    </nav>

    <div class="container">
        {% if messages %}
            <ul class="messages">
                {% for message in messages %}
                    <li class="message {{ message.tags }}">
                        {{ message }}
                    </li>
                {% endfor %}
            </ul>
        {% endif %}

        <div class="content fade-in">
            {% block content %}
            {% endblock %}
        </div>
    </div>

    <footer>
        <p>&copy; 2025 Django 게시판. All rights reserved.</p>
        <p style="margin-top: 10px; font-size: 0.9em;">
            Made with Django & Python
        </p>
    </footer>

    {% block extra_js %}
    {% endblock %}
</body>
</html>
```

## 4. 템플릿에 CSS 클래스 적용

이제 모든 템플릿에서 CSS 클래스를 사용할 수 있습니다. 예시는 생략하지만, 기존 inline 스타일을 CSS 클래스로 변경하세요.

예:
```html
<!-- 이전 -->
<button style="background-color: #3498db; color: white; padding: 10px 20px;">버튼</button>

<!-- 이후 -->
<button class="btn btn-primary">버튼</button>
```

## 5. JavaScript 추가 (선택사항)

**static/js/script.js** 파일을 만들고:

```javascript
// static/js/script.js

// 메시지 자동 숨김 (3초 후)
document.addEventListener('DOMContentLoaded', function() {
    const messages = document.querySelectorAll('.message');

    messages.forEach(function(message) {
        setTimeout(function() {
            message.style.transition = 'opacity 0.5s ease';
            message.style.opacity = '0';

            setTimeout(function() {
                message.remove();
            }, 500);
        }, 3000);
    });
});

// 폼 제출 시 버튼 비활성화 (중복 제출 방지)
const forms = document.querySelectorAll('form');
forms.forEach(function(form) {
    form.addEventListener('submit', function(e) {
        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn && !form.classList.contains('no-disable')) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '처리 중...';
        }
    });
});

// 맨 위로 스크롤 버튼
window.addEventListener('scroll', function() {
    const scrollBtn = document.getElementById('scroll-top');
    if (scrollBtn) {
        if (window.pageYOffset > 300) {
            scrollBtn.style.display = 'block';
        } else {
            scrollBtn.style.display = 'none';
        }
    }
});

function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}
```

**base.html**에 스크립트 추가:

```html
<script src="{% static 'js/script.js' %}"></script>

<!-- 맨 위로 버튼 -->
<button id="scroll-top" onclick="scrollToTop()" style="display: none; position: fixed; bottom: 30px; right: 30px; background-color: var(--primary-color); color: white; border: none; border-radius: 50%; width: 50px; height: 50px; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.3); z-index: 1000;">
    ↑
</button>
```

## 6. 배포 준비

### DEBUG 설정

**myboard_project/settings.py**:

```python
# 개발 환경
DEBUG = True
ALLOWED_HOSTS = []

# 배포 환경 (나중에)
# DEBUG = False
# ALLOWED_HOSTS = ['yourdomain.com', 'www.yourdomain.com']
```

### SECRET_KEY 보안

민감한 정보는 환경 변수로 관리:

```python
# settings.py
import os

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'your-default-secret-key')
```

### requirements.txt 생성

```bash
pip freeze > requirements.txt
```

이 파일은 프로젝트에 필요한 모든 패키지 목록입니다.

### .gitignore 생성

Git 저장소에 포함하지 않을 파일 목록:

```
# .gitignore

*.pyc
__pycache__/
db.sqlite3
venv/
.env
staticfiles/
media/
```

## 7. 보안 설정

**settings.py**에 보안 설정 추가:

```python
# myboard_project/settings.py

# CSRF 설정
CSRF_COOKIE_SECURE = False  # 배포 시 True
SESSION_COOKIE_SECURE = False  # 배포 시 True

# XSS 보호
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True

# 비밀번호 검증
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 8,
        }
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]
```

## 8. 성능 최적화

### 데이터베이스 쿼리 최적화

**board/views.py**:

```python
def post_list(request):
    # select_related로 작성자 정보 미리 가져오기 (N+1 쿼리 방지)
    posts = Post.objects.select_related('author').all()

    # prefetch_related로 댓글 개수 미리 계산
    posts = posts.prefetch_related('comments')

    # ... 나머지 코드
```

### 정적 파일 압축 (배포 시)

```bash
python manage.py collectstatic
```

## 📝 Part 8 정리

축하합니다! 전체 튜토리얼을 완료했습니다! 🎉

다음을 배웠습니다:

✅ 정적 파일 설정<br>
✅ 전문적인 CSS 스타일 적용<br>
✅ 반응형 디자인<br>
✅ JavaScript 활용<br>
✅ 보안 설정<br>
✅ 성능 최적화<br>
✅ 배포 준비

## 🎓 완성된 기능

이제 여러분의 게시판에는 다음 기능이 모두 구현되어 있습니다:

1. ✅ 회원가입 / 로그인 / 로그아웃<br>
2. ✅ 게시글 작성 / 수정 / 삭제<br>
3. ✅ 댓글 작성 / 수정 / 삭제<br>
4. ✅ 검색 기능<br>
5. ✅ 페이지네이션<br>
6. ✅ 조회수 증가<br>
7. ✅ 사용자 프로필<br>
8. ✅ 비밀번호 변경<br>
9. ✅ 반응형 디자인

## 🚀 추가 학습 주제

더 발전하고 싶다면 다음을 시도해보세요:

### 1. 파일 업로드

```python
# models.py
class Post(models.Model):
    # ...
    image = models.ImageField(upload_to='posts/', blank=True, null=True)
    file = models.FileField(upload_to='files/', blank=True, null=True)
```

### 2. 좋아요 기능

```python
# models.py
class Post(models.Model):
    # ...
    likes = models.ManyToManyField(User, related_name='liked_posts', blank=True)

# views.py
@login_required
def post_like(request, pk):
    post = get_object_or_404(Post, pk=pk)
    if request.user in post.likes.all():
        post.likes.remove(request.user)
    else:
        post.likes.add(request.user)
    return redirect('board:detail', pk=pk)
```

### 3. 태그 기능

```bash
pip install django-taggit
```

```python
# models.py
from taggit.managers import TaggableManager

class Post(models.Model):
    # ...
    tags = TaggableManager()
```

### 4. 소셜 로그인

```bash
pip install django-allauth
```

### 5. REST API

```bash
pip install djangorestframework
```

### 6. 실시간 알림

```bash
pip install channels
```

### 7. 배포

- **Heroku**: 초보자 친화적
- **PythonAnywhere**: 무료 호스팅
- **AWS / Google Cloud / Azure**: 프로덕션 환경

## 📚 추천 학습 자료

- Django 공식 문서: https://docs.djangoproject.com/
- Django Girls Tutorial: https://tutorial.djangogirls.org/
- Two Scoops of Django (책)
- Real Python Django Tutorials

## 🎉 마무리

축하합니다! 여러분은 이제:

- Django 프레임워크의 기본을 이해했습니다
- MVT(Model-View-Template) 패턴을 적용할 수 있습니다
- 사용자 인증을 구현할 수 있습니다
- 데이터베이스를 설계하고 관리할 수 있습니다
- 실제로 동작하는 웹 애플리케이션을 만들 수 있습니다

이것은 시작일 뿐입니다. 계속 연습하고, 실험하고, 새로운 기능을 추가해보세요!

**Happy Coding! 🚀**
