# Part 1: 환경 설정 및 프로젝트 생성

## 📌 이번 Part에서 배울 내용

- Python 설치 확인
- Django 설치
- Django 프로젝트 생성
- Django 앱 생성
- 기본 설정
- 첫 페이지 만들기

## 1. Python 설치 확인

Django는 Python으로 만들어진 웹 프레임워크입니다. 먼저 Python이 설치되어 있는지 확인해봅시다.

### Windows 사용자

명령 프롬프트(CMD)를 열고 다음 명령어를 입력하세요:

```bash
python --version
```

### Mac/Linux 사용자

터미널을 열고 다음 명령어를 입력하세요:

```bash
python3 --version
```

**결과 예시:**
```
Python 3.9.7
```

> **주의**: Python 3.8 이상이 설치되어 있어야 합니다. 설치되어 있지 않다면 [Python 공식 웹사이트](https://www.python.org/downloads/)에서 다운로드하세요.

## 2. 가상환경 생성

가상환경은 프로젝트마다 독립적인 Python 환경을 만들어줍니다. 이렇게 하면 프로젝트 간 패키지 충돌을 방지할 수 있습니다.

### 작업 폴더 만들기

먼저 프로젝트를 저장할 폴더를 만듭니다:

```bash
# Windows
mkdir C:\myboard
cd C:\myboard

# Mac/Linux
mkdir ~/myboard
cd ~/myboard
```

### 가상환경 생성

```bash
# Windows
python -m venv venv

# Mac/Linux
python3 -m venv venv
```

`venv`라는 이름의 가상환경이 생성됩니다.

### 가상환경 활성화

```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**성공하면 명령 프롬프트 앞에 `(venv)`가 표시됩니다:**

```
(venv) C:\myboard>
```

> **중요**: 앞으로 모든 작업은 가상환경이 활성화된 상태에서 진행해야 합니다!

## 3. Django 설치

가상환경이 활성화된 상태에서 Django를 설치합니다:

```bash
pip install django
```

설치가 완료되면 버전을 확인해봅시다:

```bash
django-admin --version
```

**결과 예시:**
```
5.0.0
```

## 4. Django 프로젝트 생성

이제 본격적으로 Django 프로젝트를 만들어봅시다!

```bash
django-admin startproject myboard_project .
```

> **주의**: 마지막에 점(`.`)을 꼭 붙여주세요! 현재 폴더에 프로젝트를 생성한다는 의미입니다.

### 생성된 파일 구조

```
myboard/
│
├── venv/                    # 가상환경 폴더
├── myboard_project/         # 프로젝트 설정 폴더
│   ├── __init__.py         # Python 패키지임을 나타냄
│   ├── settings.py         # 프로젝트 설정 파일
│   ├── urls.py             # URL 라우팅 설정
│   ├── asgi.py             # ASGI 배포용 (고급)
│   └── wsgi.py             # WSGI 배포용 (고급)
└── manage.py                # Django 명령어 도구
```

### 각 파일의 역할

- **manage.py**: Django 프로젝트 관리 명령어를 실행하는 파일
- **settings.py**: 데이터베이스, 앱, 미들웨어 등 프로젝트 설정
- **urls.py**: URL과 뷰를 연결하는 라우팅 설정
- **wsgi.py, asgi.py**: 배포 시 사용 (지금은 신경 쓰지 않아도 됨)

## 5. 개발 서버 실행

프로젝트가 제대로 생성되었는지 확인해봅시다:

```bash
python manage.py runserver
```

**다음과 같은 메시지가 나타납니다:**

```
Watching for file changes with StatReloader
Performing system checks...

System check identified no issues (0 silenced).

You have 18 unapplied migration(s). Your project may not work properly until you apply the migrations for app(s): admin, auth, contenttypes, sessions.
Run 'python manage.py migrate' to apply them.

December 30, 2025 - 10:00:00
Django version 5.0.0, using settings 'myboard_project.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

> **migration 경고**: 지금은 무시해도 됩니다. Part 2에서 해결할 예정입니다.

웹 브라우저를 열고 `http://127.0.0.1:8000/` 또는 `http://localhost:8000/`으로 접속하세요.

**성공하면 Django 로켓 화면이 나타납니다!** 🚀

서버를 중지하려면 터미널에서 `Ctrl + C`를 누르세요.

## 6. Django 앱 생성

Django 프로젝트는 여러 개의 앱으로 구성됩니다. 게시판 기능을 담당할 앱을 만들어봅시다.

```bash
python manage.py startapp board
```

### 생성된 앱 구조

```
myboard/
│
├── board/                   # 새로 생성된 앱 폴더
│   ├── migrations/         # 데이터베이스 마이그레이션 파일
│   ├── __init__.py
│   ├── admin.py            # 관리자 페이지 설정
│   ├── apps.py             # 앱 설정
│   ├── models.py           # 데이터베이스 모델
│   ├── tests.py            # 테스트 코드
│   └── views.py            # 뷰 (비즈니스 로직)
│
├── myboard_project/
└── manage.py
```

## 7. 앱 등록하기

생성한 앱을 Django 프로젝트에 등록해야 합니다.

**myboard_project/settings.py** 파일을 열고 `INSTALLED_APPS` 부분을 찾습니다:

```python
# myboard_project/settings.py

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'board',  # 이 줄을 추가하세요!
]
```

> **중요**: 'board' 뒤에 쉼표(`,`)를 꼭 붙여주세요!

## 8. 한국어 및 시간대 설정

같은 **settings.py** 파일에서 언어와 시간대를 설정합니다:

```python
# myboard_project/settings.py

# 기존 코드를 찾아서 수정하세요

LANGUAGE_CODE = 'ko-kr'  # 'en-us'에서 'ko-kr'로 변경

TIME_ZONE = 'Asia/Seoul'  # 'UTC'에서 'Asia/Seoul'로 변경

USE_I18N = True

USE_TZ = True
```

## 9. 첫 번째 뷰 만들기

이제 실제로 화면에 표시될 내용을 만들어봅시다!

**board/views.py** 파일을 열고 다음 코드를 작성하세요:

```python
# board/views.py

from django.shortcuts import render
from django.http import HttpResponse


def index(request):
    """
    게시판 메인 페이지
    """
    return HttpResponse("안녕하세요! Django 게시판입니다.")
```

### 코드 설명

- `def index(request):`: 뷰 함수를 정의합니다. 모든 뷰 함수는 `request` 파라미터를 받습니다.
- `HttpResponse()`: 간단한 텍스트를 응답으로 반환합니다.

## 10. URL 라우팅 설정

### board 앱에 urls.py 생성

**board** 폴더 안에 **urls.py** 파일을 새로 만들고 다음 내용을 작성하세요:

```python
# board/urls.py

from django.urls import path
from . import views

app_name = 'board'

urlpatterns = [
    path('', views.index, name='index'),
]
```

### 코드 설명

- `from . import views`: 현재 폴더(board)의 views.py를 import 합니다.
- `app_name = 'board'`: 나중에 URL을 참조할 때 사용할 네임스페이스입니다.
- `path('', views.index, name='index')`: 빈 경로('')로 접속하면 `views.index` 함수를 실행합니다.

### 프로젝트 urls.py에 연결

**myboard_project/urls.py** 파일을 열고 다음과 같이 수정하세요:

```python
# myboard_project/urls.py

from django.contrib import admin
from django.urls import path, include  # include 추가

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('board.urls')),  # 이 줄을 추가하세요!
]
```

### 코드 설명

- `include('board.urls')`: board 앱의 urls.py를 포함합니다.
- `path('', include('board.urls'))`: 루트 경로('')로 오는 요청을 board 앱으로 전달합니다.

## 11. 첫 페이지 확인하기

모든 설정이 완료되었습니다! 서버를 실행해봅시다:

```bash
python manage.py runserver
```

브라우저에서 `http://127.0.0.1:8000/`으로 접속하세요.

**"안녕하세요! Django 게시판입니다."** 라는 메시지가 나타나면 성공입니다! 🎉

## 12. 템플릿으로 업그레이드하기

간단한 텍스트 대신 HTML 페이지를 만들어봅시다.

### templates 폴더 생성

다음과 같은 폴더 구조를 만드세요:

```
board/
├── templates/
│   └── board/
│       └── index.html
```

명령어:
```bash
# Windows
mkdir board\templates\board

# Mac/Linux
mkdir -p board/templates/board
```

### index.html 생성

**board/templates/board/index.html** 파일을 만들고 다음 내용을 작성하세요:

```html
<!-- board/templates/board/index.html -->

<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Django 게시판</title>
</head>
<body>
    <h1>Django 게시판에 오신 것을 환영합니다!</h1>
    <p>이곳은 Django로 만든 게시판입니다.</p>
    <p>앞으로 멋진 기능들을 추가할 예정입니다.</p>
</body>
</html>
```

### views.py 수정

**board/views.py**를 다음과 같이 수정하세요:

```python
# board/views.py

from django.shortcuts import render


def index(request):
    """
    게시판 메인 페이지
    """
    return render(request, 'board/index.html')
```

### 코드 설명

- `render(request, 'board/index.html')`: 템플릿 파일을 렌더링하여 응답으로 반환합니다.
- Django는 자동으로 각 앱의 `templates` 폴더에서 템플릿을 찾습니다.

## 13. 템플릿 폴더 위치 옵션

Django에서 템플릿을 관리하는 방법은 크게 세 가지가 있습니다.

### 방법 1: 앱별 templates 폴더 (현재 사용 중)

```
board/
├── templates/
│   └── board/
│       └── index.html
```

**장점:**
- 앱을 다른 프로젝트로 쉽게 이동 가능
- 앱이 독립적으로 관리됨

**사용 방법:**
```python
# board/views.py
return render(request, 'board/index.html')
```

### 방법 2: 프로젝트 레벨 templates 폴더

프로젝트 전체에서 공유하는 템플릿(예: base.html, 404.html 등)을 위한 폴더입니다.

#### 2-1. templates 폴더 생성

프로젝트 루트에 templates 폴더를 만듭니다:

```
myboard/
├── templates/           # 프로젝트 레벨 templates
│   ├── base.html
│   └── 404.html
├── board/
│   └── templates/
│       └── board/
│           └── index.html
├── myboard_project/
└── manage.py
```

명령어:
```bash
# Windows
mkdir templates

# Mac/Linux
mkdir templates
```

#### 2-2. settings.py 설정

**myboard_project/settings.py**를 열고 `TEMPLATES` 설정을 수정합니다:

```python
# myboard_project/settings.py

import os  # 파일 최상단에 추가

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],  # 이 줄을 수정!
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```

**변경 사항:**
- `'DIRS': []` → `'DIRS': [os.path.join(BASE_DIR, 'templates')]`

**사용 방법:**
```python
# board/views.py
return render(request, 'base.html')  # 프로젝트 레벨 템플릿 사용
```

### 방법 3: 혼합 사용 (권장)

프로젝트 레벨과 앱 레벨을 함께 사용하는 방법입니다.

```
myboard/
├── templates/              # 공통 템플릿
│   ├── base.html          # 모든 페이지의 기본 레이아웃
│   ├── navbar.html        # 공통 네비게이션 바
│   └── footer.html        # 공통 푸터
├── board/
│   └── templates/
│       └── board/         # board 앱 전용 템플릿
│           ├── index.html
│           ├── detail.html
│           └── list.html
├── myboard_project/
└── manage.py
```

**사용 예시:**

프로젝트 레벨 base.html:
```html
<!-- templates/base.html -->

<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Django 게시판{% endblock %}</title>
</head>
<body>
    <nav>
        <h1>Django 게시판</h1>
    </nav>

    <main>
        {% block content %}
        {% endblock %}
    </main>

    <footer>
        <p>&copy; 2024 Django 게시판</p>
    </footer>
</body>
</html>
```

앱 레벨 index.html (base.html 상속):
```html
<!-- board/templates/board/index.html -->

{% extends 'base.html' %}

{% block title %}게시판 홈{% endblock %}

{% block content %}
    <h2>Django 게시판에 오신 것을 환영합니다!</h2>
    <p>이곳은 Django로 만든 게시판입니다.</p>
    <p>앞으로 멋진 기능들을 추가할 예정입니다.</p>
{% endblock %}
```

### 템플릿 검색 순서

Django는 다음 순서로 템플릿을 찾습니다:

1. **DIRS에 지정된 폴더** (프로젝트 레벨 templates)
2. **각 앱의 templates 폴더** (INSTALLED_APPS 순서대로)

예를 들어, `render(request, 'board/index.html')`을 호출하면:
1. 먼저 `templates/board/index.html` 확인
2. 없으면 `board/templates/board/index.html` 확인

### 언제 어떤 방법을 사용할까?

| 템플릿 종류 | 위치 | 예시 |
|----------|------|-----|
| 전체 사이트 공통 | 프로젝트 레벨 | base.html, 404.html, 500.html |
| 앱 전용 | 앱 레벨 | board/index.html, board/detail.html |
| 여러 앱에서 공유 | 프로젝트 레벨 | navbar.html, footer.html |

> **권장 사항**: 작은 프로젝트는 앱 레벨만 사용해도 충분합니다. 프로젝트가 커지면 공통 템플릿을 프로젝트 레벨로 분리하세요.

### 결과 확인

서버를 재시작하고 (`python manage.py runserver`) 브라우저를 새로고침하세요.

이제 HTML로 꾸며진 페이지가 나타납니다!

## 📝 Part 1 정리

축하합니다! Part 1을 완료했습니다. 다음을 배웠습니다:

✅ Django 설치 및 프로젝트 생성<br>
✅ Django 앱 생성 및 등록<br>
✅ 기본 설정 (언어, 시간대)<br>
✅ 뷰 함수 작성<br>
✅ URL 라우팅 설정<br>
✅ 템플릿을 사용한 HTML 페이지 생성

## 🔍 자주 발생하는 오류

### 1. ModuleNotFoundError: No module named 'django'

**원인**: 가상환경이 활성화되지 않았거나 Django가 설치되지 않았습니다.

**해결**:
```bash
# 가상환경 활성화 (Windows)
venv\Scripts\activate

# Django 설치
pip install django
```

### 2. TemplateDoesNotExist

**원인**: 템플릿 파일 경로가 잘못되었습니다.

**해결**:
- 폴더 구조 확인: `board/templates/board/index.html`
- settings.py의 INSTALLED_APPS에 'board'가 등록되어 있는지 확인

### 3. Page not found (404)

**원인**: URL 설정이 잘못되었습니다.

**해결**:
- board/urls.py가 제대로 작성되었는지 확인
- myboard_project/urls.py에 `include('board.urls')`가 추가되었는지 확인

## 🚀 다음 단계

[Part 2: 모델 및 데이터베이스 설정](./Part2-Models.md)에서 게시글을 저장할 데이터베이스를 만들어봅시다!
