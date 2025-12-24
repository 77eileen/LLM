# Django Q&A 프로젝트 상세 설명 (README)

## 1. 프로젝트 개요

이 프로젝트는 Django 프레임워크를 사용하여 개발된 간단한 질문 및 답변(Q&A) 웹 애플리케이션입니다. 사용자는 질문을 게시하고 다른 사용자의 질문에 답변할 수 있습니다. Django의 기본적인 MTV(Model-Template-View) 아키텍처를 학습하고 이해하는 것을 목표로 합니다.

## 2. 주요 기능

*   **질문 목록 보기**: 모든 질문을 최신순으로 정렬하여 보여줍니다.
*   **질문 상세 보기**: 특정 질문을 클릭하면 해당 질문의 상세 내용과 함께 등록된 모든 답변을 볼 수 있습니다.
*   **답변 작성**: 질문 상세 페이지에서 새로운 답변을 작성하고 등록할 수 있습니다.
*   **관리자 페이지**: Django Admin을 통해 질문과 답변 데이터를 쉽게 관리(생성, 수정, 삭제)할 수 있습니다.

## 3. 프로젝트 구조 및 파일 상세 설명

```
C:/python_src/8.Backend/251223/Django/
├── manage.py           # Django 프로젝트 관리 스크립트
├── db.sqlite3          # SQLite 데이터베이스 파일
├── mysite/             # 1. 프로젝트 설정 디렉터리
│   ├── settings.py     # 프로젝트 전반의 설정 (앱 등록, DB, 정적 파일 등)
│   └── urls.py         # 최상위 URL 라우팅
├── blog/               # 2. Q&A 기능 앱 디렉터리
│   ├── models.py       # 데이터 모델 (Question, Answer) 정의
│   ├── views.py        # 요청 처리 로직 (index, detail, answer_create)
│   ├── urls.py         # 앱 내부의 URL 라우팅
│   ├── admin.py        # 관리자 페이지에 모델 등록
│   └── migrations/
├── templates/          # 3. HTML 템플릿 디렉터리
│   └── blog/
│       ├── question_list.html    # 질문 목록 템플릿
│       └── question_detail.html  # 질문 상세 및 답변 템플릿
├── static/             # 4. CSS, JS 등 정적 파일 디렉터리
│   └── style.css
└── README.md           # 프로젝트 설명 파일 (현재 파일)
```

### 루트 디렉터리 (`Django/`)
*   `manage.py`: `runserver`, `makemigrations` 등 Django 프로젝트 관리 명령을 실행하는 유틸리티입니다.
*   `db.sqlite3`: 모든 데이터(질문, 답변 등)가 저장되는 SQLite 데이터베이스 파일입니다.

### 1. `mysite/` (프로젝트 설정)
*   `settings.py`: 프로젝트의 모든 설정을 관리합니다. `blog` 앱 등록, 데이터베이스 연결, 템플릿 및 정적 파일 경로 지정 등이 이곳에서 이루어집니다.
*   `urls.py`: 프로젝트의 전체 URL을 관리합니다. `/blog/`로 들어오는 요청을 `blog` 앱의 `urls.py`로 넘겨주는 역할을 합니다.

### 2. `blog/` (Q&A 앱)
*   `models.py`: 데이터의 구조를 정의합니다. `Question`(질문)과 `Answer`(답변) 클래스가 있으며, 답변은 질문에 종속됩니다.
*   `views.py`: 실제 프로그램 로직이 동작하는 부분입니다.
    *   `index`: 질문 목록을 DB에서 조회하여 페이지를 만듭니다.
    *   `detail`: 특정 질문과 그에 달린 답변들을 DB에서 조회하여 상세 페이지를 만듭니다.
    *   `answer_create`: 사용자가 작성한 답변을 DB에 저장합니다.
*   `urls.py`: `blog` 앱 내부의 URL을 관리합니다. 예를 들어, `blog/3/`과 같은 URL이 `detail` 뷰에 연결되도록 설정합니다.
*   `admin.py`: 정의된 모델을 관리자 페이지에서 다룰 수 있도록 등록합니다.
*   `migrations/`: 모델의 변경 이력을 저장하여 데이터베이스 스키마를 관리합니다.

### 3. `templates/blog/` (템플릿)
*   `question_list.html`: `views.py`에서 전달받은 질문 목록을 HTML로 표시합니다.
*   `question_detail.html`: 질문 상세 내용과 답변 목록을 표시하고, 새 답변을 작성할 폼을 제공합니다.

### 4. `static/` (정적 파일)
*   `style.css`: 웹 페이지의 디자인을 담당하는 CSS 파일입니다.

## 4. 설치 및 실행 방법

### 4.1. 개발 환경 설정
```bash
# Django 설치
conda install django -y

# 프로젝트 및 앱 생성
django-admin startproject mysite .
django-admin startapp blog
```

### 4.2. 데이터베이스 설정 및 관리자 계정 생성
```bash
# 데이터베이스 스키마(구조) 적용
python manage.py makemigrations
python manage.py migrate

# 관리자 계정 생성
python manage.py createsuperuser
```

### 4.3. 개발 서버 실행
```bash
python manage.py runserver
```
*   **애플리케이션 접속**: `http://127.0.0.1:8000/blog/`
*   **관리자 페이지 접속**: `http://127.0.0.1:8000/admin/`

## 5. 핵심 로직 흐름 (질문 상세 페이지)

1.  **사용자 요청**: 사용자가 브라우저에 `http://127.0.0.1:8000/blog/1/`을 입력합니다.
2.  **URL 라우팅 (mysite)**: `mysite/urls.py`는 `blog/`로 시작하는 URL이므로 요청을 `blog/urls.py`로 전달합니다.
3.  **URL 라우팅 (blog)**: `blog/urls.py`는 `1/` 패턴을 보고, 이 요청을 `detail` 뷰 함수가 처리해야 함을 인지하고, 숫자 `1`을 `question_id` 인자로 하여 `views.py`의 `detail` 함수를 호출합니다.
4.  **뷰 로직 처리 (views)**: `detail` 함수는 `question_id=1`인 `Question` 객체와 그에 연결된 모든 `Answer` 객체들을 데이터베이스에서 조회합니다.
5.  **템플릿 렌더링 (templates)**: `detail` 함수는 조회한 데이터를 `templates/blog/question_detail.html`에 전달하여 HTML을 생성합니다.
6.  **응답**: 완성된 HTML이 사용자의 웹 브라우저로 전송되어 화면에 질문과 답변 내용이 나타납니다.


## 6. Django 구조(MTV패턴)
| Django   | 일반 MVC     | 역할 |
| -------- | ---------- | -- |
| Model    | Model      | DB |
| Template | View       | 화면 |
| View     | Controller | 로직 |

template을 view에 갈아끼움.
기존의 controller (fast api에서 라우터같은..) 가 없음 --> Django는 urls.py의 urlpatterns가 그 기능을 함

```text
브라우저 요청
   ↓
URL
   ↓
View (python 함수 : 요청을 처리하는 파이썬 함수, request 사용자가 )
   ↓
Model (DB)
   ↓
Template (HTML)
   ↓
응답```

```text
웹이 어떻게 동작하는지
        ↓
Django 전체 구조 감 잡기
        ↓
요청 → 응답 흐름 이해
        ↓
DB(Model) 이해
        ↓
화면(Template) 이해
        ↓
URL ↔ View 연결
        ↓
CRUD (게시판 완성)```

```text
- Django는 이걸 한 프레임워크 안에 다 넣어줌
Django
 ├─ View (백엔드)
 ├─ Model (백엔드)
 ├─ Template (프론트 구조)
 └─ Static (프론트 실행)

현대 웹 기준으로 다시 나누면
🔹 전통적인 Django 방식
Django → HTML 렌더링
JS → 최소
👉 Django가 프론트 + 백엔드 다 함

🔹 요즘 실무 방식
Django → API 서버 (백엔드 100%)
프론트 → React / Vue
👉 Django는 완전 백엔드