1️⃣ Django 프로젝트 구조 예시
```
myproject/
├─ manage.py
├─ myproject/
│  ├─ __init__.py
│  ├─ settings.py
│  ├─ urls.py
│  ├─ asgi.py
│  └─ wsgi.py
└─ myapp/
    ├─ __init__.py
    ├─ admin.py
    ├─ apps.py
    ├─ models.py
    ├─ views.py
    ├─ urls.py
    ├─ forms.py
    ├─ tests.py
    └─ migrations/
```

2️⃣ 각 파일과 폴더 역할

- 핵심포인트
    - models.py → DB 구조
    - forms.py → 폼 + 유효성 검사 + HTML 위젯
    - views.py → 요청 처리 + 템플릿 렌더링
    - urls.py → URL과 view 연결
    - admin.py → 관리자 화면 커스터마이징
    - settings.py → 프로젝트 전체 환경 설정

<br>
<br>
# 프로젝트 레벨

| 파일/폴더                   | 역할                                                           |
| ----------------------- | ------------------------------------------------------------ |
| `manage.py`             | Django 명령어 실행 도구. 서버 실행(`runserver`), DB 마이그레이션(`migrate`) 등 |
| `myproject/__init__.py` | Python 패키지임을 표시                                              |
| `myproject/settings.py` | 프로젝트 전체 설정: DB, 앱, 미들웨어, 정적 파일 경로, 보안 키 등                    |
| `myproject/urls.py`     | 전체 URL 라우팅 설정. 각 앱의 urls.py와 연결                              |
| `myproject/asgi.py`     | ASGI 서버 설정 (비동기 서버용)                                         |
| `myproject/wsgi.py`     | WSGI 서버 설정 (동기 서버용, 배포 시 사용)                                 |

<br>
<br>
# 앱(app) : Django에서 앱은 기능 단위를 의미. 예: 게시판, 블로그, 쇼핑몰 장바구니

| 파일            | 역할                                 |
| ------------- | ---------------------------------- |
| `__init__.py` | Python 패키지 표시                      |
| `admin.py`    | 관리자(admin) 페이지에 모델 등록 및 설정         |
| `apps.py`     | 앱 설정 (앱 이름, 레이블 등)                 |
| `models.py`   | DB 모델 정의 (테이블 구조)                  |
| `views.py`    | HTTP 요청 처리 로직 (사용자 요청 → 응답)        |
| `urls.py`     | 앱 내부 URL 라우팅 정의                    |
| `forms.py`    | 사용자 입력 폼 정의, 유효성 검사, HTML 위젯 설정    |
| `tests.py`    | 단위 테스트 작성                          |
| `migrations/` | 모델 변경 사항을 DB에 반영하기 위한 마이그레이션 파일 저장 |


<br>
<br>
# 추가적으로 자주 쓰는 폴더/파일

| 파일/폴더                   | 역할                      |
| ----------------------- | ----------------------- |
| `templates/`            | HTML 템플릿 저장             |
| `static/`               | CSS, JS, 이미지 등 정적 파일 저장 |
| `media/`                | 업로드 파일 저장               |
| `context_processors.py` | 모든 템플릿에 공통 데이터 전달용 함수   |
| `signals.py`            | 모델 이벤트 발생 시 동작 정의       |



3️⃣ 데이터 흐름
```
     사용자 브라우저
            ↓ (HTTP 요청)
         views.py
   ┌─────────────────────────┐
   │ 1. 모델에서 데이터 조회 │
   │ 2. 폼으로 입력 검증     │
   │ 3. 템플릿에 데이터 전달 │
   └─────────────────────────┘
            ↓ (render)
    템플릿(post_detail.html)
            ↓
    브라우저에 최종 HTML 표시
```