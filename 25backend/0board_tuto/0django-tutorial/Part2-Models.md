# Part 2: 모델 및 데이터베이스 설정

## 📌 이번 Part에서 배울 내용

- Django 모델(Model) 이해하기
- 게시글 모델 설계 및 생성
- 데이터베이스 마이그레이션
- Django Admin 페이지 설정
- 테스트 데이터 추가하기

## 1. Django 모델이란?

모델은 데이터베이스의 구조를 정의하는 Python 클래스입니다. 게시판을 만들려면 게시글 데이터를 저장할 테이블이 필요한데, Django 모델로 이를 쉽게 만들 수 있습니다.

### 게시글에 필요한 정보

우리가 만들 게시글은 다음 정보를 담아야 합니다:

- **제목** (title)
- **내용** (content)
- **작성자** (author)
- **작성일시** (created_at)
- **수정일시** (updated_at)
- **조회수** (views)

## 2. Post 모델 생성하기

**board/models.py** 파일을 열고 다음 코드를 작성하세요:

```python
# board/models.py

from django.db import models
from django.contrib.auth.models import User


class Post(models.Model):
    """
    게시글 모델
    """
    title = models.CharField(max_length=200, verbose_name='제목')
    content = models.TextField(verbose_name='내용')
    author = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='작성자')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='작성일시')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='수정일시')
    views = models.PositiveIntegerField(default=0, verbose_name='조회수')

    class Meta:
        ordering = ['-created_at']  # 최신 글이 위로 오도록 정렬
        verbose_name = '게시글'
        verbose_name_plural = '게시글'

    def __str__(self):
        return self.title
```

### 코드 상세 설명

#### 필드 타입

- **CharField**: 짧은 문자열 (제목)
  - `max_length=200`: 최대 200자까지 저장

- **TextField**: 긴 텍스트 (내용)
  - 길이 제한이 없어서 긴 글을 쓸 수 있습니다

- **ForeignKey**: 다른 모델과의 관계 (작성자)
  - `User`: Django가 기본으로 제공하는 사용자 모델
  - `on_delete=models.CASCADE`: 사용자가 삭제되면 그 사용자의 게시글도 모두 삭제

- **DateTimeField**: 날짜와 시간
  - `auto_now_add=True`: 생성될 때 자동으로 현재 시간 저장
  - `auto_now=True`: 수정될 때마다 자동으로 현재 시간으로 업데이트

- **PositiveIntegerField**: 양의 정수 (조회수)
  - `default=0`: 기본값 0

#### verbose_name

관리자 페이지에서 표시될 필드 이름입니다.

#### Meta 클래스

- `ordering = ['-created_at']`: 데이터를 가져올 때 작성일시 기준 내림차순 정렬
  - `-`(마이너스)는 내림차순을 의미합니다
  - 최신 글이 먼저 나타납니다

- `verbose_name`: 관리자 페이지에서 모델 이름 (단수)
- `verbose_name_plural`: 관리자 페이지에서 모델 이름 (복수)

#### __str__ 메서드

객체를 문자열로 표현할 때 사용됩니다. 게시글의 제목을 반환합니다.

## 3. 데이터베이스 마이그레이션

모델을 만들었으면 실제 데이터베이스에 테이블을 생성해야 합니다.

### 마이그레이션 파일 생성

```bash
python manage.py makemigrations
```

**결과:**
```
Migrations for 'board':
  board\migrations\0001_initial.py
    - Create model Post
```

이 명령어는 모델의 변경사항을 기록한 마이그레이션 파일을 생성합니다.

### 마이그레이션 적용

```bash
python manage.py migrate
```

**결과:**
```
Operations to perform:
  Apply all migrations: admin, auth, board, contenttypes, sessions
Running migrations:
  Applying board.0001_initial... OK
  ...
```

이제 실제 데이터베이스에 테이블이 생성되었습니다!

### 마이그레이션이란?

- **makemigrations**: 모델 변경사항을 기록 (설계도 작성)
- **migrate**: 기록된 변경사항을 데이터베이스에 적용 (실제 공사)

> **중요**: 모델을 수정할 때마다 `makemigrations`와 `migrate`를 실행해야 합니다!

## 4. Django Admin 설정

Django는 강력한 관리자 페이지를 기본으로 제공합니다. 여기서 게시글을 쉽게 관리할 수 있습니다.

### 관리자 계정 생성

```bash
python manage.py createsuperuser
```

다음 정보를 입력하세요:

```
사용자 이름 (leave blank to use 'yourname'): admin
이메일 주소: admin@example.com
Password: ********
Password (again): ********
```

> **주의**:
> - 비밀번호는 화면에 표시되지 않습니다 (보안)
> - 간단한 비밀번호는 경고가 나오지만 개발용이므로 'y'를 입력하여 계속 진행해도 됩니다
> - **비밀번호를 꼭 기억하세요!**

### Post 모델을 Admin에 등록

**board/admin.py** 파일을 열고 다음 코드를 작성하세요:

```python
# board/admin.py

from django.contrib import admin
from .models import Post


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    """
    게시글 관리자 페이지 설정
    """
    list_display = ['title', 'author', 'created_at', 'views']  # 목록에 표시될 필드
    list_filter = ['created_at', 'author']  # 필터 옵션
    search_fields = ['title', 'content']  # 검색 가능한 필드
    readonly_fields = ['created_at', 'updated_at', 'views']  # 읽기 전용 필드
    date_hierarchy = 'created_at'  # 날짜 기반 네비게이션
```

### 코드 설명

- `@admin.register(Post)`: Post 모델을 관리자 페이지에 등록
- `list_display`: 목록 화면에서 보여줄 필드들
- `list_filter`: 오른쪽에 필터 사이드바 표시
- `search_fields`: 검색 기능 추가
- `readonly_fields`: 수정할 수 없는 필드 (자동 생성 필드)
- `date_hierarchy`: 날짜별로 필터링할 수 있는 네비게이션

### Admin 페이지 접속

서버를 실행하고:

```bash
python manage.py runserver
```

브라우저에서 `http://127.0.0.1:8000/admin/`으로 접속하세요.

앞서 만든 관리자 계정으로 로그인합니다.

**좌측 메뉴에서 "게시글"을 클릭하면 게시글 관리 화면이 나타납니다!**

## 5. 테스트 데이터 추가하기

Admin 페이지에서 직접 게시글을 추가해봅시다.

### 게시글 작성

1. Admin 페이지에서 "게시글" 옆의 **"추가"** 버튼 클릭
2. 다음 정보를 입력:
   - **제목**: "첫 번째 게시글입니다"
   - **내용**: "Django로 만든 첫 게시글입니다. 반갑습니다!"
   - **작성자**: admin (자동으로 선택됨)
3. **저장** 버튼 클릭

이런 식으로 3~5개의 테스트 게시글을 작성하세요.

### 예시 게시글

**게시글 1:**
- 제목: Django 시작하기
- 내용: Django는 Python 기반의 웹 프레임워크입니다.

**게시글 2:**
- 제목: 게시판 만들기
- 내용: 게시판 기능을 하나씩 구현하고 있습니다.

**게시글 3:**
- 제목: 모델 이해하기
- 내용: 모델은 데이터베이스의 구조를 정의합니다.

## 6. Django Shell에서 데이터 다루기

Python 코드로 직접 데이터를 다뤄볼 수도 있습니다.

### Django Shell 실행

```bash
python manage.py shell
```

### 게시글 조회

```python
# Post 모델 import
from board.models import Post

# 모든 게시글 가져오기
posts = Post.objects.all()
print(posts)

# 게시글 개수
print(Post.objects.count())

# 첫 번째 게시글
first_post = Post.objects.first()
print(first_post.title)
print(first_post.content)
print(first_post.author)

# 특정 게시글 검색
post = Post.objects.get(id=1)
print(post.title)
```

### 게시글 생성

```python
from board.models import Post
from django.contrib.auth.models import User

# 사용자 가져오기
user = User.objects.get(username='admin')

# 새 게시글 생성
post = Post.objects.create(
    title='Shell에서 만든 게시글',
    content='Django Shell을 사용하여 게시글을 만들었습니다.',
    author=user
)

print(f'게시글이 생성되었습니다: {post.title}')
```

### 게시글 수정

```python
# 게시글 가져오기
post = Post.objects.get(id=1)

# 제목 수정
post.title = '수정된 제목'
post.save()

print('게시글이 수정되었습니다.')
```

### 게시글 삭제

```python
# 게시글 가져오기
post = Post.objects.get(id=1)

# 삭제
post.delete()

print('게시글이 삭제되었습니다.')
```

### Shell 종료

```python
exit()
```

## 7. QuerySet 이해하기

Django ORM(Object-Relational Mapping)을 사용하면 SQL을 직접 작성하지 않고도 데이터베이스를 다룰 수 있습니다.

### 자주 사용하는 메서드

```python
# 모든 객체 가져오기
Post.objects.all()

# 필터링 (조건에 맞는 것만)
Post.objects.filter(author=user)
Post.objects.filter(title__contains='Django')  # 제목에 'Django'가 포함된 글

# 제외 (조건에 맞지 않는 것만)
Post.objects.exclude(author=user)

# 하나만 가져오기 (없으면 에러)
Post.objects.get(id=1)

# 첫 번째 객체
Post.objects.first()

# 마지막 객체
Post.objects.last()

# 개수 세기
Post.objects.count()

# 존재 여부 확인
Post.objects.filter(title='Django').exists()

# 정렬
Post.objects.order_by('-created_at')  # 최신순
Post.objects.order_by('title')  # 제목 가나다순

# 특정 필드만 가져오기
Post.objects.values('title', 'author')

# 슬라이싱 (처음 5개)
Post.objects.all()[:5]
```

## 8. 모델에 메서드 추가하기

모델에 유용한 메서드를 추가할 수 있습니다.

**board/models.py**를 다음과 같이 수정하세요:

```python
# board/models.py

from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse


class Post(models.Model):
    """
    게시글 모델
    """
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
        """수정 여부 확인 (작성 후 1분 이상 지나고 수정된 경우)"""
        return (self.updated_at - self.created_at).total_seconds() > 60
```

### 추가된 메서드 설명

- **get_absolute_url()**: 게시글의 URL을 반환 (나중에 사용)
- **increase_views()**: 조회수를 1 증가시킵니다
- **is_updated()**: 게시글이 수정되었는지 확인합니다

## 📝 Part 2 정리

축하합니다! Part 2를 완료했습니다. 다음을 배웠습니다:

✅ Django 모델 생성 및 필드 타입<br>
✅ 데이터베이스 마이그레이션<br>
✅ Django Admin 설정 및 사용<br>
✅ QuerySet을 사용한 데이터 조회<br>
✅ Django Shell 사용법<br>
✅ 모델 메서드 추가

## 🔍 자주 발생하는 오류

### 1. No changes detected

**원인**: 모델을 수정하지 않았거나 변경사항이 없습니다.

**해결**: 모델 파일이 제대로 저장되었는지 확인하세요.

### 2. Migrations are conflicting

**원인**: 마이그레이션 충돌이 발생했습니다.

**해결**:
```bash
# 모든 마이그레이션 삭제 후 재생성 (개발 초기에만!)
python manage.py migrate board zero
python manage.py makemigrations
python manage.py migrate
```

### 3. FOREIGN KEY constraint failed

**원인**: 존재하지 않는 사용자를 참조하려고 했습니다.

**해결**: 올바른 사용자 객체를 사용하세요.

```python
from django.contrib.auth.models import User
user = User.objects.get(username='admin')
```

## 💡 추가 학습

### 다양한 필드 타입

Django는 다양한 필드 타입을 제공합니다:

- `IntegerField`: 정수
- `FloatField`: 실수
- `BooleanField`: 참/거짓
- `DateField`: 날짜만
- `TimeField`: 시간만
- `EmailField`: 이메일
- `URLField`: URL
- `ImageField`: 이미지 파일
- `FileField`: 일반 파일

### 필드 옵션

- `null=True`: 데이터베이스에서 NULL 허용
- `blank=True`: 폼에서 빈 값 허용
- `unique=True`: 중복 불가
- `choices`: 선택 옵션 제공
- `help_text`: 도움말 텍스트

## 🚀 다음 단계

[Part 3: 게시글 목록 및 상세 보기](./Part3-ListView.md)에서 게시글을 화면에 표시해봅시다!
