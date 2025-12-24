# 📝 Django 블로그 프로젝트

Django 기반의 블로그 시스템입니다. 게시글, 댓글, 북마크, 좋아요 등의 기능을 제공합니다.

## 📊 데이터베이스 구조

### ERD (Entity Relationship Diagram)

```
┌─────────────────┐
│     User        │
│  (Django 기본)  │
└────────┬────────┘
         │
         │ 1:N (author)
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐   ┌─────────────────┐
│   Category      │   │      Tag        │
├─────────────────┤   ├─────────────────┤
│ name *          │   │ name *          │
│ description     │   │ created_at      │
│ created_at      │   └────────┬────────┘
└────────┬────────┘            │
         │                     │
         │ 1:N                 │ N:M
         │                     │
         ▼                     ▼
┌─────────────────────────────────────┐
│             Post                    │
├─────────────────────────────────────┤
│ title                               │
│ content                             │
│ author (FK → User)                  │
│ category (FK → Category)            │
│ tags (M2M → Tag)                    │
│ created_at                          │
│ updated_at                          │
│ published                           │
│ views                               │
└───────┬─────────────────────────────┘
        │
        │ 1:N
        ├──────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Comment    │ │   Bookmark   │ │     Like     │ │     ...      │
├──────────────┤ ├──────────────┤ ├──────────────┤ └──────────────┘
│ post (FK)    │ │ post (FK)    │ │ post (FK)    │
│ author (FK)  │ │ user (FK)    │ │ user (FK)    │
│ content      │ │ created_at   │ │ created_at   │
│ created_at   │ │ unique: (p,u)│ │ unique: (p,u)│
│ updated_at   │ └──────────────┘ └──────────────┘
└──────────────┘
```

## 🗂️ 모델 구조

### 👤 User (Django 기본 모델)
Django의 기본 인증 모델을 사용합니다.

**관계:**
- Post와 1:N 관계 (작성자)
- Comment와 1:N 관계 (댓글 작성자)
- Bookmark와 1:N 관계
- Like와 1:N 관계

---

### 📁 Category (카테고리)
게시글을 분류하기 위한 카테고리 모델입니다.

**필드:**
- `name`: 카테고리 이름 (고유값)
- `description`: 카테고리 설명
- `created_at`: 생성일시

**특징:**
- 카테고리명은 중복 불가 (`unique=True`)
- 최신순 정렬 (`-created_at`)

**역참조:**
```python
category.posts.all()  # 해당 카테고리의 모든 게시글
```

---

### 🏷️ Tag (태그)
게시글에 부여할 수 있는 태그 모델입니다.

**필드:**
- `name`: 태그 이름 (고유값)
- `created_at`: 생성일시

**특징:**
- 태그명은 중복 불가 (`unique=True`)
- Post와 다대다(N:M) 관계

**역참조:**
```python
tag.posts.all()  # 해당 태그가 붙은 모든 게시글
```

---

### 📝 Post (게시글) - 핵심 모델
블로그의 핵심이 되는 게시글 모델입니다.

**필드:**
- `title`: 제목 (최대 200자)
- `content`: 본문 내용
- `author`: 작성자 (User FK)
- `category`: 카테고리 (Category FK, optional)
- `tags`: 태그 목록 (Tag M2M)
- `created_at`: 작성일시 (자동 생성)
- `updated_at`: 수정일시 (자동 갱신)
- `published`: 발행 여부 (Boolean)
- `views`: 조회수

**관계:**
- User와 N:1 관계 (`author`)
- Category와 N:1 관계 (`category`, nullable)
- Tag와 N:M 관계 (`tags`)
- Comment, Bookmark, Like와 1:N 관계

**역참조:**
```python
post.comments.all()    # 게시글의 모든 댓글
post.bookmarks.count() # 북마크 개수
post.likes.count()     # 좋아요 개수
```

---

### 💬 Comment (댓글)
게시글에 달리는 댓글 모델입니다.

**필드:**
- `post`: 게시글 (Post FK)
- `author`: 작성자 (User FK)
- `content`: 댓글 내용
- `created_at`: 작성일시
- `updated_at`: 수정일시

**특징:**
- 최신순 정렬 (`-created_at`)
- 게시글 또는 작성자 삭제 시 함께 삭제 (`CASCADE`)

---

### 🔖 Bookmark (북마크)
사용자가 게시글을 저장하는 북마크 기능입니다.

**필드:**
- `post`: 게시글 (Post FK)
- `user`: 사용자 (User FK)
- `created_at`: 북마크 생성일시

**특징:**
- `unique_together`: (post, user) - 한 사용자당 같은 게시글에 1번만 북마크 가능
- 중복 북마크 방지

**사용 예시:**
```python
# 북마크 추가 (중복 체크)
bookmark, created = Bookmark.objects.get_or_create(post=post, user=user)
```

---

### ❤️ Like (좋아요)
게시글에 대한 좋아요 기능입니다.

**필드:**
- `post`: 게시글 (Post FK)
- `user`: 사용자 (User FK)
- `created_at`: 좋아요 생성일시

**특징:**
- `unique_together`: (post, user) - 한 사용자당 같은 게시글에 1번만 좋아요 가능
- 중복 좋아요 방지

**사용 예시:**
```python
# 좋아요 토글
like, created = Like.objects.get_or_create(post=post, user=user)
if not created:
    like.delete()  # 이미 좋아요한 경우 취소
```

---

## 🔗 관계 요약

| 관계 | 타입 | 설명 |
|------|------|------|
| User → Post | 1:N | 한 유저가 여러 글 작성 |
| User → Comment | 1:N | 한 유저가 여러 댓글 작성 |
| User → Bookmark | 1:N | 한 유저가 여러 북마크 |
| User → Like | 1:N | 한 유저가 여러 좋아요 |
| Category → Post | 1:N | 한 카테고리에 여러 글 |
| Tag ↔ Post | N:M | 태그와 글은 다대다 관계 |
| Post → Comment | 1:N | 한 글에 여러 댓글 |
| Post → Bookmark | 1:N | 한 글에 여러 북마크 |
| Post → Like | 1:N | 한 글에 여러 좋아요 |

## 📋 역참조(Reverse Lookup) 가이드

| 시작 모델 | 관련 모델 | 역참조 이름 | 사용 예시 |
|-----------|----------|------------|-----------|
| User | Post | `posts` | `user.posts.all()` |
| User | Comment | `comments` | `user.comments.all()` |
| User | Bookmark | `bookmarks` | `user.bookmarks.all()` |
| User | Like | `likes` | `user.likes.all()` |
| Category | Post | `posts` | `category.posts.all()` |
| Tag | Post | `posts` | `tag.posts.all()` |
| Post | Comment | `comments` | `post.comments.all()` |
| Post | Bookmark | `bookmarks` | `post.bookmarks.all()` |
| Post | Like | `likes` | `post.likes.all()` |

## 🎯 주요 기능 및 제약사항

### 🗑️ 삭제 정책 (on_delete)

#### CASCADE (연쇄 삭제)
- **User 삭제** → 해당 유저의 모든 Post, Comment, Bookmark, Like 삭제
- **Post 삭제** → 해당 게시글의 모든 Comment, Bookmark, Like 삭제

#### SET_NULL (NULL 설정)
- **Category 삭제** → 해당 카테고리의 Post들의 category 필드가 NULL로 설정

### 🔒 UNIQUE 제약 조건

| 모델 | 제약 | 설명 |
|------|------|------|
| Category | `name` | 카테고리명 중복 불가 |
| Tag | `name` | 태그명 중복 불가 |
| Bookmark | `(post, user)` | 한 유저당 게시글 하나에 북마크 1번만 |
| Like | `(post, user)` | 한 유저당 게시글 하나에 좋아요 1번만 |

### 📅 정렬(Ordering)

모든 주요 모델은 최신순으로 정렬됩니다:
- Category: `-created_at` (최신 카테고리가 먼저)
- Post: `-created_at` (최신 게시글이 먼저)
- Comment: `-created_at` (최신 댓글이 먼저)

## 💻 사용 예시

### 게시글 작성
```python
# 게시글 생성
post = Post.objects.create(
    title="Django 튜토리얼",
    content="Django는 파이썬 웹 프레임워크입니다.",
    author=user,
    category=category,
    published=True
)

# 태그 추가
post.tags.add(tag1, tag2, tag3)
```

### 데이터 조회
```python
# 유저가 작성한 모든 글
user.posts.all()

# 카테고리의 발행된 글만
category.posts.filter(published=True)

# 특정 태그가 붙은 글들
tag.posts.all()

# 게시글의 댓글 최신순
post.comments.all()

# 게시글의 좋아요 개수
post.likes.count()

# 유저가 좋아요한 글들
Post.objects.filter(likes__user=user)
```

### 댓글 작성
```python
Comment.objects.create(
    post=post,
    author=user,
    content="좋은 글 감사합니다!"
)
```

### 북마크/좋아요 토글
```python
# 북마크 추가 (중복 방지)
bookmark, created = Bookmark.objects.get_or_create(
    post=post,
    user=user
)

# 좋아요 토글
like, created = Like.objects.get_or_create(
    post=post,
    user=user
)
if not created:
    like.delete()  # 좋아요 취소
```

### 복잡한 쿼리
```python
from django.db.models import Count, Avg

# 인기 게시글 (좋아요 10개 이상)
popular_posts = Post.objects.annotate(
    like_count=Count('likes')
).filter(like_count__gte=10)

# 댓글이 많은 순으로 정렬
Post.objects.annotate(
    comment_count=Count('comments')
).order_by('-comment_count')

# 특정 태그의 평균 조회수
tag.posts.aggregate(Avg('views'))
```

## 🚀 설치 및 실행

```bash
# 마이그레이션 생성
python manage.py makemigrations

# 마이그레이션 적용
python manage.py migrate

# 관리자 계정 생성
python manage.py createsuperuser

# 개발 서버 실행
python manage.py runserver
```

## 📌 주의사항

1. **User 모델**: Django의 기본 User 모델을 사용합니다. 커스터마이징이 필요한 경우 `AbstractUser`를 상속받아 확장하세요.

2. **CASCADE 삭제**: User나 Post 삭제 시 관련된 모든 데이터가 삭제되므로 주의가 필요합니다.

3. **중복 방지**: Bookmark와 Like는 `unique_together` 제약으로 중복을 방지하므로, `get_or_create()`를 사용하는 것이 좋습니다.

4. **조회수**: Post의 `views` 필드는 자동으로 증가하지 않으므로, 뷰에서 직접 증가시켜야 합니다.

5. **발행 상태**: Post의 `published` 필드를 통해 초안/발행 상태를 관리할 수 있습니다.

## 📝 License

이 프로젝트는 [MIT 라이센스](LICENSE) 하에 배포됩니다.